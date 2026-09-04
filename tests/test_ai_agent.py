"""Unit tests for AI Recovery Agent, Gemini integration mocking, and heuristic fallback."""

import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    ProductCategory,
    RecoveryAction,
    AIRecommendation,
)
from src.ai_agent import (
    AIRecoveryAgent,
    heuristic_fallback_recommend,
    analyze_payment,
)


def sample_payment(
    amount: float = 2499.0,
    failure_reason: FailureReason = FailureReason.NETWORK_TIMEOUT,
    retry_count: int = 0,
    previous_payments: int = 4,
    total_spend: float = 9500.0,
    days_since: int = 15,
) -> PaymentRecord:
    """Helper to create valid PaymentRecord for agent tests."""
    return PaymentRecord(
        payment_id="pay_test_agent_01",
        customer_id="cust_test_agent_01",
        customer_name="Aarav Sharma",
        customer_email="aarav@example.com",
        order_id="order_test_agent_01",
        amount=amount,
        currency="INR",
        payment_status=PaymentStatus.FAILED,
        failure_reason=failure_reason,
        retry_count=retry_count,
        customer_previous_payments=previous_payments,
        customer_previous_failures=0,
        customer_total_spend=total_spend,
        days_since_last_success=days_since if previous_payments > 0 else None,
        product_category=ProductCategory.ECOMMERCE,
    )


def test_1_valid_structured_ai_recommendation():
    """1. Verify valid AIRecommendation model creation and field validation."""
    rec = AIRecommendation(
        action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reason="Network timeout during checkout; payment link allows customer to complete payment.",
        expected_recovery=2499.0,
        risk_level="LOW",
        proposed_discount_pct=0.0,
        requires_approval_recommendation=False,
        agent_mode="gemini",
        model_name="gemini-2.5-flash",
    )
    assert rec.action == RecoveryAction.PAYMENT_LINK
    assert rec.confidence == 0.88
    assert rec.expected_recovery == 2499.0
    assert rec.agent_mode == "gemini"


def test_2_invalid_action_rejected():
    """2. Verify that invalid action values are rejected by schema."""
    with pytest.raises(ValidationError):
        AIRecommendation(
            action="UNAUTHORIZED_ACTION",  # Invalid
            confidence=0.80,
            reason="Test invalid action",
            expected_recovery=1000.0,
            risk_level="LOW",
        )


def test_3_confidence_validation():
    """3. Verify confidence must strictly be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        AIRecommendation(
            action=RecoveryAction.RETRY,
            confidence=1.25,  # Invalid: > 1.0
            reason="High confidence test",
            expected_recovery=1000.0,
        )

    with pytest.raises(ValidationError):
        AIRecommendation(
            action=RecoveryAction.RETRY,
            confidence=-0.1,  # Invalid: < 0.0
            reason="Negative confidence test",
            expected_recovery=1000.0,
        )


def test_4_negative_expected_recovery_rejected():
    """4. Verify expected recovery cannot be negative."""
    with pytest.raises(ValidationError):
        AIRecommendation(
            action=RecoveryAction.STOP,
            confidence=0.90,
            reason="Negative recovery test",
            expected_recovery=-500.0,  # Invalid: < 0
        )


def test_5_gemini_response_parsing():
    """5. Mock Gemini client and test response parsing and mapping."""
    agent = AIRecoveryAgent(api_key="dummy_valid_test_key")
    mock_client = MagicMock()
    mock_response = MagicMock()

    mock_json = {
        "action": "PAYMENT_LINK",
        "confidence": 0.93,
        "reason": "Customer faced bank downtime; payment link provides best asynchronous recovery.",
        "expected_recovery": 3500.0,
        "risk_level": "LOW",
        "proposed_discount_pct": 0.0,
        "requires_approval_recommendation": False,
    }
    mock_response.text = json.dumps(mock_json)
    mock_client.models.generate_content.return_value = mock_response
    agent._client = mock_client

    payment = sample_payment(amount=3500.0)
    rec = agent.recommend(payment)

    assert rec.action == RecoveryAction.PAYMENT_LINK
    assert rec.confidence == 0.93
    assert rec.agent_mode == "gemini"
    assert rec.model_name == "gemini-2.5-flash"
    assert "bank downtime" in rec.reason


def test_6_missing_gemini_api_key_triggers_fallback():
    """6. Ensure missing API key cleanly triggers heuristic fallback."""
    agent = AIRecoveryAgent(api_key="")
    payment = sample_payment()
    rec = agent.recommend(payment)

    assert rec.agent_mode == "fallback"
    assert rec.model_name is None
    assert rec.action in [a for a in RecoveryAction]


def test_7_gemini_api_failure_triggers_fallback():
    """7. Ensure runtime Gemini API error automatically falls back to heuristic engine."""
    agent = AIRecoveryAgent(api_key="dummy_key_for_error_test")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Quota limit exceeded / Network down")
    agent._client = mock_client

    payment = sample_payment()
    rec = agent.recommend(payment)

    assert rec.agent_mode == "fallback"
    assert rec.action in [a for a in RecoveryAction]


def test_8_suspected_fraud_produces_escalate_or_stop():
    """8. Suspected fraud must produce ESCALATE or STOP with critical risk."""
    payment = sample_payment(failure_reason=FailureReason.SUSPECTED_FRAUD)
    rec = heuristic_fallback_recommend(payment)

    assert rec.action in (RecoveryAction.ESCALATE, RecoveryAction.STOP)
    assert rec.risk_level == "CRITICAL"
    assert rec.expected_recovery == 0.0
    assert rec.requires_approval_recommendation is True
    assert "SUSPECTED_FRAUD" in rec.reason


def test_9_retry_exhausted_produces_escalate_or_stop():
    """9. Exhausted retries (retry_count >= 2) must produce ESCALATE or STOP."""
    # Low-value customer with exhausted retries -> STOP
    payment = sample_payment(amount=1200.0, retry_count=3, previous_payments=1, total_spend=1200.0)
    rec = heuristic_fallback_recommend(payment)

    assert rec.action == RecoveryAction.STOP
    assert rec.risk_level == "HIGH"
    assert "exhausted automated attempts" in rec.reason

    # High-value customer with exhausted retries -> ESCALATE
    payment_high = sample_payment(amount=25000.0, retry_count=2, previous_payments=5, total_spend=60000.0)
    rec_high = heuristic_fallback_recommend(payment_high)

    assert rec_high.action == RecoveryAction.ESCALATE
    assert rec_high.requires_approval_recommendation is True


def test_10_temporary_failure_with_successful_history_can_produce_retry():
    """10. Temporary network/bank failure with loyal customer and 0 retries produces RETRY."""
    payment = sample_payment(
        failure_reason=FailureReason.NETWORK_TIMEOUT,
        retry_count=0,
        previous_payments=5,
        total_spend=12000.0,
    )
    rec = heuristic_fallback_recommend(payment)

    assert rec.action == RecoveryAction.RETRY
    assert rec.confidence >= 0.90
    assert rec.risk_level == "LOW"
    assert rec.proposed_discount_pct == 0.0


def test_11_card_expired_can_produce_payment_link():
    """11. Expired or invalid cards produce PAYMENT_LINK."""
    payment = sample_payment(failure_reason=FailureReason.CARD_EXPIRED)
    rec = heuristic_fallback_recommend(payment)

    assert rec.action == RecoveryAction.PAYMENT_LINK
    assert rec.confidence >= 0.85
    assert "CARD_EXPIRED" in rec.reason


def test_12_insufficient_funds_produces_sensible_recovery():
    """12. Insufficient funds produces PAYMENT_LINK for returning users or REMINDER for new."""
    # Returning customer
    returning = sample_payment(
        failure_reason=FailureReason.INSUFFICIENT_FUNDS,
        previous_payments=3,
        total_spend=7000.0,
    )
    rec_returning = heuristic_fallback_recommend(returning)
    assert rec_returning.action == RecoveryAction.PAYMENT_LINK

    # New customer
    new_cust = sample_payment(
        failure_reason=FailureReason.INSUFFICIENT_FUNDS,
        previous_payments=0,
        total_spend=0.0,
    )
    rec_new = heuristic_fallback_recommend(new_cust)
    assert rec_new.action in (RecoveryAction.REMINDER, RecoveryAction.STOP)


def test_13_fallback_mode_is_clearly_distinguishable_from_gemini():
    """13. Fallback mode is explicitly labeled as 'fallback', never disguised as Gemini."""
    payment = sample_payment()
    rec_fallback = heuristic_fallback_recommend(payment)
    assert rec_fallback.agent_mode == "fallback"
    assert rec_fallback.model_name is None

    # Compare with a mocked Gemini recommendation
    agent = AIRecoveryAgent(api_key="test_key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "action": "PAYMENT_LINK",
        "confidence": 0.89,
        "reason": "Gemini generated reasoning",
        "expected_recovery": 2499.0,
        "risk_level": "LOW",
    })
    mock_client.models.generate_content.return_value = mock_response
    agent._client = mock_client

    rec_gemini = agent.recommend(payment)
    assert rec_gemini.agent_mode == "gemini"
    assert rec_gemini.model_name == "gemini-2.5-flash"
