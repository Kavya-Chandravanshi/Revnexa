"""Unit tests for the Deterministic Policy Engine and Safety Guardrails.

Verifies that all 10 core safety rules, bounds, and approvals are strictly enforced
without relying on any LLM or external network services.
"""

import pytest
from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    ProductCategory,
    RecoveryAction,
    PolicyDecisionType,
)
from src.policy_engine import (
    evaluate_policy,
    PolicyEngine,
    CODE_ACTION_ALLOWED,
    CODE_APPROVAL_REQUIRED,
    CODE_HIGH_VALUE_APPROVAL,
    CODE_RETRY_LIMIT_EXCEEDED,
    CODE_FRAUD_PROTECTION,
    CODE_INCENTIVE_CAP_EXCEEDED,
    CODE_DUPLICATE_ACTION,
    CODE_INVALID_PAYMENT_STATE,
    CODE_INVALID_ACTION,
)


def create_sample_payment(
    amount: float = 2500.0,
    status: PaymentStatus = PaymentStatus.FAILED,
    failure_reason: FailureReason = FailureReason.BANK_SERVER_DOWN,
    retry_count: int = 0,
    customer_previous_payments: int = 3,
    customer_total_spend: float = 7500.0,
    days_since_last_success: int = 12,
) -> PaymentRecord:
    """Helper factory for generating valid PaymentRecord instances for testing."""
    return PaymentRecord(
        payment_id="pay_test_fixture",
        customer_id="cust_test_fixture",
        customer_name="Test Customer",
        customer_email="customer@example.com",
        order_id="order_test_fixture",
        amount=amount,
        currency="INR",
        payment_status=status,
        failure_reason=failure_reason,
        retry_count=retry_count,
        customer_previous_payments=customer_previous_payments,
        customer_previous_failures=0,
        customer_total_spend=customer_total_spend,
        days_since_last_success=days_since_last_success,
        product_category=ProductCategory.ECOMMERCE,
    )


# ===========================================================================
# 16 Mandatory Safety Test Cases
# ===========================================================================

def test_1_normal_retry_allowed():
    """1. Normal retry allowed when retries < limit and amount <= threshold."""
    payment = create_sample_payment(amount=2500.0, retry_count=0)
    result = evaluate_policy(payment, RecoveryAction.RETRY)

    assert result.decision == PolicyDecisionType.ALLOWED
    assert result.allowed is True
    assert result.requires_approval is False
    assert CODE_ACTION_ALLOWED in result.policy_codes
    assert result.effective_discount == 0.0
    assert result.final_payable_amount == 2500.0


def test_2_retry_at_limit_blocked():
    """2. Retry at limit (retry_count == 2) blocked."""
    payment = create_sample_payment(amount=2500.0, retry_count=2)
    result = evaluate_policy(payment, RecoveryAction.RETRY)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_RETRY_LIMIT_EXCEEDED in result.policy_codes
    assert "retry limit reached" in result.reason.lower()


def test_3_retry_above_limit_blocked():
    """3. Retry above limit (retry_count == 3) blocked."""
    payment = create_sample_payment(amount=2500.0, retry_count=3)
    result = evaluate_policy(payment, RecoveryAction.RETRY)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_RETRY_LIMIT_EXCEEDED in result.policy_codes


def test_4_suspected_fraud_retry_blocked():
    """4. Suspected fraud + retry blocked."""
    payment = create_sample_payment(
        amount=1500.0,
        failure_reason=FailureReason.SUSPECTED_FRAUD,
        retry_count=0,
    )
    result = evaluate_policy(payment, RecoveryAction.RETRY)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_FRAUD_PROTECTION in result.policy_codes
    assert "suspected_fraud" in result.reason.lower()


def test_5_suspected_fraud_incentive_blocked():
    """5. Suspected fraud + incentive blocked."""
    payment = create_sample_payment(
        amount=1500.0,
        failure_reason=FailureReason.SUSPECTED_FRAUD,
        retry_count=0,
    )
    result = evaluate_policy(payment, RecoveryAction.INCENTIVE, proposed_discount_pct=5.0)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_FRAUD_PROTECTION in result.policy_codes


def test_6_normal_payment_link_allowed():
    """6. Normal payment link allowed for low-value failure."""
    payment = create_sample_payment(amount=4500.0, failure_reason=FailureReason.AUTHENTICATION_FAILED)
    result = evaluate_policy(payment, RecoveryAction.PAYMENT_LINK)

    assert result.decision == PolicyDecisionType.ALLOWED
    assert result.allowed is True
    assert result.requires_approval is False
    assert CODE_ACTION_ALLOWED in result.policy_codes
    assert result.final_payable_amount == 4500.0


def test_7_high_value_payment_link_requires_approval():
    """7. High-value payment link (> ₹15,000) requires merchant approval."""
    payment = create_sample_payment(amount=35000.0)
    result = evaluate_policy(payment, RecoveryAction.PAYMENT_LINK)

    assert result.decision == PolicyDecisionType.REQUIRES_APPROVAL
    assert result.allowed is False  # Cannot auto-execute without merchant approval
    assert result.requires_approval is True
    assert CODE_HIGH_VALUE_APPROVAL in result.policy_codes
    assert CODE_APPROVAL_REQUIRED in result.policy_codes
    assert "exceeds autonomous recovery limit" in result.reason


def test_8_high_value_retry_requires_approval():
    """8. High-value retry (> ₹15,000) requires merchant approval."""
    payment = create_sample_payment(amount=22000.0, retry_count=0)
    result = evaluate_policy(payment, RecoveryAction.RETRY)

    assert result.decision == PolicyDecisionType.REQUIRES_APPROVAL
    assert result.allowed is False
    assert result.requires_approval is True
    assert CODE_HIGH_VALUE_APPROVAL in result.policy_codes
    assert CODE_APPROVAL_REQUIRED in result.policy_codes


def test_9_incentive_within_cap_allowed_for_low_value():
    """9. Incentive within cap (e.g. 5% on ₹3,000 = ₹150) allowed for low-value payment."""
    payment = create_sample_payment(amount=3000.0)
    result = evaluate_policy(payment, RecoveryAction.INCENTIVE, proposed_discount_pct=5.0)

    assert result.decision == PolicyDecisionType.ALLOWED
    assert result.allowed is True
    assert result.requires_approval is False
    assert result.effective_discount == 150.0
    assert result.final_payable_amount == 2850.0
    assert result.max_allowed_discount == 300.0  # 10% of 3000 is 300, which is < 500
    assert CODE_ACTION_ALLOWED in result.policy_codes


def test_10_incentive_over_500_blocked():
    """10. Incentive over ₹500 blocked (e.g. 8% on ₹10,000 = ₹800 > ₹500 cap)."""
    payment = create_sample_payment(amount=10000.0)
    result = evaluate_policy(payment, RecoveryAction.INCENTIVE, proposed_discount_pct=8.0)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_INCENTIVE_CAP_EXCEEDED in result.policy_codes
    assert "exceeds the maximum allowed absolute cap" in result.reason


def test_11_incentive_above_10_percent_blocked():
    """11. Incentive above 10% blocked (e.g. 15% on ₹2,000 = ₹300, under ₹500 but 15% > 10%)."""
    payment = create_sample_payment(amount=2000.0)
    result = evaluate_policy(payment, RecoveryAction.INCENTIVE, proposed_discount_pct=15.0)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_INCENTIVE_CAP_EXCEEDED in result.policy_codes
    assert "exceeds maximum allowed limit of 10.0%" in result.reason


def test_12_duplicate_recovery_action_blocked():
    """12. Duplicate recovery action blocked if already executed recently."""
    payment = create_sample_payment(amount=2500.0)
    recent = ["PAYMENT_LINK", "REMINDER"]
    result = evaluate_policy(payment, RecoveryAction.PAYMENT_LINK, recent_actions=recent)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_DUPLICATE_ACTION in result.policy_codes
    assert "already been executed recently" in result.reason


def test_13_already_recovered_payment_blocked():
    """13. Already recovered payment blocked from further recovery actions."""
    payment = create_sample_payment(amount=2500.0, status=PaymentStatus.RECOVERED)
    result = evaluate_policy(payment, RecoveryAction.PAYMENT_LINK)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_INVALID_PAYMENT_STATE in result.policy_codes
    assert "Cannot recover transaction with status 'RECOVERED'" in result.reason


def test_14_stop_allowed():
    """14. STOP is always allowed as a safe no-action outcome."""
    # Even on a high-value, fraud, exhausted-retry transaction, STOP is permitted
    payment = create_sample_payment(
        amount=50000.0,
        failure_reason=FailureReason.SUSPECTED_FRAUD,
        retry_count=4,
    )
    result = evaluate_policy(payment, RecoveryAction.STOP)

    assert result.decision == PolicyDecisionType.ALLOWED
    assert result.allowed is True
    assert result.requires_approval is False
    assert CODE_ACTION_ALLOWED in result.policy_codes


def test_15_escalate_allowed():
    """15. ESCALATE is always allowed on valid payment records."""
    # Even for high-value or fraud cases, escalating to a human is always allowed
    payment = create_sample_payment(
        amount=45000.0,
        failure_reason=FailureReason.PAYMENT_LIMIT_EXCEEDED,
        retry_count=2,
    )
    result = evaluate_policy(payment, RecoveryAction.ESCALATE)

    assert result.decision == PolicyDecisionType.ALLOWED
    assert result.allowed is True
    assert result.requires_approval is False
    assert CODE_ACTION_ALLOWED in result.policy_codes


def test_16_invalid_action_rejected():
    """16. Actions outside the allowed 6 enums are strictly rejected."""
    payment = create_sample_payment(amount=2500.0)
    result = evaluate_policy(payment, "UNAUTHORIZED_DIRECT_DEBIT")

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert result.requires_approval is False
    assert CODE_INVALID_ACTION in result.policy_codes
    assert "is invalid" in result.reason


def test_high_value_incentive_requires_approval():
    """Bonus: Valid incentive on high-value payment requires merchant approval."""
    payment = create_sample_payment(amount=20000.0)
    # 2% on 20,000 is ₹400 (under ₹500 and under 10%), but payment > ₹15,000
    result = evaluate_policy(payment, RecoveryAction.INCENTIVE, proposed_discount_pct=2.0)

    assert result.decision == PolicyDecisionType.REQUIRES_APPROVAL
    assert result.requires_approval is True
    assert result.effective_discount == 400.0
    assert result.final_payable_amount == 19600.0
    assert CODE_HIGH_VALUE_APPROVAL in result.policy_codes


def test_non_positive_amount_blocked():
    """17. Non-positive transaction amounts are blocked by policy engine."""
    payment = create_sample_payment(amount=100.0)
    payment.amount = 0.0  # bypass pydantic gt=0 to test policy engine fail-closed
    result = evaluate_policy(payment, RecoveryAction.PAYMENT_LINK)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert CODE_INVALID_PAYMENT_STATE in result.policy_codes
    assert "positive" in result.reason.lower()


def test_negative_discount_blocked():
    """18. Negative discount percentage or amount is strictly blocked."""
    payment = create_sample_payment(amount=2000.0)
    result = evaluate_policy(payment, RecoveryAction.INCENTIVE, proposed_discount_pct=-5.0)

    assert result.decision == PolicyDecisionType.BLOCKED
    assert result.allowed is False
    assert CODE_INCENTIVE_CAP_EXCEEDED in result.policy_codes
    assert "negative" in result.reason.lower()

