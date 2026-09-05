"""Unit tests for Razorpay Client, Mock Sandbox, Policy Gating, and Idempotency."""

from unittest.mock import MagicMock
import pytest

from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    ProductCategory,
    RecoveryAction,
    PolicyResult,
    PolicyDecisionType,
    RazorpayExecutionResult,
)
from src.razorpay_client import (
    RazorpayRecoveryClient,
    sanitize_secrets,
    execute_recovery,
)


def sample_payment(
    amount: float = 2499.0,
    status: PaymentStatus = PaymentStatus.FAILED,
    failure_reason: FailureReason = FailureReason.NETWORK_TIMEOUT,
    retry_count: int = 0,
) -> PaymentRecord:
    """Helper to create a test PaymentRecord."""
    return PaymentRecord(
        payment_id="pay_rzp_test_001",
        customer_id="cust_rzp_test_001",
        customer_name="Aditi Patel",
        customer_email="aditi.patel@example.com",
        customer_phone="+919876543210",
        order_id="order_rzp_test_001",
        amount=amount,
        currency="INR",
        payment_status=status,
        failure_reason=failure_reason,
        retry_count=retry_count,
        customer_previous_payments=3,
        customer_previous_failures=1,
        customer_total_spend=7500.0,
        days_since_last_success=10,
        product_category=ProductCategory.ECOMMERCE,
    )


def sample_policy_result(
    decision: PolicyDecisionType = PolicyDecisionType.ALLOWED,
    allowed: bool = True,
    requires_approval: bool = False,
    effective_discount: float = 0.0,
    final_payable: float = 2499.0,
    reason: str = "Action allowed by policy.",
) -> PolicyResult:
    """Helper to create a test PolicyResult."""
    return PolicyResult(
        decision=decision,
        allowed=allowed,
        requires_approval=requires_approval,
        reason=reason,
        effective_discount=effective_discount,
        final_payable_amount=final_payable,
        policy_codes=["ACTION_ALLOWED" if allowed else "POLICY_BLOCKED"],
    )


# ===========================================================================
# 16 Mandatory Test Cases
# ===========================================================================

def test_1_mock_mode_can_create_payment_link():
    """1. Mock mode can create a payment link with realistic structure."""
    client = RazorpayRecoveryClient(mode="mock")
    res = client.create_payment_link(
        amount=1999.0,
        customer_name="Test Customer",
        customer_email="test@example.com",
    )
    assert res["entity"] == "payment_link"
    assert res["id"].startswith("mock_plink_")
    assert res["amount"] == 199900  # Amount in paise
    assert "https://rzp.io/i/mock_" in res["short_url"]
    assert res["simulated"] is True


def test_2_mock_mode_can_create_order():
    """2. Mock mode can create an order for retry workflow."""
    client = RazorpayRecoveryClient(mode="mock")
    res = client.create_order(amount=2500.0)
    assert res["entity"] == "order"
    assert res["id"].startswith("mock_order_")
    assert res["amount"] == 250000
    assert res["status"] == "created"
    assert res["simulated"] is True


def test_3_mock_mode_returns_deterministic_ids():
    """3. Mock mode returns deterministic IDs based on inputs."""
    client = RazorpayRecoveryClient(mode="mock")
    res1 = client.create_payment_link(amount=1500.0, customer_name="A", customer_email="a@a.com", reference_id="fixed_ref_1")
    res2 = client.create_payment_link(amount=1500.0, customer_name="A", customer_email="a@a.com", reference_id="fixed_ref_1")
    assert res1["id"] == res2["id"]
    assert res1["short_url"] == res2["short_url"]


def test_4_missing_credentials_triggers_mock_behavior():
    """4. Missing or placeholder credentials cleanly trigger mock mode."""
    client = RazorpayRecoveryClient(key_id="", key_secret="", mode="auto")
    assert client.is_mock is True

    client2 = RazorpayRecoveryClient(key_id="rzp_test_your_key_here", key_secret="your_secret", mode="auto")
    assert client2.is_mock is True


def test_5_retry_maps_to_order_creation():
    """5. RETRY action maps directly to order creation."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(amount=3000.0)
    policy = sample_policy_result(final_payable=3000.0)

    res = client.execute_recovery(payment, RecoveryAction.RETRY, policy)
    assert res.success is True
    assert res.action == RecoveryAction.RETRY
    assert res.status == "ORDER_CREATED"
    assert res.razorpay_id.startswith("mock_order_")
    assert res.amount == 3000.0


def test_6_payment_link_maps_to_payment_link_creation():
    """6. PAYMENT_LINK action maps to payment link creation."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(amount=4500.0)
    policy = sample_policy_result(final_payable=4500.0)

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res.success is True
    assert res.action == RecoveryAction.PAYMENT_LINK
    assert res.status == "PAYMENT_LINK_CREATED"
    assert res.razorpay_id.startswith("mock_plink_")
    assert res.short_url is not None


def test_7_reminder_performs_no_financial_api_call():
    """7. REMINDER performs no financial API call."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment()
    policy = sample_policy_result()

    res = client.execute_recovery(payment, RecoveryAction.REMINDER, policy)
    assert res.success is True
    assert res.action == RecoveryAction.REMINDER
    assert res.status == "REMINDER_SIMULATED"
    assert res.razorpay_id is None
    assert "No payment API called" in res.message


def test_8_escalate_performs_no_financial_api_call():
    """8. ESCALATE performs no financial API call."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(amount=45000.0)
    policy = sample_policy_result(final_payable=45000.0)

    res = client.execute_recovery(payment, RecoveryAction.ESCALATE, policy)
    assert res.success is True
    assert res.action == RecoveryAction.ESCALATE
    assert res.status == "ESCALATED"
    assert res.razorpay_id is None
    assert "No payment API called" in res.message


def test_9_stop_performs_no_financial_api_call():
    """9. STOP performs no financial API call."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment()
    policy = sample_policy_result()

    res = client.execute_recovery(payment, RecoveryAction.STOP, policy)
    assert res.success is True
    assert res.action == RecoveryAction.STOP
    assert res.status == "STOPPED"
    assert res.razorpay_id is None
    assert "No action taken" in res.message


def test_10_blocked_policy_prevents_execution():
    """10. BLOCKED policy strictly prevents execution."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment()
    policy = PolicyResult(
        decision=PolicyDecisionType.BLOCKED,
        allowed=False,
        requires_approval=False,
        reason="Blocked by fraud protection rule.",
        policy_codes=["FRAUD_PROTECTION"],
        final_payable_amount=payment.amount,
    )

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res.success is False
    assert res.status == "BLOCKED"
    assert res.error_code == "POLICY_BLOCKED"
    assert "Execution blocked by Policy Engine" in res.message


def test_11_approval_required_policy_prevents_execution_when_unapproved():
    """11. Approval-required policy refuses execution when approved_by_merchant is False."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(amount=35000.0)
    policy = PolicyResult(
        decision=PolicyDecisionType.REQUIRES_APPROVAL,
        allowed=False,
        requires_approval=True,
        reason="Transaction exceeds ₹15,000 threshold.",
        policy_codes=["HIGH_VALUE_APPROVAL", "APPROVAL_REQUIRED"],
        final_payable_amount=35000.0,
    )

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy, approved_by_merchant=False)
    assert res.success is False
    assert res.status == "APPROVAL_REQUIRED"
    assert res.error_code == "APPROVAL_REQUIRED"
    assert "merchant approval required" in res.message


def test_12_approval_required_policy_executes_when_approved():
    """12. Approval-required policy executes when approved_by_merchant is True."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(amount=35000.0)
    policy = PolicyResult(
        decision=PolicyDecisionType.REQUIRES_APPROVAL,
        allowed=False,
        requires_approval=True,
        reason="Transaction exceeds ₹15,000 threshold.",
        policy_codes=["HIGH_VALUE_APPROVAL", "APPROVAL_REQUIRED"],
        final_payable_amount=35000.0,
    )

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy, approved_by_merchant=True)
    assert res.success is True
    assert res.status == "PAYMENT_LINK_CREATED"
    assert res.razorpay_id.startswith("mock_plink_")


def test_13_duplicate_recovery_is_prevented():
    """13. Duplicate recovery action on same payment is rejected."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(amount=1500.0)
    policy = sample_policy_result(final_payable=1500.0)

    # First execution succeeds
    res1 = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res1.success is True

    # Immediate second execution of same action is blocked
    res2 = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res2.success is False
    assert res2.status == "DUPLICATE_PREVENTED"
    assert res2.error_code == "DUPLICATE_ACTION"
    assert "already been executed" in res2.message


def test_14_already_recovered_payment_cannot_execute():
    """14. Already recovered payment cannot be re-executed."""
    client = RazorpayRecoveryClient(mode="mock")
    payment = sample_payment(status=PaymentStatus.RECOVERED)
    policy = sample_policy_result()

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res.success is False
    assert res.status == "INVALID_STATE"
    assert res.error_code == "ALREADY_RECOVERED"


def test_15_api_failure_converted_to_structured_result():
    """15. API exceptions are converted into clean structured failure results."""
    client = RazorpayRecoveryClient(mode="test", key_id="rzp_test_validmockkey123", key_secret="valid_mock_secret")
    client.is_mock = False
    mock_rzp = MagicMock()
    mock_rzp.payment_link.create.side_effect = RuntimeError("500 Internal Gateway Outage")
    client._client = mock_rzp

    payment = sample_payment()
    policy = sample_policy_result()

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res.success is False
    assert res.status == "FAILED"
    assert res.error_code == "API_ERROR"
    assert "500 Internal Gateway Outage" in res.message


def test_16_secrets_never_appear_in_returned_error_text():
    """16. Secret keys and IDs never leak in error messages."""
    raw_error = "Authentication failed for user rzp_test_abcdef12345678 with secret topsecretpassword999"
    sanitized = sanitize_secrets(raw_error, key_id="rzp_test_abcdef12345678", key_secret="topsecretpassword999")

    assert "topsecretpassword999" not in sanitized
    assert "[REDACTED" in sanitized
    assert "rzp_test_abcdef12345678" not in sanitized


def test_17_razorpay_429_rate_limit_handled_gracefully():
    """17. Razorpay 429 Too Many Requests error produces RATE_LIMITED status and does not mark action as executed."""
    client = RazorpayRecoveryClient(mode="test", key_id="rzp_test_validmockkey123", key_secret="valid_mock_secret")
    client.is_mock = False
    mock_rzp = MagicMock()
    # Simulate 429 error
    mock_rzp.payment_link.create.side_effect = RuntimeError("429 Client Error: Too Many Requests for url")
    client._client = mock_rzp

    payment = sample_payment()
    policy = sample_policy_result()

    res = client.execute_recovery(payment, RecoveryAction.PAYMENT_LINK, policy)
    assert res.success is False
    assert res.status == "RATE_LIMITED"
    assert res.error_code == "RATE_LIMITED"
    assert "429" in res.message or "Rate Limit" in res.message
    # Confirm action was NOT added to executed actions
    assert (payment.payment_id, RecoveryAction.PAYMENT_LINK.value) not in client._executed_actions

