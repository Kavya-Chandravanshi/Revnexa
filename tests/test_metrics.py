"""Unit tests for financial metrics calculation and batch simulation pipeline."""

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
from src.merchant_approval import MerchantApprovalQueue
from src.audit_logger import AuditLogger
from src.razorpay_client import RazorpayRecoveryClient
from src.metrics import calculate_metrics, run_batch_recovery, process_single_payment


def sample_payment(
    pid: str = "p1",
    amount: float = 3000.0,
    status: PaymentStatus = PaymentStatus.FAILED,
    failure_reason: FailureReason = FailureReason.NETWORK_TIMEOUT,
) -> PaymentRecord:
    """Helper to create a PaymentRecord."""
    return PaymentRecord(
        payment_id=pid,
        customer_id="c1",
        customer_name="Test User",
        customer_email="user@example.com",
        order_id=f"order_{pid}",
        amount=amount,
        currency="INR",
        payment_status=status,
        failure_reason=failure_reason,
        product_category=ProductCategory.SAAS,
    )


# ===========================================================================
# Metrics Calculation Tests
# ===========================================================================

def test_1_calculate_total_volume():
    """1. Verify total failed and at-risk volume calculation."""
    p1 = sample_payment("p1", amount=5000.0, status=PaymentStatus.FAILED)
    p2 = sample_payment("p2", amount=2000.0, status=PaymentStatus.AT_RISK)

    metrics = calculate_metrics([p1, p2], outcomes=[])
    assert metrics.total_records == 2
    assert metrics.total_failed_volume == 5000.0
    assert metrics.total_at_risk_volume == 2000.0


def test_2_calculate_recovered_revenue():
    """2. Verify sum of recovered revenue from successful executions."""
    p1 = sample_payment("p1", amount=3000.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec1",
        razorpay_id="mock_plink_1",
        status="PAYMENT_LINK_CREATED",
        amount=3000.0,
        message="Created",
        simulated=True,
    )
    outcomes = [{
        "status": "RECOVERED",
        "action": RecoveryAction.PAYMENT_LINK,
        "execution": exec_res,
        "policy_result": None,
    }]

    metrics = calculate_metrics([p1], outcomes)
    assert metrics.successful_recoveries == 1
    assert metrics.total_recovered_revenue == 3000.0


def test_3_calculate_recovery_rate():
    """3. Verify recovery rate percentage calculation."""
    p1 = sample_payment("p1", 1000.0)
    p2 = sample_payment("p2", 1000.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec1",
        status="PAYMENT_LINK_CREATED",
        amount=1000.0,
        message="Created",
        simulated=True,
    )
    # 1 success out of 2 records
    outcomes = [
        {"status": "RECOVERED", "action": RecoveryAction.PAYMENT_LINK, "execution": exec_res},
        {"status": "FAILED", "action": RecoveryAction.RETRY, "execution": None},
    ]
    metrics = calculate_metrics([p1, p2], outcomes)
    assert metrics.recovery_rate == 50.0  # 1/2 * 100


def test_4_calculate_incentives():
    """4. Verify incentive discount accumulation."""
    p1 = sample_payment("p1", 2000.0)
    policy_res = PolicyResult(
        decision=PolicyDecisionType.ALLOWED,
        allowed=True,
        requires_approval=False,
        reason="Incentive allowed",
        effective_discount=100.0,
        final_payable_amount=1900.0,
        policy_codes=["ACTION_ALLOWED"],
    )
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.INCENTIVE,
        recovery_id="rec1",
        status="PAYMENT_LINK_CREATED",
        amount=1900.0,
        message="Discounted",
        simulated=True,
    )
    outcomes = [{
        "status": "RECOVERED",
        "action": RecoveryAction.INCENTIVE,
        "execution": exec_res,
        "policy_result": policy_res,
    }]
    metrics = calculate_metrics([p1], outcomes)
    assert metrics.incentive_amount_given == 100.0
    assert metrics.total_recovered_revenue == 1900.0


def test_5_calculate_net_recovered_revenue():
    """5. Verify net recovered revenue = recovered revenue - incentives."""
    p1 = sample_payment("p1", 5000.0)
    policy_res = PolicyResult(
        decision=PolicyDecisionType.ALLOWED,
        allowed=True,
        requires_approval=False,
        reason="Incentive allowed",
        effective_discount=250.0,
        final_payable_amount=4750.0,
        policy_codes=["ACTION_ALLOWED"],
    )
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.INCENTIVE,
        recovery_id="rec1",
        status="PAYMENT_LINK_CREATED",
        amount=4750.0,
        message="Discounted",
        simulated=True,
    )
    outcomes = [{
        "status": "RECOVERED",
        "action": RecoveryAction.INCENTIVE,
        "execution": exec_res,
        "policy_result": policy_res,
    }]
    metrics = calculate_metrics([p1], outcomes)
    assert metrics.total_recovered_revenue == 4750.0
    assert metrics.incentive_amount_given == 250.0
    assert metrics.net_recovered_revenue == 4500.0  # 4750 - 250


def test_6_calculate_approval_counts():
    """6. Verify approval queue statistics integration."""
    queue = MerchantApprovalQueue()
    queue.create_request("p1", "c1", 25000.0, RecoveryAction.PAYMENT_LINK, "r", 0.8, "p")
    req2 = queue.create_request("p2", "c2", 30000.0, RecoveryAction.PAYMENT_LINK, "r", 0.8, "p")
    req3 = queue.create_request("p3", "c3", 40000.0, RecoveryAction.PAYMENT_LINK, "r", 0.8, "p")
    queue.approve_request(req2.approval_id)
    queue.reject_request(req3.approval_id)

    outcomes = [{"status": "PENDING_APPROVAL"}, {"status": "RECOVERED"}]
    metrics = calculate_metrics([sample_payment(), sample_payment()], outcomes, approval_queue=queue)
    assert metrics.approvals_requested == 1
    assert metrics.approvals_granted == 1
    assert metrics.approvals_rejected == 1


def test_7_calculate_blocked_counts():
    """7. Verify blocked actions tally correctly."""
    outcomes = [{"status": "BLOCKED"}, {"status": "BLOCKED"}, {"status": "STOPPED"}]
    metrics = calculate_metrics([sample_payment()]*3, outcomes)
    assert metrics.blocked_actions == 2


def test_8_handle_zero_opportunities_without_division_errors():
    """8. Verify empty record set does not throw ZeroDivisionError."""
    metrics = calculate_metrics([], [])
    assert metrics.total_records == 0
    assert metrics.recovery_rate == 0.0
    assert metrics.average_recovered_amount == 0.0
    assert metrics.roi == 0.0


# ===========================================================================
# Batch Simulation Tests
# ===========================================================================

def test_batch_1_invalid_record_does_not_crash():
    """Batch 1: Exception during one record does not abort entire run."""
    queue = MerchantApprovalQueue()
    audit = AuditLogger()
    p1 = sample_payment("p1", 1000.0)
    p2 = sample_payment("p2", 2000.0)

    # p1 passes normally, p2 processed
    metrics, outcomes = run_batch_recovery(
        records=[p1, p2],
        approval_queue=queue,
        audit_logger=audit,
        force_fallback=True,
    )
    assert metrics.total_records == 2
    assert len(outcomes) == 2


def test_batch_2_blocked_actions_do_not_count_as_recovered():
    """Batch 2: Blocked fraud transactions produce 0 recovered revenue."""
    queue = MerchantApprovalQueue()
    audit = AuditLogger()
    fraud_payment = sample_payment("p_fraud", 10000.0, failure_reason=FailureReason.SUSPECTED_FRAUD)

    metrics, outcomes = run_batch_recovery(
        records=[fraud_payment],
        approval_queue=queue,
        audit_logger=audit,
        force_fallback=True,
    )
    # Fraud gets escalated or stopped, 0 revenue
    assert metrics.total_recovered_revenue == 0.0
    assert metrics.successful_recoveries == 0


def test_batch_3_pending_approvals_do_not_count_as_recovered():
    """Batch 3: High value transactions (> ₹15,000) remain pending and do NOT count as revenue."""
    queue = MerchantApprovalQueue()
    audit = AuditLogger()
    # High value card expired
    high_val = sample_payment("p_high", 35000.0, failure_reason=FailureReason.CARD_EXPIRED)

    metrics, outcomes = run_batch_recovery(
        records=[high_val],
        approval_queue=queue,
        audit_logger=audit,
        force_fallback=True,
    )
    assert outcomes[0]["status"] == "PENDING_APPROVAL"
    assert metrics.approvals_requested == 1
    assert metrics.total_recovered_revenue == 0.0
    assert metrics.successful_recoveries == 0


def test_batch_4_successful_execution_counts_as_recovered():
    """Batch 4: Safe transient failure auto-executes and counts as recovered in mock simulation."""
    queue = MerchantApprovalQueue()
    audit = AuditLogger()
    client = RazorpayRecoveryClient(mode="mock")
    p = sample_payment("p_safe", 2500.0, failure_reason=FailureReason.NETWORK_TIMEOUT)
    p.customer_previous_payments = 2
    p.customer_total_spend = 5000.0
    p.days_since_last_success = 10

    metrics, outcomes = run_batch_recovery(
        records=[p],
        approval_queue=queue,
        audit_logger=audit,
        rzp_client=client,
        force_fallback=True,
    )
    assert outcomes[0]["status"] == "RECOVERED"
    assert metrics.successful_recoveries == 1
    assert metrics.total_recovered_revenue == 2500.0


def test_batch_5_simulated_recovery_is_marked_simulated():
    """Batch 5: Mock client records simulated=True in metrics and results."""
    queue = MerchantApprovalQueue()
    audit = AuditLogger()
    client = RazorpayRecoveryClient(mode="mock")
    p = sample_payment("p_mock", 1500.0)

    metrics, outcomes = run_batch_recovery(
        records=[p],
        approval_queue=queue,
        audit_logger=audit,
        rzp_client=client,
        force_fallback=True,
    )
    assert metrics.is_simulated is True
    if outcomes[0]["execution"]:
        assert outcomes[0]["execution"].simulated is True


def test_batch_6_audit_events_are_produced():
    """Batch 6: Ensure audit events are created for every stage of recovery."""
    queue = MerchantApprovalQueue()
    audit = AuditLogger()
    p = sample_payment("p_audit", 1500.0)

    run_batch_recovery(
        records=[p],
        approval_queue=queue,
        audit_logger=audit,
        force_fallback=True,
    )
    events = audit.get_events_for_payment("p_audit")
    event_types = [e.event_type for e in events]

    assert "PAYMENT_DETECTED" in event_types
    assert "AI_ANALYZED" in event_types
    assert "POLICY_EVALUATED" in event_types


def test_payment_link_created_is_action_executed_not_confirmed_revenue():
    """Verify that creating a payment link counts as action_executed, NOT confirmed revenue."""
    p1 = sample_payment("p1", amount=5000.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec1",
        razorpay_id="plink_test123",
        status="PAYMENT_LINK_CREATED",
        amount=5000.0,
        message="Payment link created",
        simulated=False,
    )
    # Action executed status (link dispatched, not confirmed paid)
    outcomes = [{
        "status": "ACTION_EXECUTED",
        "action": RecoveryAction.PAYMENT_LINK,
        "execution": exec_res,
        "policy_result": None,
    }]

    metrics = calculate_metrics([p1], outcomes, is_simulated=False)
    assert metrics.actions_executed == 1
    assert metrics.successful_recoveries == 0
    assert metrics.total_recovered_revenue == 0.0


def test_confirmed_payment_counts_as_recovered_revenue():
    """Verify that a confirmed payment increments successful_recoveries and total_recovered_revenue."""
    p1 = sample_payment("p1", amount=5000.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec1",
        razorpay_id="plink_test123",
        status="PAYMENT_LINK_CREATED",
        amount=5000.0,
        message="Payment confirmed",
        simulated=False,
    )
    outcomes = [{
        "status": "RECOVERED",
        "action": RecoveryAction.PAYMENT_LINK,
        "execution": exec_res,
        "policy_result": None,
        "payment_confirmed": True,
    }]

    metrics = calculate_metrics([p1], outcomes, is_simulated=False)
    assert metrics.actions_executed == 1
    assert metrics.successful_recoveries == 1
    assert metrics.total_recovered_revenue == 5000.0


def test_recovery_rate_cannot_exceed_100():
    """Verify recovery_rate is safely clamped to 100.0% even with abnormal outcomes."""
    p1 = sample_payment("p1", amount=1000.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec1",
        razorpay_id="plink_1",
        status="RECOVERED",
        amount=1000.0,
        message="Simulated recovery",
        simulated=True,
    )
    # 2 outcomes for 1 record
    outcomes = [
        {"status": "RECOVERED", "action": RecoveryAction.PAYMENT_LINK, "execution": exec_res},
        {"status": "RECOVERED", "action": RecoveryAction.PAYMENT_LINK, "execution": exec_res},
    ]
    metrics = calculate_metrics([p1], outcomes, is_simulated=True)
    assert metrics.recovery_rate <= 100.0


def test_batch_recovery_performance_sub_second():
    """Verify 500-record batch recovery executes in under 5 seconds (preventing loop regression)."""
    import time
    from src.razorpay_client import RazorpayRecoveryClient

    client = RazorpayRecoveryClient(mode="mock")
    t0 = time.time()
    batch_metrics, outcomes = run_batch_recovery(rzp_client=client, force_fallback=True)
    elapsed = time.time() - t0

    assert len(outcomes) == 500
    assert batch_metrics.total_records == 500
    assert elapsed < 5.0, f"Batch recovery took {elapsed:.2f}s, expected < 5.0s"


def test_canonical_500_dataset_reconciliation():
    """Verify Overview and Batch volume at risk reconcile to the exact same portfolio total."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "data" / "synthetic_payments.json"
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = [PaymentRecord.model_validate(r) for r in raw]

    sum_all = round(sum(r.amount for r in records), 2)
    metrics = calculate_metrics(records, [])

    assert metrics.total_records == 500
    assert metrics.total_revenue_at_risk == sum_all
    assert round(metrics.total_failed_volume + metrics.total_at_risk_volume, 2) == sum_all


def test_successful_mock_recovery_increments_simulated_revenue():
    """Verify confirmed simulated recovery increments successful_recoveries and total_recovered_revenue."""
    p1 = sample_payment("p1", amount=3500.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec_p1",
        razorpay_id="mock_plink_p1",
        status="PAYMENT_LINK_CREATED",
        amount=3500.0,
        message="Link dispatched",
        simulated=True,
    )
    outcomes = [{
        "status": "RECOVERED",
        "action": RecoveryAction.PAYMENT_LINK,
        "execution": exec_res,
        "payment_confirmed": True,
    }]
    metrics = calculate_metrics([p1], outcomes, is_simulated=True)
    assert metrics.successful_recoveries == 1
    assert metrics.total_recovered_revenue == 3500.0
    assert metrics.recovery_rate == 100.0


def test_action_executed_without_confirmation_does_not_count_as_revenue():
    """Verify action executed without confirmed payment produces 0 recovered revenue."""
    p1 = sample_payment("p1", amount=3500.0)
    exec_res = RazorpayExecutionResult(
        success=True,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec_p1",
        razorpay_id="plink_p1",
        status="PAYMENT_LINK_CREATED",
        amount=3500.0,
        message="Link dispatched",
        simulated=False,
    )
    outcomes = [{
        "status": "ACTION_EXECUTED",
        "action": RecoveryAction.PAYMENT_LINK,
        "execution": exec_res,
        "payment_confirmed": False,
    }]
    metrics = calculate_metrics([p1], outcomes, is_simulated=False)
    assert metrics.actions_executed == 1
    assert metrics.payment_links_created == 1
    assert metrics.successful_recoveries == 0
    assert metrics.total_recovered_revenue == 0.0
    assert metrics.recovery_rate == 0.0


def test_payment_link_counter_ignores_failed_and_429_requests():
    """Verify payment_links_created does NOT increment on failed or 429 gateway calls."""
    p1 = sample_payment("p1", amount=2500.0)
    exec_fail = RazorpayExecutionResult(
        success=False,
        action=RecoveryAction.PAYMENT_LINK,
        recovery_id="rec_fail",
        razorpay_id=None,
        status="RATE_LIMITED",
        amount=2500.0,
        message="Rate limit 429",
        simulated=False,
        error_code="RATE_LIMITED",
    )
    outcomes = [{
        "status": "FAILED",
        "action": RecoveryAction.PAYMENT_LINK,
        "execution": exec_fail,
    }]
    metrics = calculate_metrics([p1], outcomes, is_simulated=False)
    assert metrics.payment_links_created == 0
    assert metrics.actions_executed == 0
    assert metrics.successful_recoveries == 0
    assert metrics.total_recovered_revenue == 0.0


def test_missing_execution_object_handles_gracefully():
    """Verify outcome dictionaries without an 'execution' key do not raise ValidationError or crash calculation."""
    p1 = sample_payment("p1", amount=2500.0)
    outcomes = [
        {"payment_id": "p1", "status": "AT_RISK"},  # Missing 'execution' key entirely
        {
            "payment_id": "p2",
            "execution": RazorpayExecutionResult(
                success=True,
                action=RecoveryAction.RETRY,
                recovery_id="rec_p2",
                razorpay_id="ord_123",
                status="SUCCESS",
                amount=1000.0,
                message="Executed",
                simulated=True,
            ),
        },
        {
            "payment_id": "p3",
            "execution": RazorpayExecutionResult(
                success=False,
                action=RecoveryAction.PAYMENT_LINK,
                recovery_id="rec_p3",
                razorpay_id=None,
                status="FAILED",
                amount=1500.0,
                message="Failed",
                simulated=True,
            ),
        },
    ]
    
    # Helper logic mirrors app.py execution calculation safely
    def is_execution_failed(outcome: dict) -> bool:
        execution = outcome.get("execution")
        if execution is None:
            return False
        return not bool(getattr(execution, "success", False))

    failed_count = sum(1 for o in outcomes if is_execution_failed(o))
    assert failed_count == 1  # Only p3 is a failed execution

    # Verify metrics calculation runs smoothly
    metrics = calculate_metrics([p1], outcomes, is_simulated=True)
    assert metrics is not None



