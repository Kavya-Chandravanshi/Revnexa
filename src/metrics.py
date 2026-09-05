"""Batch Revenue Recovery Engine & Financial Metrics for Revnexa.

Orchestrates end-to-end pipeline execution (AI -> Policy -> Approval Gate -> Razorpay Execution -> Audit Logger)
and computes mathematically sound, deterministic revenue recovery financial analytics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models import (
    PaymentRecord,
    PaymentStatus,
    RecoveryAction,
    PolicyDecisionType,
    PolicyResult,
    ApprovalStatus,
    BatchMetrics,
    AIRecommendation,
    RazorpayExecutionResult,
)
from src.ai_agent import AIRecoveryAgent, analyze_payment, _default_agent
from src.policy_engine import evaluate_policy
from src.merchant_approval import (
    MerchantApprovalQueue,
    create_approval_request,
    list_pending_approvals,
    _default_queue,
)
from src.razorpay_client import (
    RazorpayRecoveryClient,
    execute_recovery,
    _default_client,
)
from src.audit_logger import (
    AuditLogger,
    log_event,
    get_events,
    get_events_for_payment,
    _default_logger,
    EVENT_PAYMENT_DETECTED,
    EVENT_AI_ANALYZED,
    EVENT_POLICY_EVALUATED,
    EVENT_APPROVAL_REQUESTED,
    EVENT_RECOVERY_EXECUTION_STARTED,
    EVENT_RECOVERY_SUCCEEDED,
    EVENT_RECOVERY_FAILED,
    EVENT_RECOVERY_BLOCKED,
    EVENT_RECOVERY_STOPPED,
    EVENT_RECOVERY_ESCALATED,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    records: List[PaymentRecord],
    outcomes: List[Dict[str, Any]],
    approval_queue: Optional[MerchantApprovalQueue] = None,
    is_simulated: bool = True,
) -> BatchMetrics:
    """Computes exact, deterministic recovery metrics from processed outcomes."""
    total_records = len(records)
    total_failed_vol = sum(r.amount for r in records if r.payment_status == PaymentStatus.FAILED)
    total_at_risk_vol = sum(r.amount for r in records if r.payment_status == PaymentStatus.AT_RISK)

    # All records in the batch represent potential recovery opportunities
    total_opportunities = total_records

    actions_executed = 0
    successful_recoveries = 0
    failed_recoveries = 0
    blocked_actions = 0
    escalations = 0
    approvals_requested = 0

    total_recovered_revenue = 0.0
    incentive_amount_given = 0.0

    for item in outcomes:
        status = item.get("status")
        action = item.get("action")
        exec_res = item.get("execution")
        policy_res: Optional[PolicyResult] = item.get("policy_result")

        if status == "BLOCKED":
            blocked_actions += 1
        elif status == "ESCALATED":
            escalations += 1
        elif status == "PENDING_APPROVAL":
            approvals_requested += 1
        elif status == "STOPPED":
            pass  # Stopped safely, no recovery attempted

        # Check execution outcome
        if exec_res:
            if exec_res.success and action in (RecoveryAction.PAYMENT_LINK, RecoveryAction.RETRY, RecoveryAction.INCENTIVE):
                actions_executed += 1
                # Revenue is counted as RECOVERED only when status is RECOVERED or payment_confirmed is True
                if status == "RECOVERED" or item.get("payment_confirmed"):
                    successful_recoveries += 1
                    total_recovered_revenue += exec_res.amount

                    # Accumulate incentive discounts
                    if action == RecoveryAction.INCENTIVE and policy_res:
                        incentive_amount_given += policy_res.effective_discount
            elif not exec_res.success:
                failed_recoveries += 1

    # Check approval queue counts
    queue = approval_queue or _default_queue
    all_requests = queue.list_all_requests()
    approvals_granted = sum(1 for req in all_requests if req.status == ApprovalStatus.APPROVED)
    approvals_rejected = sum(1 for req in all_requests if req.status == ApprovalStatus.REJECTED)

    # Compute rates safely without division by zero
    if total_opportunities > 0:
        recovery_rate = min(100.0, round((successful_recoveries / total_opportunities) * 100.0, 2))
    else:
        recovery_rate = 0.0

    if successful_recoveries > 0:
        avg_recovered = round(total_recovered_revenue / successful_recoveries, 2)
    else:
        avg_recovered = 0.0

    net_recovered = max(0.0, round(total_recovered_revenue - incentive_amount_given, 2))

    if incentive_amount_given > 0:
        roi = round(net_recovered / incentive_amount_given, 2)
    else:
        roi = round(net_recovered, 2) if net_recovered > 0 else 0.0

    return BatchMetrics(
        total_records=total_records,
        total_failed_volume=round(total_failed_vol, 2),
        total_at_risk_volume=round(total_at_risk_vol, 2),
        total_recovery_opportunities=total_opportunities,
        actions_executed=actions_executed,
        successful_recoveries=successful_recoveries,
        failed_recoveries=failed_recoveries,
        blocked_actions=blocked_actions,
        escalations=escalations,
        approvals_requested=approvals_requested,
        approvals_granted=approvals_granted,
        approvals_rejected=approvals_rejected,
        total_recovered_revenue=round(total_recovered_revenue, 2),
        recovery_rate=recovery_rate,
        average_recovered_amount=avg_recovered,
        incentive_amount_given=round(incentive_amount_given, 2),
        net_recovered_revenue=net_recovered,
        roi=roi,
        is_simulated=is_simulated,
    )


def process_single_payment(
    payment: PaymentRecord,
    approved_by_merchant: bool = False,
    payment_confirmed: bool = False,
    agent: Optional[AIRecoveryAgent] = None,
    queue: Optional[MerchantApprovalQueue] = None,
    audit: Optional[AuditLogger] = None,
    rzp: Optional[RazorpayRecoveryClient] = None,
    force_fallback: bool = True,
) -> Dict[str, Any]:
    """Processes a single payment record through the entire governed lifecycle:
    
    PaymentRecord -> AI Analysis -> Policy Evaluation -> Approval Gate -> Razorpay -> Audit Logger.
    """
    ai_agent = agent or _default_agent
    approval_queue = queue or _default_queue
    logger_instance = audit or _default_logger
    rzp_client = rzp or _default_client

    pid = payment.payment_id

    # 1. Log Payment Ingestion
    logger_instance.log_event(
        payment_id=pid,
        event_type=EVENT_PAYMENT_DETECTED,
        status="INGESTED",
        message=f"Failed payment of INR {payment.amount:,.2f} detected ({payment.failure_reason.value}).",
        metadata={"amount": payment.amount, "reason": payment.failure_reason.value},
    )

    # 2. AI Context Analysis
    ai_rec: AIRecommendation = ai_agent.recommend(payment, force_fallback=force_fallback)
    logger_instance.log_event(
        payment_id=pid,
        event_type=EVENT_AI_ANALYZED,
        action=ai_rec.action.value,
        status=ai_rec.risk_level,
        message=f"AI recommends {ai_rec.action.value} ({ai_rec.agent_mode} mode, confidence: {ai_rec.confidence:.2f}): {ai_rec.reason}",
        metadata=ai_rec.model_dump(),
    )

    # 3. Deterministic Policy Evaluation
    policy_res: PolicyResult = evaluate_policy(
        payment,
        ai_rec.action,
        proposed_discount_pct=ai_rec.proposed_discount_pct,
    )
    logger_instance.log_event(
        payment_id=pid,
        event_type=EVENT_POLICY_EVALUATED,
        action=ai_rec.action.value,
        status=policy_res.decision.value,
        message=f"Policy decision: {policy_res.decision.value}. {policy_res.reason}",
        metadata=policy_res.model_dump(),
    )

    execution_result: Optional[RazorpayExecutionResult] = None
    outcome_status = "UNKNOWN"

    # 4. Routing based on Policy Decision
    if policy_res.decision == PolicyDecisionType.BLOCKED:
        outcome_status = "BLOCKED"
        logger_instance.log_event(
            payment_id=pid,
            event_type=EVENT_RECOVERY_BLOCKED,
            action=ai_rec.action.value,
            status="BLOCKED",
            message=f"Recovery action blocked by safety policy: {policy_res.reason}",
        )

    elif policy_res.requires_approval and not approved_by_merchant:
        outcome_status = "PENDING_APPROVAL"
        app_req = approval_queue.create_request(
            payment_id=pid,
            customer_id=payment.customer_id,
            amount=payment.amount,
            requested_action=ai_rec.action,
            ai_reason=ai_rec.reason,
            ai_confidence=ai_rec.confidence,
            policy_reason=policy_res.reason,
        )
        logger_instance.log_event(
            payment_id=pid,
            event_type=EVENT_APPROVAL_REQUESTED,
            action=ai_rec.action.value,
            status="PENDING",
            message=f"Approval request {app_req.approval_id} queued for transaction above INR 15,000 threshold.",

            metadata=app_req.model_dump(),
        )

    else:
        # Safe & Auto-approved OR Explicitly Approved by Merchant
        if ai_rec.action == RecoveryAction.STOP:
            outcome_status = "STOPPED"
            logger_instance.log_event(
                payment_id=pid,
                event_type=EVENT_RECOVERY_STOPPED,
                action=ai_rec.action.value,
                status="STOPPED",
                message=f"Recovery halted safely for {pid}.",
            )
            execution_result = rzp_client.execute_recovery(
                payment, ai_rec.action, policy_res, approved_by_merchant=approved_by_merchant
            )
        elif ai_rec.action == RecoveryAction.ESCALATE:
            outcome_status = "ESCALATED"
            logger_instance.log_event(
                payment_id=pid,
                event_type=EVENT_RECOVERY_ESCALATED,
                action=ai_rec.action.value,
                status="ESCALATED",
                message=f"Escalated to merchant account manager for {payment.customer_name}.",
            )
            execution_result = rzp_client.execute_recovery(
                payment, ai_rec.action, policy_res, approved_by_merchant=approved_by_merchant
            )
        else:
            # Financial Actions: RETRY, PAYMENT_LINK, INCENTIVE (or REMINDER)
            logger_instance.log_event(
                payment_id=pid,
                event_type=EVENT_RECOVERY_EXECUTION_STARTED,
                action=ai_rec.action.value,
                status="DISPATCHING",
                message=f"Dispatching {ai_rec.action.value} to Razorpay gateway.",
            )

            execution_result = rzp_client.execute_recovery(
                payment, ai_rec.action, policy_res, approved_by_merchant=approved_by_merchant
            )

            if execution_result.success:
                # Differentiate between action execution (link/order created) vs confirmed recovery
                if payment_confirmed or (rzp_client.is_mock and ai_rec.action in (RecoveryAction.PAYMENT_LINK, RecoveryAction.RETRY, RecoveryAction.INCENTIVE)):
                    outcome_status = "RECOVERED"
                else:
                    outcome_status = "ACTION_EXECUTED"

                logger_instance.log_event(
                    payment_id=pid,
                    event_type=EVENT_RECOVERY_SUCCEEDED if outcome_status == "RECOVERED" else "ACTION_EXECUTED",
                    action=ai_rec.action.value,
                    status="SUCCESS",
                    message=f"{'Simulated recovery confirmed: ' if outcome_status == 'RECOVERED' else 'Recovery action executed (Payment Link / Order created): '}{execution_result.message}",
                    recovery_id=execution_result.recovery_id,
                    metadata=execution_result.model_dump(),
                )
            else:
                outcome_status = "FAILED"
                logger_instance.log_event(
                    payment_id=pid,
                    event_type=EVENT_RECOVERY_FAILED,
                    action=ai_rec.action.value,
                    status="FAILED",
                    message=execution_result.message,
                    recovery_id=execution_result.recovery_id,
                )

    return {
        "payment_id": pid,
        "amount": payment.amount,
        "status": outcome_status,
        "action": ai_rec.action,
        "ai_recommendation": ai_rec,
        "policy_result": policy_res,
        "execution": execution_result,
    }


def run_batch_recovery(
    records: Optional[List[PaymentRecord]] = None,
    dataset_path: str = "data/synthetic_payments.json",
    approval_queue: Optional[MerchantApprovalQueue] = None,
    audit_logger: Optional[AuditLogger] = None,
    rzp_client: Optional[RazorpayRecoveryClient] = None,
    agent: Optional[AIRecoveryAgent] = None,
    force_fallback: bool = True,
) -> Tuple[BatchMetrics, List[Dict[str, Any]]]:
    """Runs autonomous recovery simulation across the 500+ records dataset.
    
    Executes safe auto-approved actions, queues high-value records for merchant review,
    blocks unsafe actions, and aggregates financial recovery KPIs.
    """
    queue = approval_queue or _default_queue
    logger_instance = audit_logger or _default_logger
    client = rzp_client or _default_client
    active_agent = agent or _default_agent

    if records is None:
        p = Path(dataset_path)
        if not p.exists():
            p = Path(__file__).resolve().parent.parent / dataset_path
        if not p.exists():
            raise FileNotFoundError(f"Synthetic dataset not found at {dataset_path}")
        with open(p, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        records = [PaymentRecord.model_validate(item) for item in raw_data]

    outcomes: List[Dict[str, Any]] = []

    for payment in records:
        try:
            res = process_single_payment(
                payment=payment,
                approved_by_merchant=False,  # Automated batch leaves high-value pending
                agent=active_agent,
                queue=queue,
                audit=logger_instance,
                rzp=client,
                force_fallback=force_fallback,
            )
            outcomes.append(res)
        except Exception as e:
            logger.error("Error processing payment %s: %s", payment.payment_id, e)
            logger_instance.log_event(
                payment_id=payment.payment_id,
                event_type="BATCH_ERROR",
                status="ERROR",
                message=f"Pipeline exception: {e}",
            )
            outcomes.append({
                "payment_id": payment.payment_id,
                "amount": payment.amount,
                "status": "ERROR",
                "action": None,
                "ai_recommendation": None,
                "policy_result": None,
                "execution": None,
            })

    metrics = calculate_metrics(
        records=records,
        outcomes=outcomes,
        approval_queue=queue,
        is_simulated=client.is_mock,
    )

    return metrics, outcomes


# Integration API helpers for Streamlit
def analyze_single_payment(payment: PaymentRecord) -> AIRecommendation:
    """UI Helper: AI diagnosis only."""
    return analyze_payment(payment)


def get_pending_approvals(queue: Optional[MerchantApprovalQueue] = None) -> List[Any]:
    """UI Helper: Pending approval requests."""
    active_queue = queue or _default_queue
    return active_queue.list_pending_requests()


def get_audit_log(limit: Optional[int] = None, audit: Optional[AuditLogger] = None) -> List[Any]:
    """UI Helper: Chronological audit events."""
    active_audit = audit or _default_logger
    return active_audit.get_events(limit=limit, reverse=True)
