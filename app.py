"""RecoverPay AI - Interactive Streamlit Command Center Dashboard.

Track 03: AI Revenue Recovery (Razorpay AI Buildathon)
Autonomous, policy-governed revenue recovery engine for Razorpay merchants.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    ProductCategory,
    RecoveryAction,
    PolicyDecisionType,
    PolicyResult,
    ApprovalStatus,
    ApprovalRequest,
    BatchMetrics,
    AIRecommendation,
    RazorpayExecutionResult,
)
from src.ai_agent import AIRecoveryAgent, heuristic_fallback_recommend
from src.policy_engine import evaluate_policy
from src.merchant_approval import MerchantApprovalQueue
from src.audit_logger import (
    AuditLogger,
    EVENT_PAYMENT_DETECTED,
    EVENT_AI_ANALYZED,
    EVENT_POLICY_EVALUATED,
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_GRANTED,
    EVENT_APPROVAL_REJECTED,
    EVENT_RECOVERY_EXECUTION_STARTED,
    EVENT_RECOVERY_SUCCEEDED,
    EVENT_RECOVERY_FAILED,
    EVENT_RECOVERY_BLOCKED,
    EVENT_RECOVERY_STOPPED,
    EVENT_RECOVERY_ESCALATED,
)
from src.razorpay_client import RazorpayRecoveryClient
from src.metrics import calculate_metrics, run_batch_recovery, process_single_payment

load_dotenv()

# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Light Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RecoverPay AI | Merchant Command Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern Razorpay/fintech aesthetic
st.markdown(
    """
    <style>
    /* Main Layout & Fonts */
    .main { background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff !important;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .top-header h1, .top-header .top-title {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.3 !important;
        border: none !important;
    }
    .top-subtitle { font-size: 0.9rem; color: #64748b !important; margin-top: 0.2rem; }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    .badge-mock { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-test { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .badge-gemini { background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; }
    .badge-fallback { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .badge-allowed { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .badge-blocked { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .badge-approval { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .metric-label { font-size: 0.7rem; font-weight: 600; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-val { font-size: 1.35rem; font-weight: 700; color: #0f172a !important; margin: 0.2rem 0; }
    .metric-caption { font-size: 0.7rem; color: #94a3b8 !important; }
    
    /* Flow Tracker */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
    }
    .flow-step { text-align: center; flex: 1; }
    .flow-step-title { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; }
    .flow-step-status { font-size: 0.9rem; font-weight: 700; margin-top: 0.2rem; }
    .flow-arrow { color: #cbd5e1; font-weight: 700; font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization (Persistent across UI reruns)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> List[PaymentRecord]:
    """Loads 500+ records once from synthetic_payments.json."""
    dataset_path = Path("data/synthetic_payments.json")
    if not dataset_path.exists():
        from src.data_generator import generate_synthetic_dataset
        return generate_synthetic_dataset(500, output_path=dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [PaymentRecord.model_validate(item) for item in raw]


if "records" not in st.session_state:
    st.session_state.records = load_dataset()

if "queue" not in st.session_state:
    st.session_state.queue = MerchantApprovalQueue()

if "audit" not in st.session_state:
    st.session_state.audit = AuditLogger()

if "rzp_client" not in st.session_state:
    st.session_state.rzp_client = RazorpayRecoveryClient()

if "ai_agent" not in st.session_state:
    st.session_state.ai_agent = AIRecoveryAgent()

if "outcomes" not in st.session_state:
    # Maps payment_id -> outcome dict
    st.session_state.outcomes = {}

if "batch_metrics" not in st.session_state:
    # Precompute baseline metrics
    st.session_state.batch_metrics = None


# ---------------------------------------------------------------------------
# Precompute initial diagnoses fast via Heuristic for snappy table loading
# ---------------------------------------------------------------------------
@st.cache_data
def get_precomputed_diagnoses() -> Dict[str, Dict[str, Any]]:
    records = load_dataset()
    results = {}
    for r in records:
        ai_rec = heuristic_fallback_recommend(r)
        policy_res = evaluate_policy(r, ai_rec.action, proposed_discount_pct=ai_rec.proposed_discount_pct)
        results[r.payment_id] = {
            "action": ai_rec.action.value,
            "confidence": ai_rec.confidence,
            "policy": policy_res.decision.value,
            "status": "AT_RISK" if r.payment_status == PaymentStatus.AT_RISK else "FAILED",
        }
    return results

precomputed = get_precomputed_diagnoses()


# ---------------------------------------------------------------------------
# TOP HEADER COMPONENT
# ---------------------------------------------------------------------------
client: RazorpayRecoveryClient = st.session_state.rzp_client
agent: AIRecoveryAgent = st.session_state.ai_agent

rzp_badge_class = "badge-mock" if client.is_mock else "badge-test"
rzp_badge_text = "MOCK SANDBOX • SIMULATED" if client.is_mock else "RAZORPAY TEST MODE"

ai_badge_class = "badge-gemini" if agent._client is not None else "badge-fallback"
ai_badge_text = "AI: GEMINI" if agent._client is not None else "AI: FALLBACK (HEURISTIC)"

st.markdown(
    f"""
    <div class="top-header">
        <div>
            <h1 class="top-title">RecoverPay AI</h1>
            <div class="top-subtitle">Autonomous AI Revenue Recovery Engine for Razorpay Merchants</div>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
            <span class="badge {rzp_badge_class}">{rzp_badge_text}</span>
            <span class="badge {ai_badge_class}">{ai_badge_text}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP KPI SCORECARD ROW
# ---------------------------------------------------------------------------
records = st.session_state.records
total_records = len(records)
total_failed_vol = sum(r.amount for r in records)

# Calculate live metrics
live_metrics: BatchMetrics = calculate_metrics(
    records=records,
    outcomes=list(st.session_state.outcomes.values()),
    approval_queue=st.session_state.queue,
    is_simulated=client.is_mock,
)

col1, col2, col3, col4, col5 = st.columns(5, gap="small")

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Revenue At Risk</div>
            <div class="metric-val">INR {total_failed_vol:,.0f}</div>
            <div class="metric-caption">{total_records} Failed / At-Risk Orders</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    actions_exec = getattr(live_metrics, "actions_executed", 0)
    if client.is_mock:
        rev_label = "Simulated Recovered Revenue"
        caption_text = f"{live_metrics.successful_recoveries} Simulated Recoveries • {actions_exec} Links Dispatched"
    else:
        rev_label = "Confirmed Recovered Revenue"
        caption_text = f"{live_metrics.successful_recoveries} Confirmed Paid • {actions_exec} Links Dispatched"

    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-label">{rev_label}</div>
            <div class="metric-val" style="color: #059669;">INR {live_metrics.total_recovered_revenue:,.0f}</div>
            <div class="metric-caption">{caption_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Recovery Rate</div>
            <div class="metric-val" style="color: #2563eb;">{live_metrics.recovery_rate:.1f}%</div>
            <div class="metric-caption">Of total failed opportunities</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    pending_count = len(st.session_state.queue.list_pending_requests())
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Pending Approvals</div>
            <div class="metric-val" style="color: #d97706;">{pending_count}</div>
            <div class="metric-caption">Transactions > INR 15,000</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Blocked / Escalated</div>
            <div class="metric-val" style="color: #dc2626;">{live_metrics.blocked_actions + live_metrics.escalations}</div>
            <div class="metric-caption">Fraud & safety guardrails</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------------------------
tab_overview, tab_queue, tab_workbench, tab_approval, tab_audit, tab_batch = st.tabs([
    "📊 Overview",
    "📋 Recovery Queue",
    "⚡ AI Workbench",
    "🛡️ Approval Queue",
    "📜 Audit Trail",
    "🚀 Batch Simulation",
])


# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab_overview:
    st.subheader("Autonomous Revenue Recovery Overview")
    st.caption("Real-time telemetry showing recovery performance, action distribution, and root cause failure breakdown.")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Action Distribution Chart
        actions_list = [precomputed[r.payment_id]["action"] for r in records]
        action_df = pd.DataFrame({"Action": actions_list})
        action_counts = action_df["Action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]

        fig_action = px.bar(
            action_counts,
            x="Action",
            y="Count",
            color="Action",
            color_discrete_map={
                "PAYMENT_LINK": "#3b82f6",
                "RETRY": "#10b981",
                "INCENTIVE": "#8b5cf6",
                "ESCALATE": "#f59e0b",
                "STOP": "#ef4444",
                "REMINDER": "#06b6d4",
            },
            title="AI Recovery Strategy Distribution",
        )
        fig_action.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_action, use_container_width=True)

    with chart_col2:
        # Failure Reason Distribution Chart
        reasons_list = [r.failure_reason.value for r in records]
        reason_df = pd.DataFrame({"Reason": reasons_list})
        reason_counts = reason_df["Reason"].value_counts().reset_index()
        reason_counts.columns = ["Failure Reason", "Count"]

        fig_reason = px.pie(
            reason_counts,
            names="Failure Reason",
            values="Count",
            hole=0.45,
            title="Gateway Failure Root Cause Breakdown",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_reason.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_reason, use_container_width=True)

    st.markdown("---")

    # Financial Impact Breakdown
    st.subheader("Financial Impact & Recovery Health")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.info(
            f"**Total Gross Failed Volume**: INR {total_failed_vol:,.2f}\n\n"
            f"**Average Order Value**: INR {total_failed_vol/total_records:,.2f}"
        )
    with sum_col2:
        st.success(
            f"**Net Recovered Volume**: INR {live_metrics.net_recovered_revenue:,.2f}\n\n"
            f"**Incentives Disbursed**: INR {live_metrics.incentive_amount_given:,.2f} (Max 10% cap)"
        )
    with sum_col3:
        st.warning(
            f"**High-Value Transactions Held**: {live_metrics.approvals_requested}\n\n"
            f"**Terminal Stops Enforced**: {sum(1 for r in records if precomputed[r.payment_id]['action'] == 'STOP')}"
        )


# ===========================================================================
# TAB 2: RECOVERY QUEUE & TRANSACTION DETAILS
# ===========================================================================
with tab_queue:
    st.subheader("Payment Failure Ingestion Queue (500 Transactions)")
    st.caption("Search, filter, inspect, and execute policy-governed recoveries for individual merchant transactions.")

    # Search & Filter Controls
    filt_c1, filt_c2, filt_c3, filt_c4 = st.columns(4)
    with filt_c1:
        reason_filter = st.selectbox("Filter by Failure Reason", ["All"] + [r.value for r in FailureReason])
    with filt_c2:
        action_filter = st.selectbox("Filter by AI Action", ["All"] + [a.value for a in RecoveryAction])
    with filt_c3:
        policy_filter = st.selectbox("Filter by Policy Decision", ["All", "ALLOWED", "REQUIRES_APPROVAL", "BLOCKED"])
    with filt_c4:
        amount_filter = st.selectbox("Filter by Amount Range", ["All", "Under INR 5,000", "INR 5,000 - 15,000", "High Value (> INR 15,000)"])

    # Build DataFrame
    table_data = []
    for r in records:
        diag = precomputed[r.payment_id]
        act = diag["action"]
        pol = diag["policy"]
        
        # Check if already processed in session
        if r.payment_id in st.session_state.outcomes:
            current_status = st.session_state.outcomes[r.payment_id]["status"]
        else:
            current_status = r.payment_status.value

        # Apply filters
        if reason_filter != "All" and r.failure_reason.value != reason_filter:
            continue
        if action_filter != "All" and act != action_filter:
            continue
        if policy_filter != "All" and pol != policy_filter:
            continue
        if amount_filter == "Under INR 5,000" and r.amount >= 5000:
            continue
        if amount_filter == "INR 5,000 - 15,000" and (r.amount < 5000 or r.amount > 15000):
            continue
        if amount_filter == "High Value (> INR 15,000)" and r.amount <= 15000:
            continue

        table_data.append({
            "Payment ID": r.payment_id,
            "Customer": r.customer_name,
            "Amount": f"INR {r.amount:,.2f}",
            "Raw Amount": r.amount,
            "Failure Reason": r.failure_reason.value,
            "Retries": r.retry_count,
            "Customer History": f"{r.customer_previous_payments} orders (INR {r.customer_total_spend:,.0f})",
            "AI Action": act,
            "Confidence": f"{diag['confidence']*100:.0f}%",
            "Policy Decision": pol,
            "Status": current_status,
        })

    table_df = pd.DataFrame(table_data)

    if not table_df.empty:
        st.dataframe(
            table_df.drop(columns=["Raw Amount"]),
            use_container_width=True,
            height=300,
            hide_index=True,
        )
    else:
        st.warning("No records match the selected filters.")

    st.markdown("---")

    # Transaction Detail Selection
    st.subheader("🔍 Transaction Recovery Detail & Workbench")
    selected_pid = st.selectbox(
        "Select a Payment ID to inspect and recover:",
        options=[r.payment_id for r in records],
        index=0,
    )

    selected_payment = next(r for r in records if r.payment_id == selected_pid)

    # 1. Horizontal Workflow Centerpiece
    has_outcome = selected_pid in st.session_state.outcomes
    current_outcome = st.session_state.outcomes.get(selected_pid, {})
    exec_res = current_outcome.get("execution")

    ai_rec_curr = agent.recommend(selected_payment)
    policy_curr = evaluate_policy(selected_payment, ai_rec_curr.action, proposed_discount_pct=ai_rec_curr.proposed_discount_pct)

    appr_status_text = "NOT REQUIRED" if not policy_curr.requires_approval else ("APPROVED" if st.session_state.queue.is_approved(selected_pid) else "PENDING")
    rzp_status_text = "EXECUTED" if exec_res and exec_res.success else ("BLOCKED" if policy_curr.decision == PolicyDecisionType.BLOCKED else "PENDING")
    final_status_text = current_outcome.get("status", selected_payment.payment_status.value)

    st.markdown(
        f"""
        <div class="workflow-container">
            <div class="flow-step">
                <div class="flow-step-title">1. Detection</div>
                <div class="flow-step-status" style="color: #059669;">✓ Ingested</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-step-title">2. AI Diagnosis</div>
                <div class="flow-step-status" style="color: #2563eb;">✓ {ai_rec_curr.action.value}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-step-title">3. Policy Gate</div>
                <div class="flow-step-status" style="color: {'#059669' if policy_curr.decision == PolicyDecisionType.ALLOWED else '#d97706'};">✓ {policy_curr.decision.value}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-step-title">4. Merchant Review</div>
                <div class="flow-step-status" style="color: {'#059669' if appr_status_text != 'PENDING' else '#d97706'};">{appr_status_text}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-step-title">5. Razorpay</div>
                <div class="flow-step-status">{rzp_status_text}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="flow-step-title">6. Audit Trail</div>
                <div class="flow-step-status" style="color: #059669;">✓ Logged</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Detail Columns
    det_c1, det_c2, det_c3 = st.columns([1.2, 1.4, 1.4])

    with det_c1:
        st.markdown("**Transaction & Customer Profile**")
        st.write(f"• **Customer**: {selected_payment.customer_name} (`{selected_payment.customer_id}`)")
        st.write(f"• **Amount**: INR {selected_payment.amount:,.2f}")
        st.write(f"• **Order ID**: `{selected_payment.order_id}`")
        st.write(f"• **Category**: {selected_payment.product_category.value}")
        st.write(f"• **Failure Code**: `{selected_payment.failure_reason.value}`")
        st.write(f"• **Error Message**: {selected_payment.error_description}")
        st.write(f"• **Prior Retries**: {selected_payment.retry_count} (Max 2)")
        st.write(f"• **Past Spend**: INR {selected_payment.customer_total_spend:,.2f} ({selected_payment.customer_previous_payments} orders)")

    with det_c2:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 1rem;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #1e40af; text-transform: uppercase;">
                    🤖 AI Proposal (Advisory Only)
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #1e3a8a; margin: 0.3rem 0;">
                    {ai_rec_curr.action.value}
                </div>
                <p style="font-size: 0.85rem; color: #334155; margin: 0.4rem 0;">{ai_rec_curr.reason}</p>
                <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;"/>
                <div style="font-size: 0.8rem; color: #64748b;">
                    <b>Confidence</b>: {ai_rec_curr.confidence*100:.0f}% &nbsp;|&nbsp; 
                    <b>Risk</b>: {ai_rec_curr.risk_level} &nbsp;|&nbsp; 
                    <b>Discount</b>: {ai_rec_curr.proposed_discount_pct}%<br/>
                    <b>Mode</b>: {ai_rec_curr.agent_mode.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with det_c3:
        pol_color = "#15803d" if policy_curr.decision == PolicyDecisionType.ALLOWED else ("#b45309" if policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL else "#b91c1c")
        pol_bg = "#f0fdf4" if policy_curr.decision == PolicyDecisionType.ALLOWED else ("#fffbeb" if policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL else "#fef2f2")
        pol_border = "#bbf7d0" if policy_curr.decision == PolicyDecisionType.ALLOWED else ("#fde68a" if policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL else "#fecaca")

        st.markdown(
            f"""
            <div style="background: {pol_bg}; border: 1px solid {pol_border}; border-radius: 8px; padding: 1rem;">
                <div style="font-size: 0.75rem; font-weight: 700; color: {pol_color}; text-transform: uppercase;">
                    ⚖️ Deterministic Policy Engine (Authoritative)
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {pol_color}; margin: 0.3rem 0;">
                    {policy_curr.decision.value}
                </div>
                <p style="font-size: 0.85rem; color: #334155; margin: 0.4rem 0;">{policy_curr.reason}</p>
                <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid {pol_border};"/>
                <div style="font-size: 0.8rem; color: #64748b;">
                    <b>Policy Codes</b>: {", ".join(policy_curr.policy_codes)}<br/>
                    <b>Final Payable</b>: INR {policy_curr.final_payable_amount:,.2f} (Discount: INR {policy_curr.effective_discount:,.2f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # 3. Action Execution Controls
    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        if policy_curr.decision == PolicyDecisionType.ALLOWED:
            st.success(f"Action '{ai_rec_curr.action.value}' is fully permitted by safety rules.")
            if st.button(f"⚡ Execute Recovery: {ai_rec_curr.action.value}", key=f"exec_{selected_pid}"):
                res = process_single_payment(
                    payment=selected_payment,
                    approved_by_merchant=False,
                    agent=agent,
                    queue=st.session_state.queue,
                    audit=st.session_state.audit,
                    rzp=client,
                    force_fallback=True,
                )
                st.session_state.outcomes[selected_pid] = res
                st.rerun()

        elif policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL:
            st.warning("⚠️ High-value transaction requires merchant approval before execution.")
            is_appr = st.session_state.queue.is_approved(selected_pid)
            if not is_appr:
                if st.button("🛡️ Route to Merchant Approval Queue", key=f"queue_{selected_pid}"):
                    st.session_state.queue.create_request(
                        payment_id=selected_pid,
                        customer_id=selected_payment.customer_id,
                        amount=selected_payment.amount,
                        requested_action=ai_rec_curr.action,
                        ai_reason=ai_rec_curr.reason,
                        ai_confidence=ai_rec_curr.confidence,
                        policy_reason=policy_curr.reason,
                    )
                    st.session_state.audit.log_event(
                        payment_id=selected_pid,
                        event_type=EVENT_APPROVAL_REQUESTED,
                        action=ai_rec_curr.action.value,
                        status="PENDING",
                        message=f"Approval request queued for high-value transaction INR {selected_payment.amount:,.2f}.",
                    )
                    st.success("Successfully queued for review in Tab 4 (Approval Queue)!")
                    st.rerun()
            else:
                st.success("Merchant approval granted! Safe to execute.")
                if st.button(f"⚡ Execute Approved Recovery: {ai_rec_curr.action.value}", key=f"exec_appr_{selected_pid}"):
                    res = process_single_payment(
                        payment=selected_payment,
                        approved_by_merchant=True,
                        agent=agent,
                        queue=st.session_state.queue,
                        audit=st.session_state.audit,
                        rzp=client,
                        force_fallback=True,
                    )
                    st.session_state.outcomes[selected_pid] = res
                    st.rerun()

        else:  # BLOCKED
            st.error(f"🛑 RECOVERY BLOCKED: {policy_curr.reason}")
            st.caption("Safety policy strictly forbids money movement on fraudulent, exhausted, or invalid transactions.")

    with ctrl_col2:
        if exec_res:
            st.markdown("**Recovery Action Executed:**")
            if exec_res.success:
                st.success(f"✓ Recovery Action Executed: {exec_res.action.value}")
                st.json({
                    "Recovery Attempt ID": exec_res.recovery_id,
                    "Razorpay Link/Order ID": exec_res.razorpay_id,
                    "Execution Status": "Recovery Action Executed (Link Dispatched)" if not exec_res.simulated else "Simulated Action Executed",
                    "Amount": f"INR {exec_res.amount:,.2f}",
                    "Short URL": exec_res.short_url,
                    "Simulated": exec_res.simulated,
                })
            else:
                st.error(f"Execution Error: {exec_res.message}")


# ===========================================================================
# TAB 3: AI WORKBENCH & DEMO SCENARIOS
# ===========================================================================
with tab_workbench:
    st.subheader("⚡ AI Recovery Workbench & Judging Demo Scenarios")
    st.caption("Curated scenarios engineered for the 5-minute hackathon pitch to showcase AI-Policy synergy.")

    demo_scenario = st.radio(
        "Select an automated demonstration scenario:",
        [
            "1. Standard Autonomous Recovery (Transient Timeout)",
            "2. High-Value B2B Order Gate (> INR 15,000 Approval)",
            "3. Suspected Fraud Lockout (Terminal Security Risk)",
            "4. Exhausted Retry Limit (Max 2 Attempts Reached)",
        ],
        horizontal=True,
    )

    if "Standard" in demo_scenario:
        demo_record = next(r for r in records if r.amount < 5000 and r.retry_count == 0 and r.failure_reason == FailureReason.NETWORK_TIMEOUT and r.customer_previous_payments > 0)
        demo_desc = "Transient network drop for an established customer. Safe for immediate automated recovery link."
    elif "High-Value" in demo_scenario:
        demo_record = next(r for r in records if r.amount > 15000 and agent.recommend(r).action == RecoveryAction.PAYMENT_LINK)
        demo_desc = "High-value B2B order (> ₹15,000). AI recommends PAYMENT_LINK, but policy engine intercepts and requires merchant approval."
    elif "Fraud" in demo_scenario:
        demo_record = next(r for r in records if r.failure_reason == FailureReason.SUSPECTED_FRAUD)
        demo_desc = "Risk engine detected suspected fraud. All automated recovery and discounts are permanently BLOCKED."
    else:  # Exhausted
        demo_record = next(r for r in records if r.retry_count >= 2 and r.amount < 15000)
        demo_desc = "Customer has already suffered 2 prior failures. Retry limit reached; policy engine forces STOP to protect merchant reputation."

    st.info(f"**Scenario Context**: {demo_desc}")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.markdown("**1. Raw Ingested Payment**")
        st.write(f"• ID: `{demo_record.payment_id}`")
        st.write(f"• Amount: **INR {demo_record.amount:,.2f}**")
        st.write(f"• Reason: `{demo_record.failure_reason.value}`")
        st.write(f"• Retries: **{demo_record.retry_count} / 2**")
        st.write(f"• Customer: {demo_record.customer_name}")

    demo_ai = agent.recommend(demo_record)
    demo_policy = evaluate_policy(demo_record, demo_ai.action, proposed_discount_pct=demo_ai.proposed_discount_pct)

    with col_w2:
        st.markdown("**2. AI Recommendation**")
        st.write(f"• Action: **{demo_ai.action.value}**")
        st.write(f"• Confidence: **{demo_ai.confidence*100:.0f}%**")
        st.write(f"• Risk Level: `{demo_ai.risk_level}`")
        st.write(f"• Rationale: {demo_ai.reason}")

    with col_w3:
        st.markdown("**3. Policy Engine Verdict**")
        st.write(f"• Decision: **{demo_policy.decision.value}**")
        st.write(f"• Allowed: **{demo_policy.allowed}**")
        st.write(f"• Requires Approval: **{demo_policy.requires_approval}**")
        st.write(f"• Policy Codes: `{demo_policy.policy_codes}`")
        st.write(f"• Explanation: {demo_policy.reason}")

    if demo_policy.requires_approval:
        st.warning(f"⚠️ **Approval Rationale**: {demo_policy.reason}")

    st.markdown("---")

    # Deliberate Safety Failure Playground Button
    st.subheader("🚨 Deliberate Safety Failure Playground")
    st.caption("Demonstrate to judges that an unauthorized prompt or malicious discount cannot bypass the deterministic safety layer.")

    fail_col1, fail_col2 = st.columns(2)
    with fail_col1:
        sim_action = st.selectbox("Simulate AI Action Proposal:", [RecoveryAction.RETRY.value, RecoveryAction.INCENTIVE.value, "UNAUTHORIZED_SWIFT_TRANSFER"])
        sim_discount = st.slider("Simulate Proposed Discount Percentage:", 0, 50, 25)
    with fail_col2:
        test_viol = evaluate_policy(demo_record, sim_action, proposed_discount_pct=float(sim_discount))
        st.markdown("**Policy Engine Reaction:**")
        if test_viol.decision == PolicyDecisionType.BLOCKED:
            st.error(f"❌ VIOLATION INTERCEPTED & BLOCKED:\n\n{test_viol.reason}")
        elif test_viol.decision == PolicyDecisionType.REQUIRES_APPROVAL:
            st.warning(f"⚠️ APPROVAL FORCED BY SAFETY POLICY:\n\n{test_viol.reason}")
        else:
            st.success(f"✅ ALLOWED:\n\n{test_viol.reason}")


# ===========================================================================
# TAB 4: APPROVAL QUEUE (HUMAN-IN-THE-LOOP)
# ===========================================================================
with tab_approval:
    st.subheader("🛡️ Merchant Review & Approval Inbox")
    st.caption("High-value payments (> INR 15,000) or complex risk profiles strictly require human sign-off before gateway execution.")

    queue_instance: MerchantApprovalQueue = st.session_state.queue
    pending_reqs = queue_instance.list_pending_requests()

    if not pending_reqs:
        st.success("🎉 Inbox Zero: All high-value transactions have been reviewed and resolved!")
    else:
        st.warning(f"Currently **{len(pending_reqs)} transactions** are awaiting merchant review.")

        for req in pending_reqs:
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 1.1rem; color: #0f172a;">Request ID: {req.approval_id}</span>
                            <span class="badge badge-approval">PENDING REVIEW</span>
                        </div>
                        <p style="margin: 0.4rem 0; color: #475569;">
                            <b>Payment ID</b>: <code>{req.payment_id}</code> &nbsp;|&nbsp; 
                            <b>Amount</b>: <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">INR {req.amount:,.2f}</span> &nbsp;|&nbsp; 
                            <b>Customer</b>: {req.customer_id}
                        </p>
                        <p style="margin: 0.2rem 0; font-size: 0.9rem;"><b>AI Proposal</b>: Recommends <b>{req.requested_action.value}</b> (Confidence: {req.ai_confidence*100:.0f}%)<br/><i>"{req.ai_reason}"</i></p>
                        <p style="margin: 0.2rem 0; font-size: 0.9rem; color: #b45309;"><b>Policy Gate Reason</b>: {req.policy_reason}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                appr_col1, appr_col2, appr_col3 = st.columns([1.5, 1.5, 3])
                with appr_col1:
                    if st.button(f"✅ Approve & Dispatch Link", key=f"appr_{req.approval_id}"):
                        queue_instance.approve_request(req.approval_id, reviewer="Merchant Admin", notes="Approved via Dashboard")
                        
                        # Find original payment record
                        target_p = next(r for r in records if r.payment_id == req.payment_id)
                        pol = evaluate_policy(target_p, req.requested_action)
                        res = client.execute_recovery(target_p, req.requested_action, pol, approved_by_merchant=True)
                        
                        st.session_state.outcomes[req.payment_id] = {
                            "payment_id": req.payment_id,
                            "amount": req.amount,
                            "status": "RECOVERED" if res.success else "FAILED",
                            "action": req.requested_action,
                            "execution": res,
                            "policy_result": pol,
                        }
                        st.session_state.audit.log_event(
                            payment_id=req.payment_id,
                            event_type=EVENT_APPROVAL_GRANTED,
                            action=req.requested_action.value,
                            status="APPROVED",
                            message=f"Merchant approved recovery for {req.payment_id}.",
                        )
                        st.success(f"Approved and executed! Link: {res.short_url}")
                        st.rerun()

                with appr_col2:
                    if st.button(f"❌ Reject Recovery", key=f"rej_{req.approval_id}"):
                        queue_instance.reject_request(req.approval_id, reviewer="Merchant Admin", notes="Rejected by merchant")
                        st.session_state.audit.log_event(
                            payment_id=req.payment_id,
                            event_type=EVENT_APPROVAL_REJECTED,
                            action=req.requested_action.value,
                            status="REJECTED",
                            message=f"Merchant rejected recovery for {req.payment_id}.",
                        )
                        st.info("Request rejected. No payment link created.")
                        st.rerun()


# ===========================================================================
# TAB 5: AUDIT TRAIL
# ===========================================================================
with tab_audit:
    st.subheader("📜 Append-Only Chronological Audit Trail")
    st.caption("Immutable ledger providing complete provenance and state reconstruction for financial and regulatory compliance.")

    audit_instance: AuditLogger = st.session_state.audit
    events = audit_instance.get_events(reverse=True)

    filter_pid = st.text_input("Filter audit events by Payment ID (leave blank for all):", "")

    if filter_pid.strip():
        events = [e for e in events if filter_pid.strip().lower() in e.payment_id.lower()]

    if not events:
        st.info("No audit events recorded yet. Run a single transaction recovery or batch simulation to populate the ledger.")
    else:
        audit_rows = []
        for ev in events:
            audit_rows.append({
                "Timestamp": ev.timestamp,
                "Event Type": ev.event_type,
                "Payment ID": ev.payment_id,
                "Action": ev.action or "-",
                "Status": ev.status,
                "Message": ev.message,
                "Recovery ID": ev.recovery_id or "-",
            })

        st.dataframe(
            pd.DataFrame(audit_rows),
            use_container_width=True,
            height=400,
            hide_index=True,
        )


# ===========================================================================
# TAB 6: BATCH SIMULATION
# ===========================================================================
with tab_batch:
    st.subheader("🚀 500-Record Autonomous Recovery Batch Simulation")
    st.caption("Run the complete governed recovery pipeline across all 500 synthetic payment failures to measure portfolio financial recovery and net ROI.")

    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <b>Simulation Mode</b>: Safe auto-approved actions (< ₹15,000, 0 retries, non-fraud) are executed through the 
            <b>Razorpay Mock Sandbox</b>. High-value transactions (> ₹15,000) are routed to the <b>Merchant Approval Queue</b> and 
            never count as recovered until reviewed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("▶️ RUN 500-RECORD RECOVERY SIMULATION", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_msg = st.empty()

        status_msg.text("Initializing pipeline and ingesting 500 failure records...")
        progress_bar.progress(15)

        status_msg.text("Executing AI context analysis and deterministic policy evaluations...")
        progress_bar.progress(50)

        metrics_result, batch_outcomes = run_batch_recovery(
            records=records,
            approval_queue=st.session_state.queue,
            audit_logger=st.session_state.audit,
            rzp_client=client,
            force_fallback=True,
        )

        for out in batch_outcomes:
            st.session_state.outcomes[out["payment_id"]] = out

        st.session_state.batch_metrics = metrics_result
        progress_bar.progress(100)
        status_msg.success("✅ 500 Records Processed Successfully!")
        st.rerun()

    # Display Batch Simulation Scorecard
    if st.session_state.batch_metrics:
        bm: BatchMetrics = st.session_state.batch_metrics
        st.markdown("### 📊 Measured Batch Recovery KPIs")

        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Total Records Analyzed", f"{bm.total_records:,}")
            st.metric("Total Failed Volume", f"INR {bm.total_failed_volume:,.2f}")
        with b_col2:
            st.metric("Automatically Recovered", f"{bm.successful_recoveries:,}", f"{bm.recovery_rate:.1f}% Recovery Rate")
            st.metric("Simulated Recovered Revenue", f"INR {bm.total_recovered_revenue:,.2f}")
        with b_col3:
            st.metric("Held in Approval Queue", f"{bm.approvals_requested:,}", "Zero money counted until approved")
            st.metric("Net Recovered Revenue", f"INR {bm.net_recovered_revenue:,.2f}")
        with b_col4:
            st.metric("Terminal Safety Halts", f"{bm.blocked_actions + bm.escalations:,}", "Fraud & Retry Limit")
            st.metric("Financial ROI Multiple", f"{bm.roi:,.1f}x")

        st.caption(
            "⚠️ **SIMULATED RESULTS**: Results are generated from synthetic payment records using Razorpay Mock Sandbox execution. "
            "Pending high-value transactions are excluded from recovered revenue until merchant approval."
        )
