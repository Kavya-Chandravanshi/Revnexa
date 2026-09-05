"""Revnexa - Interactive Autonomous Revenue Recovery Engine Command Center.

Track 03: AI Revenue Recovery (Razorpay AI Buildathon)
Autonomous, policy-governed revenue recovery engine for Razorpay merchants.
Designed as an intelligent, dark-first AI Operations Center.
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
# Streamlit Page Configuration & Modern Dark-First Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Revnexa | AI Revenue Operations Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def format_inr(val: float, compact: bool = True) -> str:
    """Format monetary values into Indian currency notations (Lakhs, Crores, or Thousands)."""
    if val is None:
        return "₹0.00"
    abs_val = abs(val)
    if compact:
        if abs_val >= 10000000:
            return f"₹{val / 10000000:.2f} Cr"
        elif abs_val >= 100000:
            return f"₹{val / 100000:.2f}L"
        elif abs_val >= 1000:
            return f"₹{val / 1000:.1f}k"
    return f"₹{val:,.2f}"


# Custom CSS Injection for Futuristic Dark-First AI Operations Center Theme
st.markdown(
    """
    <style>
    /* Dark Surface & Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1527 0%, #080c14 100%);
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c121e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    /* Top Command Header */
    .top-header-dark {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px);
        padding: 1.25rem 2rem;
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.15);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px -5px rgba(56, 189, 248, 0.08);
        margin-bottom: 1.5rem;
    }

    .top-brand {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .brand-logo-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        font-size: 1.5rem;
    }

    .top-title-dark {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }

    .top-subtitle-dark {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0.15rem;
    }

    /* Badges */
    .badge-capsule {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .badge-cyan-glow {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }

    .badge-amber-glow {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .badge-emerald-glow {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .badge-rose-glow {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.3);
    }

    .badge-slate-pill {
        background: rgba(148, 163, 184, 0.1);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Metric Cards */
    .kpi-card-dark {
        background: linear-gradient(160deg, rgba(17, 24, 39, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card-dark:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }

    .kpi-card-dark::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38bdf8, transparent);
    }

    .kpi-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .kpi-num {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0.3rem 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-sub {
        font-size: 0.73rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Attention Box */
    .attention-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Connected Recovery Map */
    .recovery-map {
        display: flex;
        align-items: stretch;
        background: rgba(13, 21, 39, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 1.25rem 1rem;
        margin: 1.25rem 0 1.5rem 0;
        gap: 0.6rem;
        overflow-x: auto;
    }

    .map-step {
        flex: 1;
        min-width: 130px;
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.9rem 0.8rem;
        text-align: center;
        position: relative;
    }

    .map-step.active {
        border-color: rgba(56, 189, 248, 0.4);
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.08) 0%, rgba(15, 23, 42, 0.8) 100%);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);
    }

    .map-step-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .map-step-val {
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .map-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: rgba(56, 189, 248, 0.4);
        font-size: 1.2rem;
        font-weight: bold;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 0.3rem 0.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0 1.2rem;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }

    /* Panel Card */
    .panel-card {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
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
        dataset_path = Path(__file__).resolve().parent / "data" / "synthetic_payments.json"
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
    st.session_state.outcomes = {}

if "batch_metrics" not in st.session_state:
    st.session_state.batch_metrics = None


# ---------------------------------------------------------------------------
# Precompute initial diagnoses fast via Heuristic for instant responsive queue
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
# TOP COMMAND HEADER
# ---------------------------------------------------------------------------
client: RazorpayRecoveryClient = st.session_state.rzp_client
agent: AIRecoveryAgent = st.session_state.ai_agent

rzp_badge_class = "badge-amber-glow" if client.is_mock else "badge-cyan-glow"
rzp_badge_text = "MOCK SANDBOX • SIMULATED" if client.is_mock else "RAZORPAY TEST MODE"

active_ai_mode = st.session_state.get("active_ai_mode", "gemini" if agent._client is not None else "fallback")
ai_badge_class = "badge-cyan-glow" if active_ai_mode == "gemini" else "badge-slate-pill"
ai_badge_text = "AI: GEMINI" if active_ai_mode == "gemini" else "AI: FALLBACK"

st.markdown(
    f"""
    <div class="top-header-dark">
        <div class="top-brand">
            <div class="brand-logo-icon">⚡</div>
            <div>
                <h1 class="top-title-dark">REVNEXA</h1>
                <div class="top-subtitle-dark">Autonomous Revenue Recovery Engine • Razorpay Buildathon</div>
            </div>
        </div>
        <div style="display: flex; gap: 0.6rem; align-items: center;">
            <span class="badge-capsule {rzp_badge_class}">🛡️ {rzp_badge_text}</span>
            <span class="badge-capsule {ai_badge_class}">🧠 {ai_badge_text}</span>
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
        <div class="kpi-card-dark">
            <div class="kpi-title">Revenue At Risk</div>
            <div class="kpi-num">{format_inr(live_metrics.total_revenue_at_risk)}</div>
            <div class="kpi-sub" title="Exact: ₹{live_metrics.total_revenue_at_risk:,.2f}">{total_records} Orders (Failed: {format_inr(live_metrics.total_failed_volume)} • At-Risk: {format_inr(live_metrics.total_at_risk_volume)})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    links_created = getattr(live_metrics, "payment_links_created", 0)
    if client.is_mock:
        rev_label = "Simulated Recovered"
        caption_text = f"{live_metrics.successful_recoveries} Recoveries • {links_created} Links"
    else:
        rev_label = "Confirmed Recovered"
        caption_text = f"{live_metrics.successful_recoveries} Confirmed Paid • {links_created} Links"

    st.markdown(
        f"""
        <div class="kpi-card-dark" style="border-top: 2px solid #10b981;">
            <div class="kpi-title" style="color: #34d399;">{rev_label}</div>
            <div class="kpi-num" style="color: #10b981;">{format_inr(live_metrics.total_recovered_revenue)}</div>
            <div class="kpi-sub" title="Exact: ₹{live_metrics.total_recovered_revenue:,.2f}">{caption_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card-dark" style="border-top: 2px solid #38bdf8;">
            <div class="kpi-title" style="color: #38bdf8;">Recovery Rate</div>
            <div class="kpi-num" style="color: #38bdf8;">{live_metrics.recovery_rate:.1f}%</div>
            <div class="kpi-sub">Portfolio efficiency</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    pending_count = len(st.session_state.queue.list_pending_requests())
    st.markdown(
        f"""
        <div class="kpi-card-dark" style="border-top: 2px solid #f59e0b;">
            <div class="kpi-title" style="color: #fbbf24;">Pending Approvals</div>
            <div class="kpi-num" style="color: #f59e0b;">{pending_count}</div>
            <div class="kpi-sub">Transactions &gt; ₹15,000</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    halt_count = live_metrics.blocked_actions + live_metrics.escalations
    st.markdown(
        f"""
        <div class="kpi-card-dark" style="border-top: 2px solid #f43f5e;">
            <div class="kpi-title" style="color: #fb7185;">Terminal Stops</div>
            <div class="kpi-num" style="color: #f43f5e;">{halt_count}</div>
            <div class="kpi-sub">Fraud & safety guardrails</div>
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
    "⚡ Recovery Queue",
    "🧠 AI Decision Studio",
    "🛡️ Approval Center",
    "📜 Audit Trail",
    "🚀 Autonomous Simulation",
])


# ===========================================================================
# TAB 1: OVERVIEW & COCKPIT
# ===========================================================================
with tab_overview:
    # Action Center / What Needs Attention
    pending_high_val = len(st.session_state.queue.list_pending_requests())
    outcomes_dict = st.session_state.outcomes
    failed_executions = sum(1 for o in outcomes_dict.values() if not o.get("execution", RazorpayExecutionResult(success=True)).success)
    links_generated = live_metrics.payment_links_created

    st.markdown(
        f"""
        <div class="attention-box">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 1.6rem;">⚡</div>
                <div>
                    <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">Autonomous Engine Status: Active Telemetry</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        • <b>{pending_high_val} high-value orders</b> held at policy gate &nbsp;|&nbsp; 
                        • <b>{links_generated} recovery links</b> active &nbsp;|&nbsp; 
                        • <b>{failed_executions} failed executions</b> requiring attention
                    </div>
                </div>
            </div>
            <span class="badge-capsule badge-cyan-glow">REAL-TIME TELEMETRY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Signature Visual: The Connected Revenue Recovery Pipeline
    st.markdown(
        """
        <div style="margin-bottom: 0.5rem; font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
            Autonomous Recovery Pipeline Architecture
        </div>
        <div class="recovery-map">
            <div class="map-step active">
                <div class="map-step-label">1. Gateway Ingestion</div>
                <div class="map-step-val" style="color: #f8fafc;">500 Records</div>
                <div style="font-size: 0.65rem; color: #38bdf8;">Webhooks / APIs</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">2. AI Diagnosis</div>
                <div class="map-step-val" style="color: #38bdf8;">Gemini 2.5</div>
                <div style="font-size: 0.65rem; color: #94a3b8;">Heuristic Fallback</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">3. Policy Gate</div>
                <div class="map-step-val" style="color: #fbbf24;">100% Guarded</div>
                <div style="font-size: 0.65rem; color: #94a3b8;">Zero Overrides</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">4. Human Gate</div>
                <div class="map-step-val" style="color: #f59e0b;">&gt; ₹15,000</div>
                <div style="font-size: 0.65rem; color: #94a3b8;">Merchant Sign-off</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">5. Razorpay Engine</div>
                <div class="map-step-val" style="color: #34d399;">Execution</div>
                <div style="font-size: 0.65rem; color: #94a3b8;">Links / Retries</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">6. Immutable Audit</div>
                <div class="map-step-val" style="color: #10b981;">Provenance</div>
                <div style="font-size: 0.65rem; color: #94a3b8;">Full Ledger</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                "PAYMENT_LINK": "#38bdf8",
                "RETRY": "#10b981",
                "INCENTIVE": "#6366f1",
                "ESCALATE": "#f59e0b",
                "STOP": "#f43f5e",
                "REMINDER": "#06b6d4",
            },
            title="Autonomous Recovery Strategy Allocations",
            template="plotly_dark",
        )
        fig_action.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
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
            hole=0.55,
            title="Gateway Payment Failure Root Causes",
            color_discrete_sequence=["#38bdf8", "#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#06b6d4"],
            template="plotly_dark",
        )
        fig_reason.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_reason, use_container_width=True)

    # Financial Impact Breakdown
    st.markdown("### 💰 Financial Exposure & Capital Recovery Ledger")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.markdown(
            f"""
            <div class="panel-card" style="border-left: 3px solid #38bdf8;">
                <div class="kpi-title">Gross Failure Exposure</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin: 0.3rem 0;">{format_inr(live_metrics.total_failed_volume, compact=False)}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Average Order Value: {format_inr(live_metrics.total_failed_volume/total_records)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sum_col2:
        st.markdown(
            f"""
            <div class="panel-card" style="border-left: 3px solid #10b981;">
                <div class="kpi-title">Net Capital Recovered</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #10b981; margin: 0.3rem 0;">{format_inr(live_metrics.net_recovered_revenue, compact=False)}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Incentives Disbursed: {format_inr(live_metrics.incentive_amount_given)} (10% Cap Enforced)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sum_col3:
        st.markdown(
            f"""
            <div class="panel-card" style="border-left: 3px solid #f59e0b;">
                <div class="kpi-title">Security &amp; Policy Holds</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #fbbf24; margin: 0.3rem 0;">{live_metrics.approvals_requested} High-Value Held</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">{sum(1 for r in records if precomputed[r.payment_id]['action'] == 'STOP')} Terminal Security Stops Active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# TAB 2: RECOVERY QUEUE & TRANSACTION WORKSPACE
# ===========================================================================
with tab_queue:
    st.markdown("### ⚡ Live Failure Ingestion & Transaction Queue")
    st.caption("Inspect inbound payment events, review AI diagnoses, and execute automated recovery operations.")

    # Search & Filter Controls
    filt_c1, filt_c2, filt_c3, filt_c4 = st.columns(4)
    with filt_c1:
        reason_filter = st.selectbox("Failure Reason", ["All"] + [r.value for r in FailureReason])
    with filt_c2:
        action_filter = st.selectbox("AI Strategy", ["All"] + [a.value for a in RecoveryAction])
    with filt_c3:
        policy_filter = st.selectbox("Policy Gate", ["All", "ALLOWED", "REQUIRES_APPROVAL", "BLOCKED"])
    with filt_c4:
        amount_filter = st.selectbox("Order Value", ["All", "Under ₹5,000", "₹5,000 - ₹15,000", "High Value (> ₹15,000)"])

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
        if amount_filter == "Under ₹5,000" and r.amount >= 5000:
            continue
        if amount_filter == "₹5,000 - ₹15,000" and (r.amount < 5000 or r.amount > 15000):
            continue
        if amount_filter == "High Value (> ₹15,000)" and r.amount <= 15000:
            continue

        table_data.append({
            "Payment ID": r.payment_id,
            "Customer": r.customer_name,
            "Amount": format_inr(r.amount, compact=False),
            "Raw Amount": r.amount,
            "Failure Reason": r.failure_reason.value,
            "Retries": f"{r.retry_count} / 2",
            "Customer LTV": f"{r.customer_previous_payments} txns ({format_inr(r.customer_total_spend)})",
            "AI Action": act,
            "Confidence": f"{diag['confidence']*100:.0f}%",
            "Policy Gate": pol,
            "Current Status": current_status,
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
        st.warning("No records match the selected filter criteria.")

    st.markdown("---")

    # Transaction Detail Selection
    st.markdown("### 🔍 Interactive Transaction Recovery Workspace")
    selected_pid = st.selectbox(
        "Select an Ingested Payment to Inspect & Recover:",
        options=[r.payment_id for r in records],
        index=0,
    )

    selected_payment = next(r for r in records if r.payment_id == selected_pid)

    # 1. Connected Horizontal Workflow
    has_outcome = selected_pid in st.session_state.outcomes
    current_outcome = st.session_state.outcomes.get(selected_pid, {})
    exec_res = current_outcome.get("execution")

    ai_rec_curr = agent.recommend(selected_payment)
    st.session_state["active_ai_mode"] = ai_rec_curr.agent_mode
    policy_curr = evaluate_policy(selected_payment, ai_rec_curr.action, proposed_discount_pct=ai_rec_curr.proposed_discount_pct)

    appr_status_text = "BYPASSED" if not policy_curr.requires_approval else ("APPROVED" if st.session_state.queue.is_approved(selected_pid) else "PENDING SIGN-OFF")
    if exec_res:
        if exec_res.success:
            rzp_status_text = "DISPATCHED"
        elif exec_res.error_code == "RATE_LIMITED":
            rzp_status_text = "RATE LIMITED (429)"
        else:
            rzp_status_text = "FAILED"
    elif policy_curr.decision == PolicyDecisionType.BLOCKED:
        rzp_status_text = "BLOCKED"
    elif policy_curr.requires_approval:
        rzp_status_text = "HELD"
    else:
        rzp_status_text = "READY"

    pol_color = "#34d399" if policy_curr.decision == PolicyDecisionType.ALLOWED else ("#fbbf24" if policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL else "#fb7185")
    appr_color = "#34d399" if appr_status_text != "PENDING SIGN-OFF" else "#fbbf24"

    st.markdown(
        f"""
        <div class="recovery-map">
            <div class="map-step active">
                <div class="map-step-label">1. Ingested</div>
                <div class="map-step-val" style="color: #38bdf8;">✓ RECEIVED</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">2. AI Diagnosis</div>
                <div class="map-step-val" style="color: #38bdf8;">{ai_rec_curr.action.value}</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">3. Policy Gate</div>
                <div class="map-step-val" style="color: {pol_color};">{policy_curr.decision.value}</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">4. Approval</div>
                <div class="map-step-val" style="color: {appr_color};">{appr_status_text}</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">5. Razorpay</div>
                <div class="map-step-val" style="color: #f8fafc;">{rzp_status_text}</div>
            </div>
            <div class="map-arrow">➔</div>
            <div class="map-step active">
                <div class="map-step-label">6. Audit Trail</div>
                <div class="map-step-val" style="color: #10b981;">LOGGED</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Detail 3-Column Workspace
    det_c1, det_c2, det_c3 = st.columns([1.2, 1.4, 1.4])

    with det_c1:
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="kpi-title" style="color: #38bdf8; margin-bottom: 0.5rem;">Transaction &amp; Customer Profile</div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #cbd5e1;">
                    • <b>Customer</b>: {selected_payment.customer_name} (<span class="font-mono">{selected_payment.customer_id}</span>)<br/>
                    • <b>Order Amount</b>: <span style="color: #38bdf8; font-weight: 700;">{format_inr(selected_payment.amount, compact=False)}</span><br/>
                    • <b>Order ID</b>: <span class="font-mono">{selected_payment.order_id}</span><br/>
                    • <b>Category</b>: {selected_payment.product_category.value}<br/>
                    • <b>Failure Code</b>: <span class="font-mono">{selected_payment.failure_reason.value}</span><br/>
                    • <b>Error Details</b>: {selected_payment.error_description}<br/>
                    • <b>Prior Retries</b>: {selected_payment.retry_count} / 2<br/>
                    • <b>Past Spend</b>: {format_inr(selected_payment.customer_total_spend, compact=False)} ({selected_payment.customer_previous_payments} txns)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with det_c2:
        st.markdown(
            f"""
            <div class="panel-card" style="border: 1px solid rgba(56, 189, 248, 0.3);">
                <div class="kpi-title" style="color: #38bdf8; display: flex; justify-content: space-between;">
                    <span>🧠 AI Proposal (Advisory)</span>
                    <span>{ai_rec_curr.confidence*100:.0f}% Confidence</span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #38bdf8; margin: 0.4rem 0;">
                    {ai_rec_curr.action.value}
                </div>
                <p style="font-size: 0.82rem; color: #cbd5e1; margin: 0.4rem 0; line-height: 1.5;">{ai_rec_curr.reason}</p>
                <hr style="margin: 0.6rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);"/>
                <div style="font-size: 0.75rem; color: #94a3b8;">
                    <b>Risk Level</b>: <span class="badge-capsule badge-slate-pill">{ai_rec_curr.risk_level}</span> &nbsp;|&nbsp; 
                    <b>Engine</b>: {"Gemini 2.5 Flash" if ai_rec_curr.agent_mode == "gemini" else "Deterministic Fallback"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with det_c3:
        pol_box_border = "rgba(52, 211, 153, 0.3)" if policy_curr.decision == PolicyDecisionType.ALLOWED else ("rgba(251, 191, 36, 0.3)" if policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL else "rgba(251, 113, 133, 0.3)")
        pol_box_title = "#34d399" if policy_curr.decision == PolicyDecisionType.ALLOWED else ("#fbbf24" if policy_curr.decision == PolicyDecisionType.REQUIRES_APPROVAL else "#fb7185")

        st.markdown(
            f"""
            <div class="panel-card" style="border: 1px solid {pol_box_border};">
                <div class="kpi-title" style="color: {pol_box_title};">
                    ⚖️ Deterministic Policy Gate (Authoritative)
                </div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {pol_box_title}; margin: 0.4rem 0;">
                    {policy_curr.decision.value}
                </div>
                <p style="font-size: 0.82rem; color: #cbd5e1; margin: 0.4rem 0; line-height: 1.5;">{policy_curr.reason}</p>
                <hr style="margin: 0.6rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);"/>
                <div style="font-size: 0.75rem; color: #94a3b8;">
                    <b>Policy Codes</b>: <span class="font-mono">{", ".join(policy_curr.policy_codes)}</span><br/>
                    <b>Net Payable</b>: {format_inr(policy_curr.final_payable_amount, compact=False)} (Discount: {format_inr(policy_curr.effective_discount, compact=False)})
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
                if st.button("🛡️ Route to Merchant Approval Center", key=f"queue_{selected_pid}"):
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
                        message=f"Approval request queued for high-value transaction {format_inr(selected_payment.amount, compact=False)}.",
                    )
                    st.success("Successfully queued for review in Tab 4 (Approval Center)!")
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
            if exec_res.success:
                st.markdown("**Gateway Execution Result:**")
                st.success(f"✓ Recovery Action Executed: {exec_res.action.value}")
                st.json({
                    "Recovery Attempt ID": exec_res.recovery_id,
                    "Razorpay Link/Order ID": exec_res.razorpay_id,
                    "Execution Status": "Recovery Action Executed (Link Dispatched)" if not exec_res.simulated else "Simulated Action Executed",
                    "Amount": format_inr(exec_res.amount, compact=False),
                    "Short URL": exec_res.short_url,
                    "Simulated": exec_res.simulated,
                })
            else:
                st.markdown("**Gateway Execution Result:**")
                st.error(f"❌ Execution Error: {exec_res.message}")
                if exec_res.error_code == "RATE_LIMITED":
                    st.warning("⚠️ **Razorpay Rate Limited (HTTP 429)**: Too many requests. Safe backoff required. This action was NOT executed and is NOT counted as recovered revenue.")
                st.json({
                    "Recovery Attempt ID": exec_res.recovery_id,
                    "Action Status": "NOT EXECUTED",
                    "Error Code": exec_res.error_code,
                    "Message": exec_res.message,
                    "Simulated": exec_res.simulated,
                })


# ===========================================================================
# TAB 3: AI DECISION STUDIO & SAFETY LAB
# ===========================================================================
with tab_workbench:
    st.markdown("### 🧠 AI Decision Studio & Adversarial Safety Lab")
    st.caption("Inspect how the AI reasoning engine evaluates context, and verify that the deterministic policy layer blocks adversarial attacks.")

    demo_scenario = st.radio(
        "Select an automated demonstration scenario:",
        [
            "1. Standard Autonomous Recovery (Transient Timeout)",
            "2. High-Value B2B Order Gate (> ₹15,000 Approval Required)",
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
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="kpi-title" style="color: #38bdf8;">1. Raw Ingested Payment</div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #cbd5e1; margin-top: 0.5rem;">
                    • <b>Payment ID</b>: <span class="font-mono">{demo_record.payment_id}</span><br/>
                    • <b>Amount</b>: <span style="color: #38bdf8; font-weight: 700;">{format_inr(demo_record.amount, compact=False)}</span><br/>
                    • <b>Reason</b>: <span class="font-mono">{demo_record.failure_reason.value}</span><br/>
                    • <b>Retries</b>: {demo_record.retry_count} / 2<br/>
                    • <b>Customer</b>: {demo_record.customer_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    demo_ai = agent.recommend(demo_record)
    demo_policy = evaluate_policy(demo_record, demo_ai.action, proposed_discount_pct=demo_ai.proposed_discount_pct)

    with col_w2:
        st.markdown(
            f"""
            <div class="panel-card" style="border: 1px solid rgba(56, 189, 248, 0.3);">
                <div class="kpi-title" style="color: #38bdf8;">2. AI Recommendation</div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #cbd5e1; margin-top: 0.5rem;">
                    • <b>Action</b>: <span style="color: #38bdf8; font-weight: 700;">{demo_ai.action.value}</span><br/>
                    • <b>Confidence</b>: {demo_ai.confidence*100:.0f}%<br/>
                    • <b>Risk Level</b>: {demo_ai.risk_level}<br/>
                    • <b>Rationale</b>: {demo_ai.reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_w3:
        pol_code_str = ", ".join(demo_policy.policy_codes) if demo_policy.policy_codes else "NONE"
        st.markdown(
            f"""
            <div class="panel-card" style="border: 1px solid {pol_box_border};">
                <div class="kpi-title" style="color: #34d399;">3. Policy Engine Verdict</div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #cbd5e1; margin-top: 0.5rem;">
                    • <b>Decision</b>: <span style="font-weight: 700;">{demo_policy.decision.value}</span><br/>
                    • <b>Requires Approval</b>: {demo_policy.requires_approval}<br/>
                    • <b>Policy Codes</b>: <span class="font-mono">{pol_code_str}</span><br/>
                    • <b>Explanation</b>: {demo_policy.reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if demo_policy.requires_approval:
        st.warning(f"⚠️ **Approval Rationale**: {demo_policy.reason}")

    st.markdown("---")

    # Deliberate Safety Failure Playground Button (Attack vs Defense)
    st.markdown("### 🚨 Adversarial Safety Playground (Attack vs Defense)")
    st.caption("Demonstrate that an unauthorized prompt injection or excessive incentive discount cannot bypass the deterministic safety layer.")

    fail_col1, fail_col2 = st.columns(2)
    with fail_col1:
        sim_action = st.selectbox(
            "Simulate AI Action Proposal (Attacker Action):",
            [RecoveryAction.INCENTIVE.value, RecoveryAction.RETRY.value, "UNAUTHORIZED_SWIFT_TRANSFER"],
            index=0,
        )
        sim_discount = st.slider("Simulate Proposed Discount Percentage (Attacker Discount):", 0, 50, 25)
    with fail_col2:
        test_viol = evaluate_policy(demo_record, sim_action, proposed_discount_pct=float(sim_discount))
        st.markdown("**Deterministic Policy Engine Defense:**")
        if test_viol.decision == PolicyDecisionType.BLOCKED:
            st.error(f"❌ VIOLATION INTERCEPTED & BLOCKED:\n\n{test_viol.reason}")
            st.caption(f"**Policy Code**: `{', '.join(test_viol.policy_codes)}` • Deterministic policy prevents excessive incentives.")
        elif test_viol.decision == PolicyDecisionType.REQUIRES_APPROVAL:
            st.warning(f"⚠️ APPROVAL FORCED BY SAFETY POLICY:\n\n{test_viol.reason}")
        else:
            st.success(f"✅ ALLOWED:\n\n{test_viol.reason}")


# ===========================================================================
# TAB 4: APPROVAL CENTER (HUMAN-IN-THE-LOOP)
# ===========================================================================
with tab_approval:
    st.markdown("### 🛡️ Merchant Sign-off & Exposure Review Inbox")
    st.caption("High-value orders (> ₹15,000) or complex risk profiles strictly require human authorization before gateway execution.")

    queue_instance: MerchantApprovalQueue = st.session_state.queue
    pending_reqs = queue_instance.list_pending_requests()

    # Calculate held exposure
    total_held_val = sum(req.amount for req in pending_reqs)

    appr_stat1, appr_stat2 = st.columns(2)
    with appr_stat1:
        st.markdown(
            f"""
            <div class="panel-card" style="border-left: 3px solid #f59e0b;">
                <div class="kpi-title">Pending Authorizations</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #fbbf24;">{len(pending_reqs)} High-Value Orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with appr_stat2:
        st.markdown(
            f"""
            <div class="panel-card" style="border-left: 3px solid #38bdf8;">
                <div class="kpi-title">Total Capital Held Under Gate</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">{format_inr(total_held_val, compact=False)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not pending_reqs:
        st.success("🎉 Inbox Zero: All high-value transactions have been reviewed and resolved!")
    else:
        st.warning(f"Currently **{len(pending_reqs)} high-value orders** are awaiting merchant sign-off.")

        for req in pending_reqs:
            with st.container():
                st.markdown(
                    f"""
                    <div class="panel-card" style="border-left: 3px solid #f59e0b; margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">Request ID: <span class="font-mono">{req.approval_id}</span></span>
                            <span class="badge-capsule badge-amber-glow">PENDING SIGN-OFF</span>
                        </div>
                        <div style="margin: 0.5rem 0; color: #cbd5e1; font-size: 0.88rem;">
                            <b>Payment ID</b>: <span class="font-mono">{req.payment_id}</span> &nbsp;|&nbsp; 
                            <b>Amount</b>: <span style="font-size: 1.15rem; font-weight: 800; color: #38bdf8;">{format_inr(req.amount, compact=False)}</span> &nbsp;|&nbsp; 
                            <b>Customer</b>: <span class="font-mono">{req.customer_id}</span>
                        </div>
                        <div style="margin: 0.3rem 0; font-size: 0.84rem; color: #94a3b8;">
                            <b>AI Proposal</b>: Recommends <b>{req.requested_action.value}</b> ({req.ai_confidence*100:.0f}% confidence)<br/>
                            <i>"{req.ai_reason}"</i>
                        </div>
                        <div style="margin: 0.3rem 0; font-size: 0.84rem; color: #fbbf24;">
                            <b>Policy Intercept</b>: {req.policy_reason}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                appr_col1, appr_col2, appr_col3 = st.columns([1.5, 1.5, 3])
                with appr_col1:
                    if st.button(f"✅ Authorize & Dispatch", key=f"appr_{req.approval_id}"):
                        queue_instance.approve_request(req.approval_id, reviewer="Merchant Admin", notes="Approved via Dashboard")
                        
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
                    if st.button(f"❌ Reject Execution", key=f"rej_{req.approval_id}"):
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
    st.markdown("### 📜 Append-Only Chronological Audit Trail")
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
# TAB 6: AUTONOMOUS BATCH SIMULATION
# ===========================================================================
with tab_batch:
    st.markdown("### 🚀 500-Record Autonomous Recovery Batch Simulation")
    st.caption("Run the complete governed recovery pipeline across all 500 synthetic payment failures to measure portfolio financial recovery and net ROI.")

    st.markdown(
        """
        <div class="panel-card" style="border: 1px dashed rgba(56, 189, 248, 0.3); margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <b>Simulation Mode</b>: Safe auto-approved actions (&lt; ₹15,000, 0 retries, non-fraud) are executed through the 
                    <b>Razorpay Mock Sandbox</b>. High-value transactions (&gt; ₹15,000) are routed to the <b>Merchant Approval Center</b> and 
                    never count as recovered until reviewed.
                </div>
                <span class="badge-capsule badge-amber-glow">SIMULATED • MOCK SANDBOX</span>
            </div>
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

        # Batch simulation strictly runs against Mock Sandbox to model confirmed recoveries without hitting external APIs
        sim_client = RazorpayRecoveryClient(mode="mock")
        metrics_result, batch_outcomes = run_batch_recovery(
            records=records,
            approval_queue=st.session_state.queue,
            audit_logger=st.session_state.audit,
            rzp_client=sim_client,
            force_fallback=True,
        )

        for out in batch_outcomes:
            st.session_state.outcomes[out["payment_id"]] = out

        st.session_state.batch_metrics = metrics_result
        progress_bar.progress(100)
        status_msg.success("✅ 500 Records Processed Successfully in Mock Sandbox!")
        st.rerun()

    # Display Batch Simulation Scorecard
    if st.session_state.batch_metrics:
        bm: BatchMetrics = st.session_state.batch_metrics
        st.markdown("### 📊 Measured Batch Recovery KPIs")

        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.markdown(
                f"""
                <div class="kpi-card-dark">
                    <div class="kpi-title">Records Ingested</div>
                    <div class="kpi-num">{bm.total_records:,}</div>
                    <div class="kpi-sub">Total Volume: {format_inr(bm.total_revenue_at_risk)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_col2:
            st.markdown(
                f"""
                <div class="kpi-card-dark" style="border-top: 2px solid #10b981;">
                    <div class="kpi-title" style="color: #34d399;">Simulated Recovered</div>
                    <div class="kpi-num" style="color: #10b981;">{format_inr(bm.total_recovered_revenue)}</div>
                    <div class="kpi-sub">{bm.successful_recoveries:,} Orders ({bm.recovery_rate:.1f}% Rate)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_col3:
            st.markdown(
                f"""
                <div class="kpi-card-dark" style="border-top: 2px solid #38bdf8;">
                    <div class="kpi-title" style="color: #38bdf8;">Net Capital Recovered</div>
                    <div class="kpi-num" style="color: #38bdf8;">{format_inr(bm.net_recovered_revenue)}</div>
                    <div class="kpi-sub">Held at Policy Gate: {bm.approvals_requested:,} Orders</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_col4:
            st.markdown(
                f"""
                <div class="kpi-card-dark" style="border-top: 2px solid #f59e0b;">
                    <div class="kpi-title" style="color: #fbbf24;">Financial ROI Multiple</div>
                    <div class="kpi-num" style="color: #f59e0b;">{bm.roi:,.1f}x</div>
                    <div class="kpi-sub">Terminal Halts: {bm.blocked_actions + bm.escalations:,} Orders</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "⚠️ **SIMULATED RESULTS**: Results are generated from synthetic payment records using Razorpay Mock Sandbox execution. "
            "Pending high-value transactions are excluded from recovered revenue until merchant approval."
        )
