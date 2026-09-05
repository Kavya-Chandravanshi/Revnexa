# Revnexa

**AI-Powered Revenue Recovery Engine**

> *Recover lost revenue before it slips away.*

**Razorpay AI Buildathon — Track 03: AI Revenue Recovery**

Revnexa is an autonomous, policy-governed revenue recovery engine designed for Razorpay merchants. It intercepts failed payments, applies AI to diagnose failure root causes and customer context, subjects all recovery proposals to a strict deterministic policy engine, requires merchant approval for high-value transactions, executes recovery via Razorpay test-mode APIs, logs an append-only audit trail, and calculates batch recovery financial analytics.

---

## 🏛 System Architecture & Safety Boundaries

```
[Failed Transaction Detected]
            │
            ▼
[AI Recovery Agent (Gemini / Heuristic)]
   ↳ Selects from: {RETRY, PAYMENT_LINK, REMINDER, INCENTIVE, ESCALATE, STOP}
            │
            ▼
[Deterministic Policy Engine] ──(Gatekeeper: Hard Caps, Fraud Lock, Max 2 Retries)
            │
    ┌───────┴───────────────────────┐
    ▼                               ▼
[Safe & Auto-Approved]    [High-Value / Threshold Gate]
    │                               │
    │                    [Merchant Approval Queue]
    │                               │
    └───────────────┬───────────────┘
                    ▼
       [Razorpay Test Mode / Mock]
          │ (Payment Links / Orders)
          ▼
      [Immutable Audit Trail]
          │
          ▼
   [Batch Financial Analytics]
          │
          ▼
   [Streamlit Command Center]
```

> [!IMPORTANT]
> **Financial Safety Guarantee**: The AI Agent never possesses API access to Razorpay and cannot control payment amounts directly. Every action and incentive is strictly validated and calculated by deterministic Python code before reaching Razorpay.

---

## 🛡️ Deterministic Policy Engine Rules

1. **Allowed Actions**: Only `RETRY`, `PAYMENT_LINK`, `REMINDER`, `INCENTIVE`, `ESCALATE`, `STOP` may ever be executed.
2. **Retry Cap**: Maximum 2 automated retries. If `retry_count >= 2`, action `RETRY` is strictly **BLOCKED** (`RETRY_LIMIT_EXCEEDED`).
3. **Fraud Lockout**: If `failure_reason == SUSPECTED_FRAUD`, all automated recovery actions (`RETRY`, `INCENTIVE`, `PAYMENT_LINK`) are **BLOCKED** (`FRAUD_PROTECTION`). Only human `ESCALATE` or `STOP` are permitted.
4. **High-Value Gate**: Transactions exceeding **₹15,000** require merchant sign-off (`REQUIRES_APPROVAL`). Autonomous execution is prevented.
5. **Incentive Caps**: Strictly capped at **10% of amount** AND **maximum ₹500 absolute discount**. The effective discount is computed deterministically in Python.
6. **Duplicate Action Protection**: Prevents re-executing the same recovery action on the same transaction (`DUPLICATE_ACTION`).
7. **State Gate**: Refuses to touch already `RECOVERED` transactions.

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Windows OS (PowerShell)
- Python 3.11 or 3.12 (Installed at `.venv`)

### 2. Activate Virtual Environment
Open PowerShell in the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

Edit `.env` to configure your keys:

```ini
# Razorpay Test Mode Credentials (Optional - Defaults to Mock Sandbox if omitted)
RAZORPAY_KEY_ID=rzp_test_YourKeyIdHere
RAZORPAY_KEY_SECRET=YourSecretKeyHere
RAZORPAY_MODE=auto

# Google Gemini API Key (Optional - Defaults to Heuristic Fallback if omitted)
GEMINI_API_KEY=your_gemini_api_key_here

# Policy Engine Defaults
MAX_RETRY_COUNT=2
MAX_INCENTIVE_PCT=10
MAX_INCENTIVE_AMOUNT=500
HIGH_VALUE_THRESHOLD=15000
```

### 5. Launch the Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💳 Razorpay Test-Mode vs Mock Sandbox

The application includes a resilient multi-mode execution layer:

| Mode | Trigger | Behavior |
| :--- | :--- | :--- |
| **`MOCK SANDBOX`** | Default when credentials are empty or `RAZORPAY_MODE=mock` | Runs 100% offline, generating deterministic simulated Razorpay entities (`mock_plink_...`, `mock_order_...`). Marked with badge `MOCK SANDBOX • SIMULATED`. |
| **`RAZORPAY TEST MODE`** | Set valid `rzp_test_...` credentials and `RAZORPAY_MODE=test` | Connects directly to Razorpay's live sandbox servers to create real test-mode payment links and orders. |

### Gateway Action Mapping

- **`PAYMENT_LINK`**: Calls `client.payment_link.create` to generate a hosted checkout URL.
- **`RETRY`**: Calls `client.order.create` to create a fresh Razorpay order for direct retry.
- **`INCENTIVE`**: Calls `client.payment_link.create` applying the deterministically calculated discount.
- **`REMINDER` / `ESCALATE` / `STOP`**: Non-financial actions. Zero payment API calls made.

---

## 🤖 AI Diagnosis vs Heuristic Fallback

- **Gemini 2.5 Flash**: Employs structured JSON schema output when `GEMINI_API_KEY` is provided.
- **Heuristic Fallback**: An offline rule-based diagnosis engine that triggers automatically if the API key is missing or the network drops. Labeled transparently as `AI: FALLBACK (HEURISTIC)`.

---

## 🎯 5-Minute Hackathon Demo Script (Judging Walkthrough)

1. **Header & Scorecard**:
   - Point out the top header badges: `MOCK SANDBOX • SIMULATED` and `AI: FALLBACK (HEURISTIC)`.
   - Review top KPIs: **₹5.75M Revenue at Risk** across 500 records.
2. **Tab 1 — Overview**:
   - Inspect the **Action Distribution** and **Failure Root Cause** charts.
3. **Tab 2 — Recovery Queue**:
   - Filter by `NETWORK_TIMEOUT` and pick a low-value transaction (`pay_synth_00002`, ₹2,274.76).
   - Observe the horizontal workflow: Ingested → AI Diagnosis (`RETRY`) → Policy Gate (`ALLOWED`) → Razorpay.
   - Click **`⚡ Execute Recovery`** and show the generated Razorpay ID and short link.
4. **Tab 3 — AI Workbench**:
   - Walk through the 4 curated scenarios:
     1. Standard Recovery (Auto-allowed).
     2. High-Value B2B Gate (> ₹15,000 threshold requires approval).
     3. Suspected Fraud (Terminal halt).
     4. Exhausted Retries (Stop candidate).
   - Demonstrate the **Deliberate Safety Failure Playground**: attempt an unauthorized 25% discount or Swift transfer and watch the Policy Engine intercept and block the action.
5. **Tab 4 — Merchant Approval Queue**:
   - View pending high-value orders (`> ₹15,000`).
   - Click **`✅ Approve & Dispatch Link`** on a pending item. Notice how it executes immediately and transitions the workflow.
6. **Tab 5 — Audit Trail**:
   - Filter by the approved Payment ID.
   - Show the complete, tamper-evident timeline from `PAYMENT_DETECTED` to `APPROVAL_GRANTED` and `RECOVERY_SUCCEEDED`.
7. **Tab 6 — Batch Simulation**:
   - Click **`▶️ RUN 500-RECORD RECOVERY SIMULATION`**.
   - Watch the live progress bar process the portfolio.
   - Review the final financial impact: **~62.4% recovery rate**, **₹1.82M recovered revenue**, and **2,038x ROI**.

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

Revnexa is built as a native **Streamlit** dashboard application. The recommended platform for zero-config cloud deployment is **Streamlit Community Cloud**.

### Steps to Deploy:
1. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
2. Click **New app** and connect your GitHub account.
3. Configure the repository settings:
   - **Repository**: `Kavya-Chandravanshi/Revnexa`
   - **Branch**: `master`
   - **Main file path**: `app.py`
4. Expand **Advanced settings...** $\rightarrow$ **Secrets** and configure your environment variables:
   ```toml
   RAZORPAY_KEY_ID = "rzp_test_your_key_id"
   RAZORPAY_KEY_SECRET = "your_razorpay_secret"
   GEMINI_API_KEY = "your_gemini_api_key"
   ```
5. Click **Deploy!**

---

## ⚠️ Simulation Disclaimer

All customer profiles and transaction records in `data/synthetic_payments.json` are synthetic and generated for demonstration purposes. In Mock Sandbox mode, all recovered revenue metrics are simulated.

