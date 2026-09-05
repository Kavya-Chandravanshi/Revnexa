"""AI Recovery Agent for Revnexa.

Analyzes transaction context and customer payment history to recommend
the optimal revenue recovery action using Google Gemini structured output,
with a transparent, deterministic heuristic fallback when Gemini is unavailable.
"""

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    RecoveryAction,
    AIRecommendation,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a revenue recovery decision assistant for a payment platform.

Your task is to recommend the safest and most economically reasonable recovery action for a failed or at-risk payment.

You are advisory only.
You do not have authority to execute financial actions.
You must select only from the allowed actions:
- RETRY
- PAYMENT_LINK
- REMINDER
- INCENTIVE
- ESCALATE
- STOP

You must prefer STOP or ESCALATE when evidence is insufficient or automated recovery is unsafe.
You must never invent transaction facts.
You must use only the supplied payment/customer context.
You must provide concise business reasoning.

General Principles:
- Temporary/network/bank availability failures (BANK_SERVER_DOWN, NETWORK_TIMEOUT) may favor RETRY or PAYMENT_LINK when retry history is low.
- Repeated failures (retry_count >= 2) should push toward PAYMENT_LINK, ESCALATE, or STOP rather than unlimited retries.
- Insufficient funds (INSUFFICIENT_FUNDS) may favor PAYMENT_LINK, REMINDER, or STOP.
- Expired/invalid payment details (CARD_EXPIRED, INVALID_CARD_DETAILS) should generally favor PAYMENT_LINK or REMINDER rather than RETRY.
- Suspected fraud (SUSPECTED_FRAUD) must result in ESCALATE or STOP. Never recommend incentives or retries on suspected fraud.
- High-value transactions (amount > INR 15,000) should recommend conservative actions and note that approval may be necessary.
- Incentives should only be suggested when there is a plausible retention/recovery reason (e.g. high-spend customer drop-off on modest order), not automatically.
- Do not recommend incentives on suspected fraud.
- Do not recommend endless retries.
"""


def heuristic_fallback_recommend(payment: PaymentRecord) -> AIRecommendation:
    """Deterministic, transparent heuristic fallback when Gemini is unavailable or unconfigured.
    
    Produces an AIRecommendation-shaped object with explainable rationale
    grounded strictly in the supplied payment record facts.
    """
    amount = payment.amount
    reason = payment.failure_reason
    retries = payment.retry_count
    prev_success = payment.customer_previous_payments
    total_spend = payment.customer_total_spend
    is_high_value = amount > 15000.0

    # 1. Fraud Protection: Terminal stop or escalation
    if reason == FailureReason.SUSPECTED_FRAUD:
        return AIRecommendation(
            action=RecoveryAction.ESCALATE,
            confidence=0.98,
            reason=(
                f"Transaction flagged as SUSPECTED_FRAUD on INR {amount:,.2f} order. "
                "Automated recovery is unsafe and strictly prohibited; escalates immediately to risk analyst."
            ),
            expected_recovery=0.0,
            risk_level="CRITICAL",
            proposed_discount_pct=0.0,
            requires_approval_recommendation=True,
            agent_mode="fallback",
            model_name=None,
        )

    # 2. Retries exhausted (>= 2)
    if retries >= 2:
        if is_high_value or total_spend >= 15000.0:
            return AIRecommendation(
                action=RecoveryAction.ESCALATE,
                confidence=0.88,
                reason=(
                    f"Customer experienced {retries} consecutive failures on a high-value order (INR {amount:,.2f}). "
                    "Automated retries exhausted; escalating to dedicated account manager for direct concierge outreach."
                ),
                expected_recovery=amount,
                risk_level="HIGH",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=True,
                agent_mode="fallback",
                model_name=None,
            )
        else:
            return AIRecommendation(
                action=RecoveryAction.STOP,
                confidence=0.86,
                reason=(
                    f"Transaction has exhausted automated attempts ({retries} prior retries). "
                    "Halting further automated outreach to protect merchant sender reputation and prevent customer fatigue."
                ),
                expected_recovery=0.0,
                risk_level="HIGH",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=False,
                agent_mode="fallback",
                model_name=None,
            )

    # 3. Expired or Invalid Card details: Payment Link is optimal
    if reason in (FailureReason.CARD_EXPIRED, FailureReason.INVALID_CARD_DETAILS):
        return AIRecommendation(
            action=RecoveryAction.PAYMENT_LINK,
            confidence=0.91,
            reason=(
                f"Payment failed due to {reason.value}. Retrying with stored card is futile; "
                "a secure payment link allows the customer to enter fresh, valid card or UPI credentials."
            ),
            expected_recovery=amount,
            risk_level="HIGH" if is_high_value else "LOW",
            proposed_discount_pct=0.0,
            requires_approval_recommendation=is_high_value,
            agent_mode="fallback",
            model_name=None,
        )

    # 4. Payment Limit Exceeded
    if reason == FailureReason.PAYMENT_LIMIT_EXCEEDED:
        if is_high_value:
            return AIRecommendation(
                action=RecoveryAction.ESCALATE,
                confidence=0.85,
                reason=(
                    f"Transaction amount INR {amount:,.2f} exceeded the customer's single-transaction banking limit. "
                    "Escalating to finance operations to offer corporate invoicing or split payment options."
                ),
                expected_recovery=amount,
                risk_level="HIGH",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=True,
                agent_mode="fallback",
                model_name=None,
            )
        else:
            return AIRecommendation(
                action=RecoveryAction.PAYMENT_LINK,
                confidence=0.82,
                reason=(
                    f"UPI/card transaction limit exceeded on INR {amount:,.2f} order. "
                    "Issuing a payment link so customer can switch to net banking or a different payment method."
                ),
                expected_recovery=amount,
                risk_level="MEDIUM",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=False,
                agent_mode="fallback",
                model_name=None,
            )

    # 5. Insufficient funds
    if reason == FailureReason.INSUFFICIENT_FUNDS:
        if prev_success > 0:
            return AIRecommendation(
                action=RecoveryAction.PAYMENT_LINK,
                confidence=0.80,
                reason=(
                    f"Established customer ({prev_success} past orders, INR {total_spend:,.2f} total spend) had insufficient balance. "
                    "Providing a payment link with a 48-hour expiration window so they can settle once funds are available."
                ),
                expected_recovery=amount,
                risk_level="HIGH" if is_high_value else "MEDIUM",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=is_high_value,
                agent_mode="fallback",
                model_name=None,
            )
        else:
            return AIRecommendation(
                action=RecoveryAction.REMINDER,
                confidence=0.74,
                reason=(
                    f"New customer transaction failed due to insufficient funds. "
                    "A polite digital reminder is recommended prior to initiating active recovery."
                ),
                expected_recovery=amount,
                risk_level="MEDIUM",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=False,
                agent_mode="fallback",
                model_name=None,
            )

    # 6. Transient network or bank outages (NETWORK_TIMEOUT, BANK_SERVER_DOWN)
    if reason in (FailureReason.NETWORK_TIMEOUT, FailureReason.BANK_SERVER_DOWN):
        if retries == 0 and prev_success > 0 and not is_high_value:
            return AIRecommendation(
                action=RecoveryAction.RETRY,
                confidence=0.92,
                reason=(
                    f"Temporary {reason.value} outage detected for loyal customer ({prev_success} past purchases). "
                    "Zero prior retries attempted; automated gateway retry has high conversion probability."
                ),
                expected_recovery=amount,
                risk_level="LOW",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=False,
                agent_mode="fallback",
                model_name=None,
            )
        else:
            return AIRecommendation(
                action=RecoveryAction.PAYMENT_LINK,
                confidence=0.86,
                reason=(
                    f"Transient gateway issue ({reason.value}). "
                    f"{'Prior retry failed; ' if retries > 0 else ''}"
                    "Sending a payment link ensures checkout continuity without repeatedly hammering bank switches."
                ),
                expected_recovery=amount,
                risk_level="HIGH" if is_high_value else "LOW",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=is_high_value,
                agent_mode="fallback",
                model_name=None,
            )

    # 7. Authentication Failed (OTP drop-off / abandonment)
    if reason == FailureReason.AUTHENTICATION_FAILED:
        if total_spend >= 10000.0 and amount <= 5000.0 and retries == 0:
            discount = 5.0
            return AIRecommendation(
                action=RecoveryAction.INCENTIVE,
                confidence=0.84,
                reason=(
                    f"High-value returning customer (INR {total_spend:,.2f} cumulative spend) dropped off during 3DS verification. "
                    f"A modest {discount}% incentive provides psychological re-engagement to complete order."
                ),
                expected_recovery=round(amount * (1.0 - discount / 100.0), 2),
                risk_level="LOW",
                proposed_discount_pct=discount,
                requires_approval_recommendation=False,
                agent_mode="fallback",
                model_name=None,
            )
        else:
            return AIRecommendation(
                action=RecoveryAction.PAYMENT_LINK,
                confidence=0.87,
                reason=(
                    "Customer abandoned OTP or failed 3D Secure verification. "
                    "Payment link sent via SMS/Email for effortless one-tap retry."
                ),
                expected_recovery=amount,
                risk_level="HIGH" if is_high_value else "LOW",
                proposed_discount_pct=0.0,
                requires_approval_recommendation=is_high_value,
                agent_mode="fallback",
                model_name=None,
            )

    # Default fallback
    return AIRecommendation(
        action=RecoveryAction.PAYMENT_LINK,
        confidence=0.75,
        reason=f"Recovery initiated for {reason.value} failure on INR {amount:,.2f} order.",
        expected_recovery=amount,
        risk_level="HIGH" if is_high_value else "MEDIUM",
        proposed_discount_pct=0.0,
        requires_approval_recommendation=is_high_value,
        agent_mode="fallback",
        model_name=None,
    )


class AIRecoveryAgent:
    """Agent orchestrating Gemini structured recovery recommendations with deterministic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None

        if self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning("Failed to initialize Google GenAI Client: %s", e)
                self._client = None

    def recommend(
        self,
        payment: PaymentRecord,
        force_fallback: bool = False,
    ) -> AIRecommendation:
        """Emits an AI-driven recovery recommendation for a payment record.
        
        If Gemini API key is missing or fails, seamlessly delegates to heuristic fallback.
        """
        if force_fallback or self._client is None or not self.api_key:
            return heuristic_fallback_recommend(payment)

        # Build detailed prompt containing rich payment and customer context
        user_prompt = (
            f"Analyze the following failed transaction and recommend the optimal recovery action:\n\n"
            f"TRANSACTION DETAILS:\n"
            f"- Payment ID: {payment.payment_id}\n"
            f"- Order ID: {payment.order_id}\n"
            f"- Amount: INR {payment.amount:,.2f}\n"
            f"- Failure Reason: {payment.failure_reason.value}\n"
            f"- Error Description: {payment.error_description or 'N/A'}\n"
            f"- Retry Count: {payment.retry_count}\n"
            f"- Product Category: {payment.product_category.value}\n"
            f"- Payment Status: {payment.payment_status.value}\n\n"
            f"CUSTOMER PROFILE:\n"
            f"- Customer ID: {payment.customer_id}\n"
            f"- Previous Successful Payments: {payment.customer_previous_payments}\n"
            f"- Previous Failed Payments: {payment.customer_previous_failures}\n"
            f"- Total Historical Spend: INR {payment.customer_total_spend:,.2f}\n"
            f"- Days Since Last Successful Payment: {payment.days_since_last_success if payment.days_since_last_success is not None else 'None (New customer)'}\n\n"
            f"Respond with JSON adhering to the required schema."
        )

        try:
            from google.genai import types
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=AIRecommendation,
                        temperature=0.2,
                    ),
                )
            except Exception as model_err:
                if "404" in str(model_err) or "not found" in str(model_err).lower():
                    response = self._client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=AIRecommendation,
                            temperature=0.2,
                        ),
                    )
                else:
                    raise model_err

            # Parse and validate response
            raw_text = response.text or ""
            raw_text = raw_text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            data = json.loads(raw_text)

            # Ensure valid action enum
            action_val = data.get("action")
            if isinstance(action_val, str):
                action_enum = RecoveryAction(action_val.upper())
            else:
                action_enum = RecoveryAction(action_val)

            recommendation = AIRecommendation(
                action=action_enum,
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.8)))),
                reason=str(data.get("reason", "AI recovery recommendation generated.")),
                expected_recovery=max(0.0, float(data.get("expected_recovery", payment.amount))),
                risk_level=str(data.get("risk_level", "LOW")).upper(),
                proposed_discount_pct=max(0.0, float(data.get("proposed_discount_pct", 0.0))),
                requires_approval_recommendation=bool(
                    data.get("requires_approval_recommendation", payment.amount > 15000.0)
                ),
                agent_mode="gemini",
                model_name=self.model_name,
            )
            return recommendation

        except Exception as e:
            logger.warning("Gemini generation failed: %s. Falling back to heuristic engine.", e)
            return heuristic_fallback_recommend(payment)


# Global default agent instance
_default_agent = AIRecoveryAgent()


def analyze_payment(
    payment: PaymentRecord,
    force_fallback: bool = False,
) -> AIRecommendation:
    """Convenience helper to analyze a single payment transaction."""
    return _default_agent.recommend(payment, force_fallback=force_fallback)


def analyze_batch(
    payments: List[PaymentRecord],
    force_fallback: bool = False,
) -> List[AIRecommendation]:
    """Convenience helper to analyze a batch of payment records."""
    return [_default_agent.recommend(p, force_fallback=force_fallback) for p in payments]
