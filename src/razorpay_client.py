"""Razorpay Test-Mode Service & Mock Sandbox Client for RecoverPay AI.

Wraps Razorpay Test-Mode APIs (Payment Links, Orders) with a deterministic
mock fallback sandbox, strict policy enforcement, and duplicate action protection.
"""

import hashlib
import logging
import os
import re
from typing import Any, Dict, Optional, Set, Tuple
from uuid import uuid4

from dotenv import load_dotenv

from src.models import (
    PaymentRecord,
    PaymentStatus,
    RecoveryAction,
    PolicyResult,
    PolicyDecisionType,
    RazorpayExecutionResult,
)

load_dotenv()

logger = logging.getLogger(__name__)


def sanitize_secrets(text: str, key_id: Optional[str] = None, key_secret: Optional[str] = None) -> str:
    """Sanitizes text to guarantee API keys and secrets never leak into logs or messages."""
    if not text:
        return ""
    sanitized = text
    # Mask specific key patterns
    if key_id and len(key_id) > 4:
        sanitized = sanitized.replace(key_id, f"rzp_test_...{key_id[-4:]}")
    if key_secret and len(key_secret) > 4:
        sanitized = sanitized.replace(key_secret, "[REDACTED_SECRET]")

    # Generic regex for any razorpay secret or key pattern
    sanitized = re.sub(r'rzp_test_[a-zA-Z0-9]{14,}', 'rzp_test_[REDACTED]', sanitized)
    sanitized = re.sub(r'rzp_live_[a-zA-Z0-9]{14,}', 'rzp_live_[REDACTED]', sanitized)
    return sanitized


class RazorpayRecoveryClient:
    """Manages Razorpay Test-Mode execution with a high-fidelity mock sandbox fallback."""

    def __init__(
        self,
        key_id: Optional[str] = None,
        key_secret: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        raw_key_id = key_id if key_id is not None else os.getenv("RAZORPAY_KEY_ID", "")
        raw_key_secret = key_secret if key_secret is not None else os.getenv("RAZORPAY_KEY_SECRET", "")
        configured_mode = (mode or os.getenv("RAZORPAY_MODE", "auto")).lower()

        self.key_id = raw_key_id.strip()
        self.key_secret = raw_key_secret.strip()

        # Check if real test credentials exist
        has_real_credentials = (
            bool(self.key_id and self.key_secret)
            and self.key_id.startswith("rzp_test_")
            and not self.key_id.startswith("rzp_test_your")
            and not self.key_secret.startswith("your_")
        )

        # Decide whether to run real test mode or mock sandbox
        if configured_mode == "mock":
            self.is_mock = True
        elif configured_mode == "test":
            self.is_mock = not has_real_credentials
        else:  # auto
            self.is_mock = not has_real_credentials

        self._client = None
        if not self.is_mock:
            try:
                import razorpay
                self._client = razorpay.Client(auth=(self.key_id, self.key_secret))
                self._client.set_app_details({"title": "RecoverPay AI", "version": "0.1.0"})
            except Exception as e:
                logger.warning("Failed to initialize Razorpay Client: %s. Reverting to Mock Sandbox.", e)
                self.is_mock = True
                self._client = None

        # Duplicate protection registry: (payment_id, action_name)
        self._executed_actions: Set[Tuple[str, str]] = set()

    def clear_execution_history(self) -> None:
        """Clears local execution cache (useful in automated tests)."""
        self._executed_actions.clear()

    # -----------------------------------------------------------------------
    # Low-Level Razorpay API Wrappers (Real or Mock)
    # -----------------------------------------------------------------------

    def create_payment_link(
        self,
        amount: float,
        customer_name: str,
        customer_email: str,
        customer_phone: Optional[str] = None,
        description: str = "RecoverPay AI Payment Link",
        reference_id: Optional[str] = None,
        expire_by_hours: int = 48,
    ) -> Dict[str, Any]:
        """Creates a Razorpay Payment Link (or mock sandbox link)."""
        amount_paise = int(round(amount * 100))
        ref_id = reference_id or f"plink_ref_{uuid4().hex[:8]}"

        if self.is_mock:
            digest = hashlib.md5(f"{ref_id}:{amount}".encode()).hexdigest()[:8]
            return {
                "id": f"mock_plink_{digest}",
                "entity": "payment_link",
                "amount": amount_paise,
                "amount_paid": 0,
                "currency": "INR",
                "status": "created",
                "reference_id": ref_id,
                "description": description,
                "customer": {
                    "name": customer_name,
                    "email": customer_email,
                    "contact": customer_phone or "+919800000000",
                },
                "short_url": f"https://rzp.io/i/mock_{digest}",
                "simulated": True,
            }

        payload: Dict[str, Any] = {
            "amount": amount_paise,
            "currency": "INR",
            "accept_partial": False,
            "reference_id": ref_id,
            "description": description,
            "customer": {
                "name": customer_name,
                "email": customer_email,
                "contact": customer_phone or "+919800000000",
            },
            "notify": {"sms": False, "email": True},
            "reminder_enable": True,
            "notes": {"source": "RecoverPay_AI_Autonomous_Recovery"},
        }
        return self._client.payment_link.create(payload)

    def create_order(
        self,
        amount: float,
        currency: str = "INR",
        receipt: Optional[str] = None,
        notes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates a Razorpay Order for retrying payments (or mock sandbox order)."""
        amount_paise = int(round(amount * 100))
        rcpt = receipt or f"rcpt_{uuid4().hex[:8]}"

        if self.is_mock:
            digest = hashlib.md5(f"{rcpt}:{amount}".encode()).hexdigest()[:8]
            return {
                "id": f"mock_order_{digest}",
                "entity": "order",
                "amount": amount_paise,
                "amount_paid": 0,
                "amount_due": amount_paise,
                "currency": currency,
                "receipt": rcpt,
                "status": "created",
                "attempts": 0,
                "notes": notes or {},
                "simulated": True,
            }

        payload = {
            "amount": amount_paise,
            "currency": currency,
            "receipt": rcpt,
            "notes": notes or {"source": "RecoverPay_AI_Retry"},
        }
        return self._client.order.create(payload)

    def fetch_order(self, order_id: str) -> Dict[str, Any]:
        """Fetches status of a Razorpay Order."""
        if self.is_mock or order_id.startswith("mock_"):
            return {
                "id": order_id,
                "entity": "order",
                "status": "created",
                "simulated": True,
            }
        return self._client.order.fetch(order_id)

    def fetch_payment(self, payment_id: str) -> Dict[str, Any]:
        """Fetches status of a payment."""
        if self.is_mock or payment_id.startswith("mock_"):
            return {
                "id": payment_id,
                "entity": "payment",
                "status": "captured",
                "simulated": True,
            }
        return self._client.payment.fetch(payment_id)

    def fetch_payment_link(self, plink_id: str) -> Dict[str, Any]:
        """Fetches status of a Payment Link."""
        if self.is_mock or plink_id.startswith("mock_"):
            return {
                "id": plink_id,
                "entity": "payment_link",
                "status": "created",
                "simulated": True,
            }
        return self._client.payment_link.fetch(plink_id)

    # -----------------------------------------------------------------------
    # High-Level Recovery Execution Engine
    # -----------------------------------------------------------------------

    def execute_recovery(
        self,
        payment: PaymentRecord,
        action: RecoveryAction,
        policy_result: PolicyResult,
        approved_by_merchant: bool = False,
    ) -> RazorpayExecutionResult:
        """Executes a recovery action strictly governed by the policy engine.
        
        Refuses execution if:
        - Policy decision is BLOCKED
        - Policy requires approval but approved_by_merchant is False
        - Payment is already RECOVERED
        - Action has already been executed for this payment (duplicate protection)
        """
        recovery_id = f"rec_{payment.payment_id}_{uuid4().hex[:6]}"

        # 1. State Check: Never recover already RECOVERED payments
        if payment.payment_status not in (PaymentStatus.FAILED, PaymentStatus.AT_RISK):
            return RazorpayExecutionResult(
                success=False,
                action=action,
                recovery_id=recovery_id,
                razorpay_id=None,
                status="INVALID_STATE",
                amount=payment.amount,
                message=(
                    f"Execution blocked: Payment status is {payment.payment_status.value}. "
                    "Already recovered or invalid transactions cannot be re-executed."
                ),
                simulated=self.is_mock,
                error_code="ALREADY_RECOVERED",
            )

        # 2. Policy Decision Check: Blocked policies cannot execute
        if policy_result.decision == PolicyDecisionType.BLOCKED:
            return RazorpayExecutionResult(
                success=False,
                action=action,
                recovery_id=recovery_id,
                razorpay_id=None,
                status="BLOCKED",
                amount=payment.amount,
                message=f"Execution blocked by Policy Engine: {policy_result.reason}",
                simulated=self.is_mock,
                error_code="POLICY_BLOCKED",
            )

        # 3. Approval Gate: If approval is required, merchant approval must be True
        if policy_result.requires_approval and not approved_by_merchant:
            return RazorpayExecutionResult(
                success=False,
                action=action,
                recovery_id=recovery_id,
                razorpay_id=None,
                status="APPROVAL_REQUIRED",
                amount=payment.amount,
                message="Execution blocked: merchant approval required.",
                simulated=self.is_mock,
                error_code="APPROVAL_REQUIRED",
            )

        # 4. Duplicate Action Protection
        action_key = (payment.payment_id, action.value)
        if action_key in self._executed_actions:
            return RazorpayExecutionResult(
                success=False,
                action=action,
                recovery_id=recovery_id,
                razorpay_id=None,
                status="DUPLICATE_PREVENTED",
                amount=payment.amount,
                message=f"Execution blocked: Action '{action.value}' has already been executed for payment {payment.payment_id}.",
                simulated=self.is_mock,
                error_code="DUPLICATE_ACTION",
            )

        # 5. Dispatch Action Execution
        try:
            # ---------------------------------------------------------------
            # NON-FINANCIAL ACTIONS (Never touch payment APIs)
            # ---------------------------------------------------------------
            if action == RecoveryAction.REMINDER:
                self._executed_actions.add(action_key)
                return RazorpayExecutionResult(
                    success=True,
                    action=action,
                    recovery_id=recovery_id,
                    razorpay_id=None,
                    status="REMINDER_SIMULATED",
                    amount=payment.amount,
                    message=f"Simulated payment reminder dispatched to {payment.customer_email}. No payment API called.",
                    simulated=self.is_mock,
                )

            if action == RecoveryAction.ESCALATE:
                self._executed_actions.add(action_key)
                return RazorpayExecutionResult(
                    success=True,
                    action=action,
                    recovery_id=recovery_id,
                    razorpay_id=None,
                    status="ESCALATED",
                    amount=payment.amount,
                    message=f"Transaction escalated to merchant support team for customer {payment.customer_name}. No payment API called.",
                    simulated=self.is_mock,
                )

            if action == RecoveryAction.STOP:
                self._executed_actions.add(action_key)
                return RazorpayExecutionResult(
                    success=True,
                    action=action,
                    recovery_id=recovery_id,
                    razorpay_id=None,
                    status="STOPPED",
                    amount=payment.amount,
                    message=f"Recovery stopped for payment {payment.payment_id}. No action taken.",
                    simulated=self.is_mock,
                )

            # ---------------------------------------------------------------
            # FINANCIAL ACTION: RETRY -> Fresh Razorpay Order
            # ---------------------------------------------------------------
            if action == RecoveryAction.RETRY:
                order = self.create_order(
                    amount=payment.amount,
                    currency=payment.currency,
                    receipt=f"rcpt_{payment.order_id[-10:]}",
                    notes={
                        "original_payment_id": payment.payment_id,
                        "recovery_id": recovery_id,
                    },
                )
                self._executed_actions.add(action_key)
                return RazorpayExecutionResult(
                    success=True,
                    action=action,
                    recovery_id=recovery_id,
                    razorpay_id=order.get("id"),
                    status="ORDER_CREATED",
                    amount=payment.amount,
                    message=f"Fresh Razorpay order {order.get('id')} created for retry.",
                    simulated=self.is_mock,
                )

            # ---------------------------------------------------------------
            # FINANCIAL ACTION: PAYMENT_LINK -> Hosted Checkout Link
            # ---------------------------------------------------------------
            if action == RecoveryAction.PAYMENT_LINK:
                plink = self.create_payment_link(
                    amount=payment.amount,
                    customer_name=payment.customer_name,
                    customer_email=payment.customer_email,
                    customer_phone=payment.customer_phone,
                    description=f"Payment recovery for order {payment.order_id}",
                    reference_id=recovery_id,
                )
                self._executed_actions.add(action_key)
                return RazorpayExecutionResult(
                    success=True,
                    action=action,
                    recovery_id=recovery_id,
                    razorpay_id=plink.get("id"),
                    short_url=plink.get("short_url"),
                    status="PAYMENT_LINK_CREATED",
                    amount=payment.amount,
                    message=f"Razorpay Payment Link {plink.get('id')} created successfully ({plink.get('short_url')}).",
                    simulated=self.is_mock,
                )

            # ---------------------------------------------------------------
            # FINANCIAL ACTION: INCENTIVE -> Discounted Payment Link
            # (Uses strictly the final payable amount calculated by policy engine)
            # ---------------------------------------------------------------
            if action == RecoveryAction.INCENTIVE:
                final_amount = (
                    policy_result.final_payable_amount
                    if policy_result.final_payable_amount is not None
                    else payment.amount
                )
                discount_amount = policy_result.effective_discount

                plink = self.create_payment_link(
                    amount=final_amount,
                    customer_name=payment.customer_name,
                    customer_email=payment.customer_email,
                    customer_phone=payment.customer_phone,
                    description=f"Special discounted recovery for order {payment.order_id}",
                    reference_id=recovery_id,
                )
                self._executed_actions.add(action_key)
                return RazorpayExecutionResult(
                    success=True,
                    action=action,
                    recovery_id=recovery_id,
                    razorpay_id=plink.get("id"),
                    short_url=plink.get("short_url"),
                    status="PAYMENT_LINK_CREATED",
                    amount=final_amount,
                    message=(
                        f"Razorpay Payment Link {plink.get('id')} created with INR {discount_amount:,.2f} "
                        f"deterministic discount applied. Final payable: INR {final_amount:,.2f}."
                    ),
                    simulated=self.is_mock,
                )

        except Exception as e:
            raw_err = str(e)
            safe_err = sanitize_secrets(raw_err, self.key_id, self.key_secret)
            logger.error("Razorpay execution failed: %s", safe_err)
            return RazorpayExecutionResult(
                success=False,
                action=action,
                recovery_id=recovery_id,
                razorpay_id=None,
                status="FAILED",
                amount=payment.amount,
                message=f"Razorpay API error: {safe_err}",
                simulated=self.is_mock,
                error_code="API_ERROR",
            )

        return RazorpayExecutionResult(
            success=False,
            action=action,
            recovery_id=recovery_id,
            status="UNKNOWN",
            amount=payment.amount,
            message="Unhandled recovery action.",
            simulated=self.is_mock,
            error_code="UNHANDLED_ACTION",
        )


# Global default client instance
_default_client = RazorpayRecoveryClient()


def execute_recovery(
    payment: PaymentRecord,
    action: RecoveryAction,
    policy_result: PolicyResult,
    approved_by_merchant: bool = False,
    client: Optional[RazorpayRecoveryClient] = None,
) -> RazorpayExecutionResult:
    """Convenience helper to execute recovery through default client."""
    active_client = client or _default_client
    return active_client.execute_recovery(
        payment=payment,
        action=action,
        policy_result=policy_result,
        approved_by_merchant=approved_by_merchant,
    )
