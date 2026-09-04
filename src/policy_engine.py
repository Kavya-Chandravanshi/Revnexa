"""Deterministic Policy Engine and Safety Guardrails for RecoverPay AI.

Enforces non-negotiable financial, risk, and operational boundaries.
Contains ZERO LLM / generative components and ZERO external API dependencies.
"""

from typing import List, Optional, Sequence, Union, Set
from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    RecoveryAction,
    PolicyResult,
    PolicyDecisionType,
)

# Standard Policy Audit Codes
CODE_ACTION_ALLOWED = "ACTION_ALLOWED"
CODE_APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
CODE_HIGH_VALUE_APPROVAL = "HIGH_VALUE_APPROVAL"
CODE_RETRY_LIMIT_EXCEEDED = "RETRY_LIMIT_EXCEEDED"
CODE_FRAUD_PROTECTION = "FRAUD_PROTECTION"
CODE_INCENTIVE_CAP_EXCEEDED = "INCENTIVE_CAP_EXCEEDED"
CODE_DUPLICATE_ACTION = "DUPLICATE_ACTION"
CODE_INVALID_PAYMENT_STATE = "INVALID_PAYMENT_STATE"
CODE_INVALID_ACTION = "INVALID_ACTION"


class PolicyEngine:
    """Configurable, deterministic policy validator for payment recovery actions."""

    def __init__(
        self,
        high_value_threshold: float = 15000.0,
        max_retries: int = 2,
        max_discount_pct: float = 10.0,
        max_discount_amount: float = 500.0,
    ):
        self.high_value_threshold = float(high_value_threshold)
        self.max_retries = int(max_retries)
        self.max_discount_pct = float(max_discount_pct)
        self.max_discount_amount = float(max_discount_amount)

    def evaluate(
        self,
        payment: PaymentRecord,
        action: Union[RecoveryAction, str],
        proposed_discount_pct: float = 0.0,
        proposed_discount_amount: Optional[float] = None,
        recent_actions: Optional[Sequence[Union[RecoveryAction, str]]] = None,
    ) -> PolicyResult:
        """Evaluates a proposed recovery action against deterministic safety rules.
        
        Evaluation Order:
        1. Validate action enum
        2. Validate payment state (FAILED or AT_RISK only)
        3. Special safe actions (STOP, ESCALATE)
        4. Check fraud protection
        5. Check duplicate action protection
        6. Check retry limit
        7. Calculate & validate incentive caps
        8. Check high-value transaction approval threshold
        9. Return final decision
        """
        # -------------------------------------------------------------------
        # STEP 1: Validate action
        # -------------------------------------------------------------------
        validated_action: Optional[RecoveryAction] = None
        if isinstance(action, RecoveryAction):
            validated_action = action
        elif isinstance(action, str):
            try:
                validated_action = RecoveryAction(action.upper())
            except ValueError:
                validated_action = None

        if validated_action is None:
            allowed_actions_str = ", ".join(a.value for a in RecoveryAction)
            return PolicyResult(
                decision=PolicyDecisionType.BLOCKED,
                allowed=False,
                requires_approval=False,
                reason=f"Action '{action}' is invalid. Allowed recovery actions are: {allowed_actions_str}.",
                policy_codes=[CODE_INVALID_ACTION],
                payment_id=payment.payment_id,
                action=None,
                final_payable_amount=payment.amount,
            )

        # -------------------------------------------------------------------
        # STEP 2: Validate payment state
        # -------------------------------------------------------------------
        allowed_states = {PaymentStatus.FAILED, PaymentStatus.AT_RISK}
        if payment.payment_status not in allowed_states:
            return PolicyResult(
                decision=PolicyDecisionType.BLOCKED,
                allowed=False,
                requires_approval=False,
                reason=(
                    f"Cannot recover transaction with status '{payment.payment_status.value}'. "
                    "Only FAILED or AT_RISK transactions are eligible for recovery. "
                    "Never attempt recovery on already RECOVERED, PROCESSING, or ABANDONED payments."
                ),
                policy_codes=[CODE_INVALID_PAYMENT_STATE],
                payment_id=payment.payment_id,
                action=validated_action,
                final_payable_amount=payment.amount,
            )

        # -------------------------------------------------------------------
        # STEP 3: Special safe actions (STOP & ESCALATE)
        # -------------------------------------------------------------------
        # RULE 8: STOP is always allowed because it is a safe no-action outcome
        if validated_action == RecoveryAction.STOP:
            return PolicyResult(
                decision=PolicyDecisionType.ALLOWED,
                allowed=True,
                requires_approval=False,
                reason="STOP action accepted. Recovery halted safely without customer contact or risk.",
                effective_discount=0.0,
                max_allowed_discount=0.0,
                policy_codes=[CODE_ACTION_ALLOWED],
                payment_id=payment.payment_id,
                action=validated_action,
                final_payable_amount=payment.amount,
            )

        # RULE 9: ESCALATE is always allowed on valid payments (routes to human review / account manager)
        if validated_action == RecoveryAction.ESCALATE:
            return PolicyResult(
                decision=PolicyDecisionType.ALLOWED,
                allowed=True,
                requires_approval=False,
                reason="ESCALATE action accepted. Routed to account manager / support specialist for manual handling.",
                effective_discount=0.0,
                max_allowed_discount=0.0,
                policy_codes=[CODE_ACTION_ALLOWED],
                payment_id=payment.payment_id,
                action=validated_action,
                final_payable_amount=payment.amount,
            )

        # -------------------------------------------------------------------
        # STEP 4: Check fraud protection (RULE 3)
        # -------------------------------------------------------------------
        if payment.failure_reason == FailureReason.SUSPECTED_FRAUD:
            return PolicyResult(
                decision=PolicyDecisionType.BLOCKED,
                allowed=False,
                requires_approval=False,
                reason=(
                    "Transaction flagged as SUSPECTED_FRAUD by risk engine. "
                    f"Automated action '{validated_action.value}' is blocked to prevent financial loss. "
                    "Only ESCALATE or STOP is permitted for fraudulent records."
                ),
                policy_codes=[CODE_FRAUD_PROTECTION],
                payment_id=payment.payment_id,
                action=validated_action,
                final_payable_amount=payment.amount,
            )

        # -------------------------------------------------------------------
        # STEP 5: Check duplicate action protection (RULE 6)
        # -------------------------------------------------------------------
        if recent_actions:
            normalized_recent: Set[str] = set()
            for item in recent_actions:
                if isinstance(item, RecoveryAction):
                    normalized_recent.add(item.value)
                elif isinstance(item, str):
                    normalized_recent.add(item.upper())

            action_key = validated_action.value
            scoped_key = f"{payment.payment_id}:{action_key}"

            if action_key in normalized_recent or scoped_key in normalized_recent:
                return PolicyResult(
                    decision=PolicyDecisionType.BLOCKED,
                    allowed=False,
                    requires_approval=False,
                    reason=(
                        f"Action '{validated_action.value}' has already been executed recently for payment {payment.payment_id}. "
                        "Duplicate action blocked to protect customer experience."
                    ),
                    policy_codes=[CODE_DUPLICATE_ACTION],
                    payment_id=payment.payment_id,
                    action=validated_action,
                    final_payable_amount=payment.amount,
                )

        # -------------------------------------------------------------------
        # STEP 6: Check retry limit (RULE 2)
        # -------------------------------------------------------------------
        if validated_action == RecoveryAction.RETRY:
            if payment.retry_count >= self.max_retries:
                return PolicyResult(
                    decision=PolicyDecisionType.BLOCKED,
                    allowed=False,
                    requires_approval=False,
                    reason=(
                        f"Automated retry limit reached. Transaction has already been attempted {payment.retry_count} times "
                        f"(maximum allowed: {self.max_retries}). Action RETRY blocked. Safe outcome: ESCALATE or STOP."
                    ),
                    policy_codes=[CODE_RETRY_LIMIT_EXCEEDED],
                    payment_id=payment.payment_id,
                    action=validated_action,
                    final_payable_amount=payment.amount,
                )

        # -------------------------------------------------------------------
        # STEP 7: Calculate incentive cap (RULE 5)
        # -------------------------------------------------------------------
        # Max allowed discount is min(10% of amount, ₹500)
        percentage_cap = round(payment.amount * (self.max_discount_pct / 100.0), 2)
        max_allowed_discount = round(min(percentage_cap, self.max_discount_amount), 2)

        effective_discount = 0.0

        if validated_action == RecoveryAction.INCENTIVE:
            # 1. Check if percentage exceeds cap
            if proposed_discount_pct > (self.max_discount_pct + 0.001):
                return PolicyResult(
                    decision=PolicyDecisionType.BLOCKED,
                    allowed=False,
                    requires_approval=False,
                    reason=(
                        f"Proposed incentive of {proposed_discount_pct:.1f}% exceeds "
                        f"maximum allowed limit of {self.max_discount_pct:.1f}%."
                    ),
                    max_allowed_discount=max_allowed_discount,
                    policy_codes=[CODE_INCENTIVE_CAP_EXCEEDED],
                    payment_id=payment.payment_id,
                    action=validated_action,
                    final_payable_amount=payment.amount,
                )

            # 2. Determine raw calculated discount
            if proposed_discount_amount is not None:
                calculated_discount = float(proposed_discount_amount)
            else:
                calculated_discount = payment.amount * (proposed_discount_pct / 100.0)

            # 3. Check absolute rupee cap (₹500)
            if calculated_discount > (self.max_discount_amount + 0.001):
                return PolicyResult(
                    decision=PolicyDecisionType.BLOCKED,
                    allowed=False,
                    requires_approval=False,
                    reason=(
                        f"Proposed incentive amount INR {calculated_discount:,.2f} exceeds "
                        f"the maximum allowed absolute cap of INR {self.max_discount_amount:,.2f}."
                    ),
                    max_allowed_discount=max_allowed_discount,
                    policy_codes=[CODE_INCENTIVE_CAP_EXCEEDED],
                    payment_id=payment.payment_id,
                    action=validated_action,
                    final_payable_amount=payment.amount,
                )

            # 4. Check if proposed discount amount exceeds percentage limit
            if calculated_discount > (percentage_cap + 0.001):
                return PolicyResult(
                    decision=PolicyDecisionType.BLOCKED,
                    allowed=False,
                    requires_approval=False,
                    reason=(
                        f"Proposed incentive amount INR {calculated_discount:,.2f} exceeds "
                        f"{self.max_discount_pct:.1f}% of transaction value (INR {percentage_cap:,.2f})."
                    ),
                    max_allowed_discount=max_allowed_discount,
                    policy_codes=[CODE_INCENTIVE_CAP_EXCEEDED],
                    payment_id=payment.payment_id,
                    action=validated_action,
                    final_payable_amount=payment.amount,
                )

            effective_discount = round(calculated_discount, 2)

        final_payable = round(max(0.0, payment.amount - effective_discount), 2)

        # -------------------------------------------------------------------
        # STEP 8: Check high-value transaction approval (RULE 4)
        # -------------------------------------------------------------------
        # Financial actions on high-value orders (> ₹15,000) require merchant approval
        financial_actions = {
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.RETRY,
            RecoveryAction.INCENTIVE,
        }

        if payment.amount > self.high_value_threshold and validated_action in financial_actions:
            reason_parts: List[str] = [
                f"Transaction amount INR {payment.amount:,.2f} exceeds autonomous recovery limit of INR {self.high_value_threshold:,.2f}."
            ]
            if validated_action == RecoveryAction.INCENTIVE:
                reason_parts.append(
                    f"Incentive of INR {effective_discount:,.2f} is within safety caps, but requires merchant confirmation due to high transaction value."
                )
            reason_parts.append("Merchant approval required before executing payment gateway action.")

            return PolicyResult(
                decision=PolicyDecisionType.REQUIRES_APPROVAL,
                allowed=False,
                requires_approval=True,
                reason=" ".join(reason_parts),
                effective_discount=effective_discount,
                max_allowed_discount=max_allowed_discount,
                policy_codes=[CODE_HIGH_VALUE_APPROVAL, CODE_APPROVAL_REQUIRED],
                payment_id=payment.payment_id,
                action=validated_action,
                final_payable_amount=final_payable,
            )

        # -------------------------------------------------------------------
        # STEP 9: Final Decision: ACTION_ALLOWED
        # -------------------------------------------------------------------
        reason_msg = f"Action '{validated_action.value}' successfully validated against all safety rules."
        if effective_discount > 0:
            reason_msg += f" Applied deterministic discount of INR {effective_discount:,.2f} (final amount: INR {final_payable:,.2f})."

        return PolicyResult(
            decision=PolicyDecisionType.ALLOWED,
            allowed=True,
            requires_approval=False,
            reason=reason_msg,
            effective_discount=effective_discount,
            max_allowed_discount=max_allowed_discount,
            policy_codes=[CODE_ACTION_ALLOWED],
            payment_id=payment.payment_id,
            action=validated_action,
            final_payable_amount=final_payable,
        )


# Global default policy engine instance
_default_engine = PolicyEngine()


def evaluate_policy(
    payment: PaymentRecord,
    action: Union[RecoveryAction, str],
    proposed_discount_pct: float = 0.0,
    proposed_discount_amount: Optional[float] = None,
    recent_actions: Optional[Sequence[Union[RecoveryAction, str]]] = None,
    high_value_threshold: float = 15000.0,
    max_retries: int = 2,
    max_discount_pct: float = 10.0,
    max_discount_amount: float = 500.0,
) -> PolicyResult:
    """Convenience module-level function for evaluating policy on a payment record."""
    if (
        high_value_threshold == 15000.0
        and max_retries == 2
        and max_discount_pct == 10.0
        and max_discount_amount == 500.0
    ):
        return _default_engine.evaluate(
            payment=payment,
            action=action,
            proposed_discount_pct=proposed_discount_pct,
            proposed_discount_amount=proposed_discount_amount,
            recent_actions=recent_actions,
        )

    engine = PolicyEngine(
        high_value_threshold=high_value_threshold,
        max_retries=max_retries,
        max_discount_pct=max_discount_pct,
        max_discount_amount=max_discount_amount,
    )
    return engine.evaluate(
        payment=payment,
        action=action,
        proposed_discount_pct=proposed_discount_pct,
        proposed_discount_amount=proposed_discount_amount,
        recent_actions=recent_actions,
    )
