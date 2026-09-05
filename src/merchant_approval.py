"""Merchant Approval Queue for Revnexa.

Manages high-value and policy-flagged recovery actions requiring human-in-the-loop sign-off.
Provides a local, in-memory queue that can be easily queried and updated by the dashboard.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from src.models import (
    ApprovalRequest,
    ApprovalStatus,
    RecoveryAction,
)


class MerchantApprovalQueue:
    """Manages merchant review requests for high-value or policy-flagged recovery actions."""

    def __init__(self):
        # Maps approval_id -> ApprovalRequest
        self._requests: Dict[str, ApprovalRequest] = {}
        # Index payment_id -> approval_id
        self._payment_index: Dict[str, str] = {}

    def clear(self) -> None:
        """Clears all requests from memory (useful for testing and re-runs)."""
        self._requests.clear()
        self._payment_index.clear()

    def create_request(
        self,
        payment_id: str,
        customer_id: str,
        amount: float,
        requested_action: RecoveryAction,
        ai_reason: str,
        ai_confidence: float,
        policy_reason: str,
    ) -> ApprovalRequest:
        """Creates a new pending approval request with a unique ID."""
        # If an approval already exists for this payment and is PENDING, return it
        existing_id = self._payment_index.get(payment_id)
        if existing_id and existing_id in self._requests:
            existing = self._requests[existing_id]
            if existing.status == ApprovalStatus.PENDING:
                return existing

        approval_id = f"app_req_{payment_id}_{uuid4().hex[:6]}"
        request = ApprovalRequest(
            approval_id=approval_id,
            payment_id=payment_id,
            customer_id=customer_id,
            amount=amount,
            requested_action=requested_action,
            ai_reason=ai_reason,
            ai_confidence=ai_confidence,
            policy_reason=policy_reason,
            status=ApprovalStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._requests[approval_id] = request
        self._payment_index[payment_id] = approval_id
        return request

    def get_request(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Retrieves an approval request by its unique approval_id."""
        return self._requests.get(approval_id)

    def get_request_by_payment_id(self, payment_id: str) -> Optional[ApprovalRequest]:
        """Retrieves the latest approval request associated with a payment_id."""
        approval_id = self._payment_index.get(payment_id)
        return self._requests.get(approval_id) if approval_id else None

    def list_all_requests(self) -> List[ApprovalRequest]:
        """Returns all approval requests ordered by creation time descending."""
        return sorted(self._requests.values(), key=lambda r: r.created_at, reverse=True)

    def list_pending_requests(self) -> List[ApprovalRequest]:
        """Returns all requests currently awaiting review."""
        return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def approve_request(
        self,
        approval_id: str,
        reviewer: str = "Merchant",
        notes: Optional[str] = None,
    ) -> ApprovalRequest:
        """Approves a pending request. Rejects double-approval or approval after rejection."""
        request = self._requests.get(approval_id)
        if not request:
            raise KeyError(f"Approval request '{approval_id}' not found.")

        if request.status == ApprovalStatus.APPROVED:
            raise ValueError(f"Request '{approval_id}' is already APPROVED by {request.reviewer}.")

        if request.status == ApprovalStatus.REJECTED:
            raise ValueError(f"Cannot approve request '{approval_id}' because it was already REJECTED.")

        request.status = ApprovalStatus.APPROVED
        request.reviewer = reviewer
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        request.notes = notes
        return request

    def reject_request(
        self,
        approval_id: str,
        reviewer: str = "Merchant",
        notes: Optional[str] = None,
    ) -> ApprovalRequest:
        """Rejects a pending request. Rejects repeated rejection or rejection after approval."""
        request = self._requests.get(approval_id)
        if not request:
            raise KeyError(f"Approval request '{approval_id}' not found.")

        if request.status == ApprovalStatus.REJECTED:
            raise ValueError(f"Request '{approval_id}' is already REJECTED by {request.reviewer}.")

        if request.status == ApprovalStatus.APPROVED:
            raise ValueError(f"Cannot reject request '{approval_id}' because it was already APPROVED.")

        request.status = ApprovalStatus.REJECTED
        request.reviewer = reviewer
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        request.notes = notes
        return request

    def is_approved(self, payment_id: str) -> bool:
        """Checks if a payment has an approved sign-off."""
        req = self.get_request_by_payment_id(payment_id)
        return bool(req and req.status == ApprovalStatus.APPROVED)


# Global default queue instance
_default_queue = MerchantApprovalQueue()


def create_approval_request(
    payment_id: str,
    customer_id: str,
    amount: float,
    requested_action: RecoveryAction,
    ai_reason: str,
    ai_confidence: float,
    policy_reason: str,
    queue: Optional[MerchantApprovalQueue] = None,
) -> ApprovalRequest:
    """Convenience function to create an approval request."""
    active_queue = queue or _default_queue
    return active_queue.create_request(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=amount,
        requested_action=requested_action,
        ai_reason=ai_reason,
        ai_confidence=ai_confidence,
        policy_reason=policy_reason,
    )


def list_pending_approvals(queue: Optional[MerchantApprovalQueue] = None) -> List[ApprovalRequest]:
    """Convenience function to list all pending approvals."""
    active_queue = queue or _default_queue
    return active_queue.list_pending_requests()


def approve_request(
    approval_id: str,
    reviewer: str = "Merchant",
    notes: Optional[str] = None,
    queue: Optional[MerchantApprovalQueue] = None,
) -> ApprovalRequest:
    """Convenience function to approve a request."""
    active_queue = queue or _default_queue
    return active_queue.approve_request(approval_id, reviewer=reviewer, notes=notes)


def reject_request(
    approval_id: str,
    reviewer: str = "Merchant",
    notes: Optional[str] = None,
    queue: Optional[MerchantApprovalQueue] = None,
) -> ApprovalRequest:
    """Convenience function to reject a request."""
    active_queue = queue or _default_queue
    return active_queue.reject_request(approval_id, reviewer=reviewer, notes=notes)
