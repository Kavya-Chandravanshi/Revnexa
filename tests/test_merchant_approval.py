"""Unit tests for Merchant Approval Queue."""

import pytest
from src.models import RecoveryAction, ApprovalStatus
from src.merchant_approval import MerchantApprovalQueue


def test_1_create_pending_request():
    """1. Verify creation of a pending approval request."""
    queue = MerchantApprovalQueue()
    req = queue.create_request(
        payment_id="pay_app_001",
        customer_id="cust_app_001",
        amount=35000.0,
        requested_action=RecoveryAction.PAYMENT_LINK,
        ai_reason="High value order",
        ai_confidence=0.88,
        policy_reason="Exceeds ₹15,000 threshold",
    )
    assert req.approval_id.startswith("app_req_pay_app_001_")
    assert req.status == ApprovalStatus.PENDING
    assert req.amount == 35000.0
    assert req.payment_id == "pay_app_001"


def test_2_list_pending_requests():
    """2. Verify listing pending requests filters correctly."""
    queue = MerchantApprovalQueue()
    queue.create_request("p1", "c1", 20000.0, RecoveryAction.PAYMENT_LINK, "reason", 0.9, "policy")
    queue.create_request("p2", "c2", 25000.0, RecoveryAction.PAYMENT_LINK, "reason", 0.9, "policy")

    pending = queue.list_pending_requests()
    assert len(pending) == 2

    # Approve one
    queue.approve_request(pending[0].approval_id, reviewer="Admin")
    assert len(queue.list_pending_requests()) == 1


def test_3_approve_request():
    """3. Verify approving a request changes status and records reviewer."""
    queue = MerchantApprovalQueue()
    req = queue.create_request("p1", "c1", 20000.0, RecoveryAction.PAYMENT_LINK, "reason", 0.9, "policy")
    approved = queue.approve_request(req.approval_id, reviewer="Senior Merchant", notes="Approved high-value SaaS client")

    assert approved.status == ApprovalStatus.APPROVED
    assert approved.reviewer == "Senior Merchant"
    assert approved.notes == "Approved high-value SaaS client"
    assert approved.reviewed_at is not None
    assert queue.is_approved("p1") is True


def test_4_reject_request():
    """4. Verify rejecting a request changes status to REJECTED."""
    queue = MerchantApprovalQueue()
    req = queue.create_request("p1", "c1", 20000.0, RecoveryAction.PAYMENT_LINK, "reason", 0.9, "policy")
    rejected = queue.reject_request(req.approval_id, reviewer="Risk Officer", notes="Suspicious pattern")

    assert rejected.status == ApprovalStatus.REJECTED
    assert rejected.reviewer == "Risk Officer"
    assert queue.is_approved("p1") is False


def test_5_prevent_double_approval():
    """5. Verify that approving an already approved request raises ValueError."""
    queue = MerchantApprovalQueue()
    req = queue.create_request("p1", "c1", 20000.0, RecoveryAction.PAYMENT_LINK, "reason", 0.9, "policy")
    queue.approve_request(req.approval_id)

    with pytest.raises(ValueError, match="already APPROVED"):
        queue.approve_request(req.approval_id)


def test_6_prevent_approval_after_rejection():
    """6. Verify that an already rejected request cannot be subsequently approved."""
    queue = MerchantApprovalQueue()
    req = queue.create_request("p1", "c1", 20000.0, RecoveryAction.PAYMENT_LINK, "reason", 0.9, "policy")
    queue.reject_request(req.approval_id)

    with pytest.raises(ValueError, match="already REJECTED"):
        queue.approve_request(req.approval_id)


def test_7_unique_approval_ids():
    """7. Verify approval IDs are unique across multiple distinct requests."""
    queue = MerchantApprovalQueue()
    req1 = queue.create_request("p1", "c1", 20000.0, RecoveryAction.PAYMENT_LINK, "r", 0.9, "p")
    req2 = queue.create_request("p2", "c2", 20000.0, RecoveryAction.PAYMENT_LINK, "r", 0.9, "p")
    assert req1.approval_id != req2.approval_id
