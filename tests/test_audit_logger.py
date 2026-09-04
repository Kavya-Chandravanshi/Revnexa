"""Unit tests for Append-Only Audit Logger."""

import pytest
from src.audit_logger import AuditLogger, EVENT_PAYMENT_DETECTED, EVENT_AI_ANALYZED


def test_1_append_event():
    """1. Verify appending an event to the audit trail."""
    logger = AuditLogger()
    entry = logger.log_event(
        payment_id="pay_001",
        event_type=EVENT_PAYMENT_DETECTED,
        status="INGESTED",
        message="Payment failure recorded",
        metadata={"amount": 2500.0},
    )
    assert entry.event_id.startswith("evt_")
    assert entry.payment_id == "pay_001"
    assert entry.event_type == EVENT_PAYMENT_DETECTED
    assert logger.count() == 1


def test_2_retrieve_events_chronologically():
    """2. Verify events are retrieved in order of insertion."""
    logger = AuditLogger()
    logger.log_event("pay_001", "STEP_1", message="First event")
    logger.log_event("pay_002", "STEP_2", message="Second event")
    logger.log_event("pay_003", "STEP_3", message="Third event")

    events = logger.get_events()
    assert len(events) == 3
    assert events[0].message == "First event"
    assert events[1].message == "Second event"
    assert events[2].message == "Third event"

    # Test reverse
    rev = logger.get_events(reverse=True)
    assert rev[0].message == "Third event"


def test_3_retrieve_events_for_one_payment():
    """3. Verify filtering events for a specific payment ID."""
    logger = AuditLogger()
    logger.log_event("pay_AAA", "EVT_1", message="AAA-1")
    logger.log_event("pay_BBB", "EVT_2", message="BBB-1")
    logger.log_event("pay_AAA", "EVT_3", message="AAA-2")

    aaa_events = logger.get_events_for_payment("pay_AAA")
    assert len(aaa_events) == 2
    assert aaa_events[0].message == "AAA-1"
    assert aaa_events[1].message == "AAA-2"

    bbb_events = logger.get_events_for_payment("pay_BBB")
    assert len(bbb_events) == 1
    assert bbb_events[0].message == "BBB-1"


def test_4_event_ids_are_unique():
    """4. Verify each event receives a unique identifier."""
    logger = AuditLogger()
    e1 = logger.log_event("p1", "E1")
    e2 = logger.log_event("p1", "E2")
    assert e1.event_id != e2.event_id


def test_5_secrets_are_not_logged():
    """5. Verify that API keys and secrets are masked upon logging."""
    logger = AuditLogger()
    entry = logger.log_event(
        payment_id="pay_001",
        event_type="AUTH",
        message="Request using key rzp_test_abcdef12345678 and secret AIzaSySuperSecretKey123456",
        metadata={"key": "rzp_live_1234567890abcdef", "safe_param": "regular_value"},
    )
    assert "rzp_test_abcdef12345678" not in entry.message
    assert "[REDACTED]" in entry.message or "rzp_[REDACTED]" in entry.message
    assert "AIzaSySuperSecretKey123456" not in entry.message
    assert "rzp_live_1234567890abcdef" not in entry.metadata["key"]


def test_6_metadata_is_preserved():
    """6. Verify complex structured metadata payloads are accurately stored."""
    logger = AuditLogger()
    entry = logger.log_event(
        payment_id="pay_001",
        event_type="DIAGNOSTIC",
        metadata={"retry_count": 2, "discount": 5.0, "nested": {"tier": "VIP"}},
    )
    assert entry.metadata["retry_count"] == 2
    assert entry.metadata["discount"] == 5.0
    assert entry.metadata["nested"]["tier"] == "VIP"
