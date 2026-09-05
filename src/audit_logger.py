"""Append-Only Audit Logger for Revnexa.

Captures immutable, chronological records of all system state transitions,
AI proposals, policy decisions, merchant approvals, and Razorpay recovery results.
Guarantees that sensitive secrets are sanitized before being recorded.
"""

from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.models import AuditLogEntry


# Standard Event Types
EVENT_PAYMENT_DETECTED = "PAYMENT_DETECTED"
EVENT_AI_ANALYZED = "AI_ANALYZED"
EVENT_POLICY_EVALUATED = "POLICY_EVALUATED"
EVENT_APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
EVENT_APPROVAL_GRANTED = "APPROVAL_GRANTED"
EVENT_APPROVAL_REJECTED = "APPROVAL_REJECTED"
EVENT_RECOVERY_EXECUTION_STARTED = "RECOVERY_EXECUTION_STARTED"
EVENT_RECOVERY_SUCCEEDED = "RECOVERY_SUCCEEDED"
EVENT_RECOVERY_FAILED = "RECOVERY_FAILED"
EVENT_RECOVERY_BLOCKED = "RECOVERY_BLOCKED"
EVENT_RECOVERY_STOPPED = "RECOVERY_STOPPED"
EVENT_RECOVERY_ESCALATED = "RECOVERY_ESCALATED"
EVENT_DUPLICATE_BLOCKED = "DUPLICATE_BLOCKED"


def _sanitize_value(val: Any) -> Any:
    """Recursively redacts API keys and secrets from logged data."""
    if isinstance(val, str):
        # Mask razorpay test/live keys or potential secrets
        val = re.sub(r'rzp_(?:test|live)_[a-zA-Z0-9]{10,}', 'rzp_[REDACTED]', val)
        val = re.sub(r'(?:AIzaSy|sk-|secret)[a-zA-Z0-9_\-]{16,}', '[REDACTED_SECRET]', val)
        return val
    elif isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize_value(item) for item in val]
    return val


class AuditLogger:
    """In-memory append-only audit trail logger with chronological indexing."""

    def __init__(self):
        # Append-only chronological list
        self._entries: List[AuditLogEntry] = []
        # Index payment_id -> list of AuditLogEntry indices
        self._payment_index: Dict[str, List[int]] = {}

    def clear(self) -> None:
        """Clears logs from memory (useful for testing and re-runs)."""
        self._entries.clear()
        self._payment_index.clear()

    def log_event(
        self,
        payment_id: str,
        event_type: str,
        status: str = "INFO",
        action: Optional[str] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        recovery_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Appends a new event to the audit ledger."""
        event_id = f"evt_{uuid4().hex[:10]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Sanitize message and metadata
        safe_message = _sanitize_value(message)
        safe_metadata = _sanitize_value(metadata or {})

        entry = AuditLogEntry(
            event_id=event_id,
            timestamp=timestamp,
            payment_id=payment_id,
            action=action,
            event_type=event_type,
            status=status,
            message=safe_message,
            metadata=safe_metadata,
            recovery_id=recovery_id,
        )

        idx = len(self._entries)
        self._entries.append(entry)

        if payment_id not in self._payment_index:
            self._payment_index[payment_id] = []
        self._payment_index[payment_id].append(idx)

        return entry

    def get_events(self, limit: Optional[int] = None, reverse: bool = False) -> List[AuditLogEntry]:
        """Retrieves events in chronological order (or reverse chronological if specified)."""
        if reverse:
            items = list(reversed(self._entries))
        else:
            items = list(self._entries)
        return items[:limit] if limit is not None else items

    def get_events_for_payment(self, payment_id: str) -> List[AuditLogEntry]:
        """Retrieves all chronological audit events for a specific payment ID."""
        indices = self._payment_index.get(payment_id, [])
        return [self._entries[i] for i in indices]

    def count(self) -> int:
        """Returns total events logged."""
        return len(self._entries)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Exports audit trail as a list of primitive dicts."""
        return [e.model_dump() for e in self._entries]


# Global default audit logger
_default_logger = AuditLogger()


def log_event(
    payment_id: str,
    event_type: str,
    status: str = "INFO",
    action: Optional[str] = None,
    message: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    recovery_id: Optional[str] = None,
    logger: Optional[AuditLogger] = None,
) -> AuditLogEntry:
    """Convenience function to append an event to the default audit logger."""
    active_logger = logger or _default_logger
    return active_logger.log_event(
        payment_id=payment_id,
        event_type=event_type,
        status=status,
        action=action,
        message=message,
        metadata=metadata,
        recovery_id=recovery_id,
    )


def get_events(limit: Optional[int] = None, reverse: bool = False, logger: Optional[AuditLogger] = None) -> List[AuditLogEntry]:
    """Convenience function to get events."""
    active_logger = logger or _default_logger
    return active_logger.get_events(limit=limit, reverse=reverse)


def get_events_for_payment(payment_id: str, logger: Optional[AuditLogger] = None) -> List[AuditLogEntry]:
    """Convenience function to get events for a payment."""
    active_logger = logger or _default_logger
    return active_logger.get_events_for_payment(payment_id)
