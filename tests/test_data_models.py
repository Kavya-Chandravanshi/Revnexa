"""Unit tests for PaymentRecord model validation and synthetic data generator."""

import pytest
from pydantic import ValidationError

from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    ProductCategory,
    RecoveryAction,
)
from src.data_generator import generate_synthetic_dataset


def test_valid_payment_record():
    """Verify that a well-formed PaymentRecord passes validation."""
    record = PaymentRecord(
        payment_id="pay_test_001",
        customer_id="cust_test_001",
        customer_name="Test User",
        customer_email="test@example.com",
        order_id="order_test_001",
        amount=2500.0,
        currency="INR",
        payment_status=PaymentStatus.FAILED,
        failure_reason=FailureReason.BANK_SERVER_DOWN,
        retry_count=1,
        customer_previous_payments=4,
        customer_previous_failures=1,
        customer_total_spend=10000.0,
        days_since_last_success=14,
        product_category=ProductCategory.SAAS,
    )
    assert record.payment_id == "pay_test_001"
    assert record.amount == 2500.0
    assert record.currency == "INR"


def test_amount_must_be_strictly_positive():
    """Verify that zero or negative transaction amounts are rejected."""
    with pytest.raises(ValidationError):
        PaymentRecord(
            payment_id="pay_test_002",
            customer_id="cust_test_002",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_002",
            amount=0.0,  # Invalid
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )

    with pytest.raises(ValidationError):
        PaymentRecord(
            payment_id="pay_test_003",
            customer_id="cust_test_003",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_003",
            amount=-150.0,  # Invalid
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )


def test_retry_count_cannot_be_negative():
    """Verify that negative retry counts are rejected."""
    with pytest.raises(ValidationError):
        PaymentRecord(
            payment_id="pay_test_004",
            customer_id="cust_test_004",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_004",
            amount=500.0,
            retry_count=-1,  # Invalid
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )


def test_currency_validation():
    """Verify that non-INR currencies are rejected in this MVP."""
    with pytest.raises(ValidationError):
        PaymentRecord(
            payment_id="pay_test_005",
            customer_id="cust_test_005",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_005",
            amount=500.0,
            currency="USD",  # Invalid
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )


def test_customer_history_consistency_rules():
    """Verify that customer payment history and spend must be internally consistent."""
    # 1. 0 previous payments BUT positive spend -> must fail
    with pytest.raises(ValidationError, match="customer_total_spend"):
        PaymentRecord(
            payment_id="pay_test_006",
            customer_id="cust_test_006",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_006",
            amount=500.0,
            customer_previous_payments=0,
            customer_total_spend=2500.0,  # Inconsistent!
            days_since_last_success=None,
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )

    # 2. 0 previous payments BUT days_since_last_success is set -> must fail
    with pytest.raises(ValidationError, match="days_since_last_success"):
        PaymentRecord(
            payment_id="pay_test_007",
            customer_id="cust_test_007",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_007",
            amount=500.0,
            customer_previous_payments=0,
            customer_total_spend=0.0,
            days_since_last_success=5,  # Inconsistent!
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )

    # 3. Positive previous payments BUT 0 spend -> must fail
    with pytest.raises(ValidationError, match="customer_total_spend"):
        PaymentRecord(
            payment_id="pay_test_008",
            customer_id="cust_test_008",
            customer_name="Test User",
            customer_email="test@example.com",
            order_id="order_test_008",
            amount=500.0,
            customer_previous_payments=3,
            customer_total_spend=0.0,  # Inconsistent!
            days_since_last_success=10,
            failure_reason=FailureReason.NETWORK_TIMEOUT,
            product_category=ProductCategory.ECOMMERCE,
        )


def test_data_generator_500_records():
    """Verify that data generator creates >= 500 records with full consistency and diversity."""
    records = generate_synthetic_dataset(count=500, seed=42)
    assert len(records) == 500

    payment_ids = set()
    order_ids = set()
    failure_reasons_seen = set()

    high_value_count = 0
    repeat_retries_count = 0
    new_customer_count = 0
    fraud_count = 0

    for r in records:
        assert isinstance(r, PaymentRecord)
        assert r.amount > 0
        assert r.currency == "INR"
        assert r.retry_count >= 0

        # Unique IDs
        assert r.payment_id not in payment_ids
        assert r.order_id not in order_ids
        payment_ids.add(r.payment_id)
        order_ids.add(r.order_id)

        failure_reasons_seen.add(r.failure_reason)

        if r.amount > 15000.0:
            high_value_count += 1
        if r.retry_count >= 3:
            repeat_retries_count += 1
        if r.customer_previous_payments == 0:
            new_customer_count += 1
        if r.failure_reason == FailureReason.SUSPECTED_FRAUD:
            fraud_count += 1

    # Verify all 8 failure reasons are present
    assert len(failure_reasons_seen) == len(FailureReason)

    # Verify diversity thresholds
    assert high_value_count >= 50, f"Expected >= 50 high value orders, got {high_value_count}"
    assert repeat_retries_count >= 20, f"Expected >= 20 repeat retry orders, got {repeat_retries_count}"
    assert new_customer_count >= 50, f"Expected >= 50 new customers, got {new_customer_count}"
    assert fraud_count >= 5, f"Expected >= 5 fraud records, got {fraud_count}"


def test_data_generator_reproducibility():
    """Verify that the same random seed yields identical datasets."""
    run1 = generate_synthetic_dataset(count=50, seed=123)
    run2 = generate_synthetic_dataset(count=50, seed=123)

    assert len(run1) == len(run2)
    for r1, r2 in zip(run1, run2):
        assert r1.payment_id == r2.payment_id
        assert r1.amount == r2.amount
        assert r1.failure_reason == r2.failure_reason
        assert r1.customer_id == r2.customer_id
