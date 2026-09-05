"""Synthetic Payment Dataset Generator for Revnexa.

Generates 500+ realistic, internally consistent failed or at-risk payment transactions
reflecting real Razorpay failure modes, customer profiles, and product categories.
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.models import (
    PaymentRecord,
    PaymentStatus,
    FailureReason,
    ProductCategory,
)

# Realistic error descriptions mapped to Razorpay error codes
ERROR_DESCRIPTIONS: Dict[FailureReason, List[str]] = {
    FailureReason.BANK_SERVER_DOWN: [
        "Issuing bank servers are down or unresponsive (BAD_GATEWAY)",
        "Bank partner system degraded during settlement check",
        "Intermittent core banking switch outage at HDFC/SBI",
    ],
    FailureReason.NETWORK_TIMEOUT: [
        "Gateway connection timed out waiting for 3DS challenge response",
        "UPI payment switch timeout after 45 seconds",
        "Network dropped during handshake with card network",
    ],
    FailureReason.AUTHENTICATION_FAILED: [
        "Customer dropped off or entered incorrect 3D Secure OTP",
        "Biometric authentication canceled by payer",
        "Two-factor authentication session expired",
    ],
    FailureReason.INSUFFICIENT_FUNDS: [
        "Account balance insufficient to complete debit",
        "Card credit limit exceeded for current billing cycle",
        "Prepaid wallet balance lower than required order value",
    ],
    FailureReason.PAYMENT_LIMIT_EXCEEDED: [
        "Per-transaction UPI limit of ₹1,00,000 exceeded",
        "Daily bank-mandated card spending threshold exceeded",
        "Net banking corporate limit requires secondary authorization",
    ],
    FailureReason.CARD_EXPIRED: [
        "Card expiration date is in the past (EXPIRED_CARD)",
        "Saved tokenized card has expired; update required",
    ],
    FailureReason.INVALID_CARD_DETAILS: [
        "CVV or expiration month format validation failed",
        "Card number Luhn checksum mismatch",
    ],
    FailureReason.SUSPECTED_FRAUD: [
        "High risk score assigned by Razorpay Thirdwatch fraud engine",
        "Velocity threshold triggered: multiple cards tried from blacklisted IP",
        "Stolen card alert signaled by issuing bank network",
    ],
}

CUSTOMER_NAMES = [
    "Aarav Sharma", "Aditi Patel", "Rohan Mehta", "Pooja Verma", "Vikram Singh",
    "Ananya Iyer", "Kavya Nair", "Rahul Deshmukh", "Sneha Roy", "Arjun Gupta",
    "Deepak Joshi", "Neha Kulkarni", "Sanjay Reddy", "Priya Sen", "Rajesh Kumar",
    "Meera Pillai", "Gaurav Malhotra", "Divya Bhat", "Amitabh Rao", "Shreya Das",
    "Manish Choudhary", "Ritu Mishra", "Karan Kapoor", "Simran Bajaj", "Naveen Menon",
    "Tarun Saxena", "Preeti Tiwari", "Harsh Vardhan", "Ishita Sengupta", "Suresh Nair"
]


def generate_synthetic_dataset(
    count: int = 500,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[PaymentRecord]:
    """Generates a reproducible dataset of PaymentRecord instances."""
    random.seed(seed)
    records: List[PaymentRecord] = []

    # Create a reusable customer pool (300 distinct customers)
    # Some customers appear multiple times to simulate repeat purchase/failure patterns
    num_customers = 300
    customer_pool: List[Dict[str, Any]] = []

    for i in range(1, num_customers + 1):
        cust_id = f"cust_synth_{i:04d}"
        name = random.choice(CUSTOMER_NAMES)
        first_name = name.split()[0].lower()
        email = f"{first_name}.{i}@example.com"
        phone = f"+9198{random.randint(10000000, 99999999)}"

        # 4 distinct customer archetypes:
        # 1. VIP / High Spend (15%): high previous payments, high total spend, low failures
        # 2. Regular / Repeat (40%): moderate payments, moderate spend
        # 3. Churn-prone / Weak (25%): few payments, multiple past failures
        # 4. Brand New (20%): 0 previous payments, 0 spend
        archetype = random.choices(
            ["VIP", "REGULAR", "WEAK", "NEW"],
            weights=[0.15, 0.40, 0.25, 0.20],
            k=1
        )[0]

        if archetype == "VIP":
            prev_payments = random.randint(8, 25)
            prev_failures = random.randint(0, 2)
            avg_ticket = random.uniform(3000, 15000)
            total_spend = round(prev_payments * avg_ticket, 2)
            days_since = random.randint(2, 30)
        elif archetype == "REGULAR":
            prev_payments = random.randint(3, 7)
            prev_failures = random.randint(0, 3)
            avg_ticket = random.uniform(800, 4000)
            total_spend = round(prev_payments * avg_ticket, 2)
            days_since = random.randint(5, 60)
        elif archetype == "WEAK":
            prev_payments = random.randint(1, 2)
            prev_failures = random.randint(3, 7)
            avg_ticket = random.uniform(500, 2000)
            total_spend = round(prev_payments * avg_ticket, 2)
            days_since = random.randint(30, 180)
        else:  # NEW
            prev_payments = 0
            prev_failures = random.randint(0, 1)
            total_spend = 0.0
            days_since = None

        customer_pool.append({
            "customer_id": cust_id,
            "customer_name": name,
            "customer_email": email,
            "customer_phone": phone,
            "archetype": archetype,
            "customer_previous_payments": prev_payments,
            "customer_previous_failures": prev_failures,
            "customer_total_spend": total_spend,
            "days_since_last_success": days_since,
        })

    # Realistic failure distribution:
    # Most common: BANK_SERVER_DOWN, NETWORK_TIMEOUT, AUTHENTICATION_FAILED
    # Less common: INSUFFICIENT_FUNDS, PAYMENT_LIMIT_EXCEEDED, CARD_EXPIRED
    # Edge cases: INVALID_CARD_DETAILS, SUSPECTED_FRAUD (terminal stops)
    failure_distribution = [
        (FailureReason.NETWORK_TIMEOUT, 0.24),
        (FailureReason.BANK_SERVER_DOWN, 0.22),
        (FailureReason.AUTHENTICATION_FAILED, 0.20),
        (FailureReason.INSUFFICIENT_FUNDS, 0.12),
        (FailureReason.PAYMENT_LIMIT_EXCEEDED, 0.08),
        (FailureReason.CARD_EXPIRED, 0.06),
        (FailureReason.INVALID_CARD_DETAILS, 0.04),
        (FailureReason.SUSPECTED_FRAUD, 0.04),
    ]
    failure_reasons = [item[0] for item in failure_distribution]
    failure_weights = [item[1] for item in failure_distribution]

    product_categories = [
        (ProductCategory.SAAS, 0.30),
        (ProductCategory.ECOMMERCE, 0.35),
        (ProductCategory.B2B_SERVICES, 0.15),
        (ProductCategory.EDTECH, 0.12),
        (ProductCategory.HEALTHCARE, 0.08),
    ]
    prod_cats = [item[0] for item in product_categories]
    prod_weights = [item[1] for item in product_categories]

    base_time = datetime.now(timezone.utc) - timedelta(days=7)

    for i in range(1, count + 1):
        payment_id = f"pay_synth_{i:05d}"
        order_id = f"order_synth_{i:05d}"
        
        # Pick customer
        cust = random.choice(customer_pool)
        
        # Pick failure reason & product category
        reason = random.choices(failure_reasons, weights=failure_weights, k=1)[0]
        category = random.choices(prod_cats, weights=prod_weights, k=1)[0]
        
        # Determine amount based on category and risk profile
        # Ensure ~15-20% high-value transactions (> ₹15,000 threshold)
        is_high_value = random.random() < 0.18
        
        if is_high_value:
            if category in (ProductCategory.B2B_SERVICES, ProductCategory.SAAS):
                amount = round(random.uniform(18000, 85000), 2)
            else:
                amount = round(random.uniform(15500, 45000), 2)
        else:
            if category == ProductCategory.ECOMMERCE:
                amount = round(random.uniform(299, 4999), 2)
            elif category == ProductCategory.SAAS:
                amount = round(random.uniform(999, 12000), 2)
            elif category == ProductCategory.EDTECH:
                amount = round(random.uniform(1500, 14500), 2)
            elif category == ProductCategory.HEALTHCARE:
                amount = round(random.uniform(499, 8000), 2)
            else:  # B2B_SERVICES
                amount = round(random.uniform(5000, 14999), 2)

        # Retry count generation:
        # Most transactions are 0 retries (fresh failures).
        # Some have 1 or 2 retries (repeated transient issues).
        # A few have 3 or more retries (deliberate failure stop candidates!).
        if reason == FailureReason.SUSPECTED_FRAUD:
            retry_count = random.choice([0, 1])  # Fraud shouldn't have many retries
        else:
            retry_count = random.choices([0, 1, 2, 3, 4], weights=[0.55, 0.22, 0.13, 0.07, 0.03], k=1)[0]

        # Payment status: mostly FAILED, some AT_RISK (e.g. pending/abandoned session)
        payment_status = PaymentStatus.AT_RISK if (reason == FailureReason.AUTHENTICATION_FAILED and random.random() < 0.35) else PaymentStatus.FAILED

        # Timestamp distributed over the last 7 days
        record_time = base_time + timedelta(
            days=random.uniform(0, 6.9),
            hours=random.uniform(0, 23),
            minutes=random.uniform(0, 59)
        )
        created_at_str = record_time.isoformat()

        error_desc = random.choice(ERROR_DESCRIPTIONS[reason])

        # Instantiate PaymentRecord with full Pydantic validation
        record = PaymentRecord(
            payment_id=payment_id,
            customer_id=cust["customer_id"],
            customer_name=cust["customer_name"],
            customer_email=cust["customer_email"],
            customer_phone=cust["customer_phone"],
            order_id=order_id,
            amount=amount,
            currency="INR",
            payment_status=payment_status,
            failure_reason=reason,
            error_description=error_desc,
            retry_count=retry_count,
            customer_previous_payments=cust["customer_previous_payments"],
            customer_previous_failures=cust["customer_previous_failures"],
            customer_total_spend=cust["customer_total_spend"],
            days_since_last_success=cust["days_since_last_success"],
            product_category=category,
            created_at=created_at_str,
        )
        records.append(record)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2)

    return records


def print_dataset_summary(records: List[PaymentRecord]) -> None:
    """Computes and prints key statistical metrics of the dataset."""
    total_records = len(records)
    total_value = sum(r.amount for r in records)
    high_value_count = sum(1 for r in records if r.amount > 15000.0)
    high_value_total = sum(r.amount for r in records if r.amount > 15000.0)

    # Counts by failure reason
    reason_counts: Dict[str, int] = {}
    for r in records:
        reason_counts[r.failure_reason.value] = reason_counts.get(r.failure_reason.value, 0) + 1

    # Counts by category
    cat_counts: Dict[str, int] = {}
    for r in records:
        cat_counts[r.product_category.value] = cat_counts.get(r.product_category.value, 0) + 1

    # Counts by retry count
    retry_counts: Dict[int, int] = {}
    for r in records:
        retry_counts[r.retry_count] = retry_counts.get(r.retry_count, 0) + 1

    # Counts by customer history
    new_custs = sum(1 for r in records if r.customer_previous_payments == 0)
    vip_custs = sum(1 for r in records if r.customer_total_spend >= 25000.0)

    print("=" * 65)
    print(" REVNEXA - SYNTHETIC DATASET SUMMARY")
    print("=" * 65)
    print(f"Total Records Generated    : {total_records}")
    print(f"Total Failed Volume        : INR {total_value:,.2f}")
    print(f"Average Transaction Amount : INR {total_value / total_records:,.2f}")
    print(f"High-Value Orders (>15k)   : {high_value_count} ({high_value_count / total_records * 100:.1f}%) [Total: INR {high_value_total:,.2f}]")
    print(f"New Customers (0 history)  : {new_custs} ({new_custs / total_records * 100:.1f}%)")
    print(f"High-Spend / VIP Customers : {vip_custs} ({vip_custs / total_records * 100:.1f}%)")
    print("-" * 65)
    print("Distribution by Failure Reason:")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  - {reason:<26}: {cnt:>4} ({cnt / total_records * 100:>5.1f}%)")
    print("-" * 65)
    print("Distribution by Product Category:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cat:<26}: {cnt:>4} ({cnt / total_records * 100:>5.1f}%)")
    print("-" * 65)
    print("Distribution by Retry Count:")
    for retries, cnt in sorted(retry_counts.items()):
        status = " (Stop candidate)" if retries >= 3 else ""
        print(f"  - Retry count {retries:<14}: {cnt:>4} ({cnt / total_records * 100:>5.1f}%){status}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic payment records for Revnexa.")
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of records to generate (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_payments.json",
        help="Destination JSON file path (default: data/synthetic_payments.json)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    records = generate_synthetic_dataset(count=args.count, seed=args.seed, output_path=out_path)
    print(f"\n[SUCCESS] Wrote {len(records)} validated records to: {out_path.resolve()}\n")
    print_dataset_summary(records)


if __name__ == "__main__":
    main()
