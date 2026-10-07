"""
inject_ring.py

Injects one coordinated fraud ring into the real Indian Financial Fraud Dataset.

The real dataset has zero coordinated rings — all fraud is per-transaction and
rule-based. To demonstrate the ring detection capability required by PS04, we
add one controlled ring: 5 customers who share a card, merchant, and location
across a 3-hour window.

Injection details (documented for transparency):
  - 5 accounts: chosen randomly from accounts with >=8 prior transactions
  - 1 new shared card: CARD_INJECTED_01
  - 1 existing low-risk merchant: chosen randomly
  - Shared location: a fixed state/city pair (unusual combination)
  - 6 transactions per account, all between 02:00-05:00 on 2024-06-15
  - Amounts: uniform random in [45000, 95000]
  - Fraud_Flag = 1, Fraud_Reason = "Coordinated Ring Activity (injected)"

Input:  data/transactions_clean.csv
Output: data/transactions_with_ring.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

INPUT = Path("data/transactions_clean.csv")
OUTPUT = Path("data/transactions_with_ring.csv")

# --- Reproducibility ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Ring configuration ---
N_RING_ACCOUNTS = 5
MIN_PRIOR_TXNS_PER_RING_ACCOUNT = 8
RING_CARD_ID = "CARD_INJECTED_01"
RING_TXNS_PER_ACCOUNT = 6
RING_AMOUNT_RANGE = (60000, 150000)
RING_DATE = datetime(2024, 6, 15)  # a Saturday
RING_HOUR_RANGE = (2, 5)           # 02:00 to 05:00
RING_LOCATION_STATE = "Lakshadweep"     # unusual — real dataset likely has few txns here
RING_LOCATION_CITY = "Kavaratti"
RING_REASON = "Coordinated Ring Activity (injected)"


def pick_ring_accounts(df: pd.DataFrame, injection_time: pd.Timestamp) -> list:
    """Pick N accounts with >= MIN_PRIOR_TXNS_PER_RING_ACCOUNT transactions
    strictly BEFORE the injection timestamp. This ensures the per-account
    baseline is genuine (not contaminated by the ring itself)."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    prior = df[df["timestamp"] < injection_time]
    counts = prior.groupby("account").size()
    eligible = counts[counts >= MIN_PRIOR_TXNS_PER_RING_ACCOUNT].index.tolist()
    if len(eligible) < N_RING_ACCOUNTS:
        raise ValueError(
            f"Only {len(eligible)} accounts have >= {MIN_PRIOR_TXNS_PER_RING_ACCOUNT} txns "
            f"before {injection_time}; need {N_RING_ACCOUNTS}."
        )
    return sorted(np.random.choice(eligible, N_RING_ACCOUNTS, replace=False).tolist())


def pick_ring_merchant(df: pd.DataFrame) -> str:
    """Pick a low-risk merchant so the ring's signal is not confused with merchant risk."""
    low_risk = df[df["Merchant_Risk_Level"] == "Low"]["merchant"].unique().tolist()
    if not low_risk:
        raise ValueError("No low-risk merchants in dataset.")
    return str(np.random.choice(low_risk))


def build_ring_transactions(df: pd.DataFrame, ring_accounts: list, ring_merchant: str) -> pd.DataFrame:
    """Build the 30 ring transactions (5 accounts × 6 each)."""
    rows = []
    for i, account in enumerate(ring_accounts):
        # Each account's 6 transactions spread across the 3-hour window
        for j in range(RING_TXNS_PER_ACCOUNT):
            hour = RING_HOUR_RANGE[0] + (j % (RING_HOUR_RANGE[1] - RING_HOUR_RANGE[0]))
            minute = (i * 7 + j * 11) % 60
            second = (i * 13 + j * 17) % 60

            rows.append({
                "txn_id": f"RING_{i:02d}_{j:02d}",
                "account": account,
                "device": RING_CARD_ID,
                "merchant": ring_merchant,
                "Transaction_Date": RING_DATE.strftime("%Y-%m-%d"),
                "Transaction_Time": f"{hour:02d}:{minute:02d}:{second:02d}",
                "amount": round(float(np.random.uniform(*RING_AMOUNT_RANGE)), 2),
                "Payment_Method": "Credit Card",
                "Transaction_Channel": "Online",
                "Device_Type": "Web Browser",
                "Transaction_Status": "Successful",
                "Is_International": 0,
                "Fraud_Flag": 1,
                "Fraud_Reason": RING_REASON,
                "location": RING_LOCATION_STATE,
                "Customer_City": RING_LOCATION_CITY,
                "Merchant_State": RING_LOCATION_STATE,
                "Merchant_City": RING_LOCATION_CITY,
                "timestamp": pd.Timestamp(
                    RING_DATE.replace(hour=hour, minute=minute, second=second)
                ),
                "Card_Type": "Platinum",
                "Card_Network": "Visa",
                "Credit_Limit": 500000,
                "Card_Status": "Active",
                "Contactless": "No",
                "Card_Mode": "Virtual",
                "Issue_Date": "2024-06-01",
                "Expiry_Date": "2027-06-01",
                "Merchant_Category": "Electronics",
                "Merchant_Risk_Level": "Low",
                "Merchant_Rating": 4.9,
                "Merchant_Status": "Active",
            })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    injection_time = pd.Timestamp(RING_DATE.replace(hour=RING_HOUR_RANGE[0], minute=0, second=0))
    ring_accounts = pick_ring_accounts(df, injection_time)
    ring_merchant = pick_ring_merchant(df)

    ring_df = build_ring_transactions(df, ring_accounts, ring_merchant)

    # Align column order: fill missing with NaN, drop extras
    full_cols = list(df.columns)
    for col in full_cols:
        if col not in ring_df.columns:
            ring_df[col] = None
    ring_df = ring_df[full_cols]

    combined = pd.concat([df, ring_df], ignore_index=True)
    combined = combined.sort_values(["account", "timestamp"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)

    print(f"✓ Injected ring into dataset")
    print(f"  Original rows: {len(df)}")
    print(f"  Ring rows:     {len(ring_df)}")
    print(f"  Total rows:    {len(combined)}")
    print()
    print(f"  Ring accounts: {ring_accounts}")
    print(f"  Ring card:     {RING_CARD_ID}")
    print(f"  Ring merchant: {ring_merchant}")
    print(f"  Ring location: {RING_LOCATION_CITY}, {RING_LOCATION_STATE}")
    print(f"  Ring time:     {RING_DATE.date()} 02:00–05:00")
    print(f"  Amount range:  ₹{RING_AMOUNT_RANGE[0]}–₹{RING_AMOUNT_RANGE[1]}")
    print()
    print(f"  Fraud rate before: {df['Fraud_Flag'].mean()*100:.4f}%")
    print(f"  Fraud rate after:  {combined['Fraud_Flag'].mean()*100:.4f}%")
    print(f"  Output: {OUTPUT}")


if __name__ == "__main__":
    main()