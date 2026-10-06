"""
generate_data.py

Generates a synthetic transaction dataset containing:
- 200 normal accounts with realistic behavior
- 1 injected fraud ring of 5 coordinated accounts

Output: data/sample_transactions.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# --- Reproducibility ---
np.random.seed(42)

# --- Configuration ---
NORMAL_ACCOUNTS = 200
NORMAL_DEVICES = 150
NORMAL_MERCHANTS = ["M_grocery", "M_fuel", "M_restaurant", "M_online", "M_retail"]
LOCATIONS = ["NY", "CA", "TX", "FL", "IL"]

FRAUD_ACCOUNTS = ["A17", "A23", "A41", "A58", "A72"]
FRAUD_DEVICE = "D05"
FRAUD_MERCHANT = "M09"
FRAUD_LOCATION = "RU"
FRAUD_AMOUNT_RANGE = (5000, 12000)
FRAUD_TXNS_PER_ACCOUNT = 6

OUTPUT_PATH = Path("data/sample_transactions.csv")


def generate():
    rows = []
    base_time = datetime(2025, 1, 1)

    # --- Normal accounts ---
    normal_accounts = [f"A{i:03d}" for i in range(NORMAL_ACCOUNTS)]
    normal_devices = [f"D{i:03d}" for i in range(NORMAL_DEVICES)]

    for acc in normal_accounts:
        home_loc = np.random.choice(LOCATIONS)
        home_dev = np.random.choice(normal_devices)
        n_txns = np.random.randint(5, 16)

        for _ in range(n_txns):
            t = base_time + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(6, 22),
                minutes=np.random.randint(0, 60),
            )
            # 90% use home location, 10% travel
            loc = home_loc if np.random.random() > 0.1 else np.random.choice(LOCATIONS)

            rows.append({
                "txn_id": f"T{len(rows):05d}",
                "timestamp": t,
                "account": acc,
                "device": home_dev,
                "merchant": np.random.choice(NORMAL_MERCHANTS),
                "amount": round(np.random.lognormal(mean=3, sigma=0.8), 2),
                "location": loc,
            })

    # --- Fraud ring ---
    ring_start = base_time + timedelta(days=20)

    for i, acc in enumerate(FRAUD_ACCOUNTS):
        for j in range(FRAUD_TXNS_PER_ACCOUNT):
            t = ring_start + timedelta(hours=j * 3 + i)
            rows.append({
                "txn_id": f"T{len(rows):05d}",
                "timestamp": t,
                "account": acc,
                "device": FRAUD_DEVICE,
                "merchant": FRAUD_MERCHANT,
                "amount": round(np.random.uniform(*FRAUD_AMOUNT_RANGE), 2),
                "location": FRAUD_LOCATION,
            })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # --- Summary ---
    print(f"✓ Generated {len(df)} transactions")
    print(f"  Normal accounts: {NORMAL_ACCOUNTS}")
    print(f"  Fraud ring accounts: {FRAUD_ACCOUNTS}")
    print(f"  Fraud ring size: {len(FRAUD_ACCOUNTS)} accounts")
    print(f"  Shared device: {FRAUD_DEVICE}")
    print(f"  Shared merchant: {FRAUD_MERCHANT}")
    print(f"  Shared location: {FRAUD_LOCATION}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()