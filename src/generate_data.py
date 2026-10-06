"""
generate_data.py

Generates a synthetic transaction dataset containing:
- 200 normal accounts with realistic behavior
- 1 fraud ring of 5 accounts, each with a normal history followed by fraud burst

Design rationale:
- Each ring account first behaves like a normal user (small amounts, own device,
  common merchants, home location).
- Then, starting on day 20, they switch to coordinated ring behavior:
  shared device D05, shared merchant M09, shared location RU, high amounts.
- This models a realistic "account takeover into coordinated fraud" scenario
  and gives the scoring engine a genuine per-account baseline to compare against.

Output: data/sample_transactions.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

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
FRAUD_BASELINE_TXNS = (8, 13)   # random int range for normal-history phase
FRAUD_ONSET_DAY = 20            # fraud begins on day 20

OUTPUT_PATH = Path("data/sample_transactions.csv")


def _normal_txn(rows, acc, t, device, merchant, amount, location):
    rows.append({
        "txn_id": f"T{len(rows):05d}",
        "timestamp": t,
        "account": acc,
        "device": device,
        "merchant": merchant,
        "amount": round(amount, 2),
        "location": location,
    })


def generate():
    rows = []
    base_time = datetime(2025, 1, 1)

    normal_accounts = [f"A{i:03d}" for i in range(NORMAL_ACCOUNTS)]
    normal_devices = [f"D{i:03d}" for i in range(NORMAL_DEVICES)]

    # --- Normal accounts ---
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
            loc = home_loc if np.random.random() > 0.1 else np.random.choice(LOCATIONS)
            _normal_txn(
                rows, acc, t,
                device=home_dev,
                merchant=np.random.choice(NORMAL_MERCHANTS),
                amount=np.random.lognormal(mean=3, sigma=0.8),
                location=loc,
            )

    # --- Fraud ring accounts: normal phase then fraud phase ---
    # IMPORTANT: exclude A17..A72 from being generated twice.
    # The normal loop above already created 200 accounts A000..A199.
    # Ring accounts A017, A023, A041, A058, A072 are part of that set.
    # We must remove their "normal" transactions before adding ring behavior,
    # OR skip them during the normal loop. Here we do the cleaner thing:
    # remove any rows whose account is in FRAUD_ACCOUNTS, then regenerate them.

    rows = [r for r in rows if r["account"] not in FRAUD_ACCOUNTS]

    for i, acc in enumerate(FRAUD_ACCOUNTS):
        home_loc = np.random.choice(LOCATIONS)
        home_dev = np.random.choice(normal_devices)
        n_baseline = np.random.randint(*FRAUD_BASELINE_TXNS)

        # --- Normal history phase (before day 20) ---
        for _ in range(n_baseline):
            t = base_time + timedelta(
                days=np.random.randint(0, FRAUD_ONSET_DAY - 1),
                hours=np.random.randint(6, 22),
                minutes=np.random.randint(0, 60),
            )
            loc = home_loc if np.random.random() > 0.1 else np.random.choice(LOCATIONS)
            _normal_txn(
                rows, acc, t,
                device=home_dev,
                merchant=np.random.choice(NORMAL_MERCHANTS),
                amount=np.random.lognormal(mean=3, sigma=0.8),
                location=loc,
            )

        # --- Fraud burst phase (from day 20) ---
        ring_start = base_time + timedelta(days=FRAUD_ONSET_DAY)
        for j in range(FRAUD_TXNS_PER_ACCOUNT):
            t = ring_start + timedelta(hours=j * 3 + i)
            _normal_txn(
                rows, acc, t,
                device=FRAUD_DEVICE,
                merchant=FRAUD_MERCHANT,
                amount=np.random.uniform(*FRAUD_AMOUNT_RANGE),
                location=FRAUD_LOCATION,
            )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # --- Summary ---
    print(f"✓ Generated {len(df)} transactions")
    print(f"  Normal accounts: {NORMAL_ACCOUNTS}")
    print(f"  Ring accounts: {FRAUD_ACCOUNTS}")
    print()

    for acc in FRAUD_ACCOUNTS:
        sub = df[df["account"] == acc]
        normal = sub[sub["device"] != FRAUD_DEVICE]
        ring = sub[sub["device"] == FRAUD_DEVICE]
        print(f"  {acc}: {len(normal)} normal txns (avg ₹{normal['amount'].mean():.0f}), "
              f"{len(ring)} ring txns (avg ₹{ring['amount'].mean():.0f})")

    print(f"\n  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()