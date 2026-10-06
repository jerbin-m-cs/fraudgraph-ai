"""
engine.py

Pass-1 fraud detection engine.

Scores each transaction using six independent risk factors:
  1. Unusual amount       (+20)
  2. New device           (+20)
  3. New merchant         (+15)
  4. Shared device        (+20)
  5. Shared merchant      (+15)
  6. New location         (+10)

Each factor produces a human-readable reason string.
Total is capped at 100.

Factor 7 (fraud ring connection) is added in a later pass (Step 5)
once the fraud ring has been detected from the graph.

Input:  data/sample_transactions.csv
Output: data/transactions_scored_pass1.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/sample_transactions.csv")
OUTPUT_PATH = Path("data/transactions_scored_pass1.csv")

# --- Weights (single source of truth) ---
WEIGHTS = {
    "unusual_amount": 20,
    "new_device": 20,
    "new_merchant": 15,
    "shared_device": 20,
    "shared_merchant": 15,
    "new_location": 10,
}

# --- Decision thresholds ---
BLOCK_THRESHOLD = 70
REVIEW_THRESHOLD = 40


def _reason_strings(row, acc_mean, acc_std, device_share, merchant_share):
    """
    Returns a list of (points, reason_text) tuples for each triggered factor.
    """
    triggered = []

    # --- Factor 1: Unusual amount ---
    if acc_std and not np.isnan(acc_std) and acc_std > 0:
        z = (row["amount"] - acc_mean) / acc_std
        if z > 2:
            triggered.append((
                WEIGHTS["unusual_amount"],
                f"Unusual amount (₹{row['amount']:.0f}, {z:.1f}σ above account avg)"
            ))

    # --- Factor 2: New device ---
    if row["device_is_new"]:
        triggered.append((
            WEIGHTS["new_device"],
            f"New device ({row['device']}) never seen for this account"
        ))

    # --- Factor 3: New merchant ---
    if row["merchant_is_new"]:
        triggered.append((
            WEIGHTS["new_merchant"],
            f"New merchant ({row['merchant']})"
        ))

    # --- Factor 4: Shared device ---
    if device_share > 1:
        triggered.append((
            WEIGHTS["shared_device"],
            f"Device {row['device']} shared by {device_share} accounts"
        ))

    # --- Factor 5: Shared merchant (unusual only) ---
    if 1 < merchant_share <= 10:
        triggered.append((
            WEIGHTS["shared_merchant"],
            f"Merchant {row['merchant']} shared by {merchant_share} accounts"
        ))

    # --- Factor 6: New location ---
    if row["location_is_new"]:
        triggered.append((
            WEIGHTS["new_location"],
            f"New location ({row['location']})"
        ))

    return triggered


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account", "timestamp"]).reset_index(drop=True)

    # --- Per-account amount baseline ---
    acc_stats = df.groupby("account")["amount"].agg(["mean", "std"]).reset_index()
    acc_stats.columns = ["account", "acc_mean", "acc_std"]
    df = df.merge(acc_stats, on="account", how="left")

    # --- Per-account seen-flags (chronological) ---
    df["device_is_new"] = df.groupby("account")["device"].transform(
        lambda s: ~s.duplicated(keep="first")
    )
    df["merchant_is_new"] = df.groupby("account")["merchant"].transform(
        lambda s: ~s.duplicated(keep="first")
    )
    df["location_is_new"] = df.groupby("account")["location"].transform(
        lambda s: ~s.duplicated(keep="first")
    )

    # --- Graph-level shared counts ---
    device_share = df.groupby("device")["account"].nunique().to_dict()
    merchant_share = df.groupby("merchant")["account"].nunique().to_dict()

    # --- Score each transaction ---
    scores, decisions, reasons = [], [], []

    for _, row in df.iterrows():
        triggered = _reason_strings(
            row,
            row["acc_mean"],
            row["acc_std"],
            device_share.get(row["device"], 1),
            merchant_share.get(row["merchant"], 1),
        )

        total = min(sum(p for p, _ in triggered), 100)

        if total >= BLOCK_THRESHOLD:
            decision = "BLOCK"
        elif total >= REVIEW_THRESHOLD:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        scores.append(total)
        decisions.append(decision)
        reasons.append([f"+{p} {txt}" for p, txt in triggered])

    df["risk_score"] = scores
    df["decision"] = decisions
    df["reasons"] = reasons

    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    scored = score_transactions(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_PATH, index=False)

    # --- Summary ---
    print(f"✓ Scored {len(scored)} transactions")
    print(f"  Output: {OUTPUT_PATH}")
    print()
    print("Decision breakdown:")
    print(scored["decision"].value_counts().to_string())
    print()

    # --- Sanity check: ring transactions ---
    ring = scored[scored["device"] == "D05"]
    print(f"Ring transactions (D05): {len(ring)}")
    print(f"  Avg risk score: {ring['risk_score'].mean():.1f}")
    print(f"  Min risk score: {ring['risk_score'].min()}")
    print(f"  Max risk score: {ring['risk_score'].max()}")
    print()

    print("Top 10 highest-risk transactions:")
    top = scored.nlargest(10, "risk_score")[
        ["txn_id", "account", "device", "merchant", "amount", "risk_score", "decision"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()