"""
engine.py

Pass-1 fraud detection engine.

Scores each transaction using five independent risk factors:
  1. Unusual amount       (+25)  — z-score > 3 vs. PRIOR transactions only
  2. New device           (+20)  — first time, account has >=5 prior txns
  3. New merchant         (+15)  — first time, account has >=5 prior txns
  4. Shared device        (+25)  — device used by >=4 accounts
  5. Shared merchant      (+15)  — merchant used by 3-10 accounts (unusual)

Key design: per-account baselines are computed using ONLY PRIOR transactions
(rolling/expanding window), never future ones. This avoids look-ahead bias
and correctly flags the first anomalous transaction.

Input:  data/sample_transactions.csv
Output: data/transactions_scored_pass1.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/sample_transactions.csv")
OUTPUT_PATH = Path("data/transactions_scored_pass1.csv")

WEIGHTS = {
    "unusual_amount": 25,
    "new_device": 20,
    "new_merchant": 15,
    "shared_device": 25,
    "shared_merchant": 15,
}

AMOUNT_ZSCORE_THRESHOLD = 3.0
MIN_PRIOR_TXNS_FOR_NOVELTY = 5
SHARED_DEVICE_MIN_ACCOUNTS = 4
SHARED_MERCHANT_MIN_ACCOUNTS = 3
SHARED_MERCHANT_MAX_ACCOUNTS = 10

BLOCK_THRESHOLD = 70
REVIEW_THRESHOLD = 40


def _compute_prior_baselines(df):
    """
    For each row, compute mean and std of THAT account's amounts
    using only transactions strictly BEFORE this row (chronologically).

    Returns two arrays aligned with df's index.
    """
    means = np.full(len(df), np.nan)
    stds = np.full(len(df), np.nan)

    df = df.reset_index(drop=True)
    for acc, group in df.groupby("account", sort=False):
        amounts = group["amount"].values
        idxs = group.index.values
        for i, idx in enumerate(idxs):
            if i == 0:
                continue  # no prior history
            prior = amounts[:i]
            means[idx] = prior.mean()
            # Use at least 2 prior txns for a std; otherwise NaN
            stds[idx] = prior.std() if len(prior) >= 2 else np.nan

    return means, stds


def _evaluate_factors(row, device_share, merchant_share):
    triggered = []

    # --- Factor 1: Unusual amount (z > 3 vs. PRIOR baseline) ---
    if row["acc_std"] and not np.isnan(row["acc_std"]) and row["acc_std"] > 0:
        z = (row["amount"] - row["acc_mean"]) / row["acc_std"]
        if z > AMOUNT_ZSCORE_THRESHOLD:
            triggered.append((
                WEIGHTS["unusual_amount"],
                f"Unusual amount (₹{row['amount']:.0f}, {z:.1f}σ above prior avg ₹{row['acc_mean']:.0f})"
            ))

    # --- Factor 2: New device ---
    if row["device_is_new"] and row["prior_txn_count"] >= MIN_PRIOR_TXNS_FOR_NOVELTY:
        triggered.append((
            WEIGHTS["new_device"],
            f"New device ({row['device']}) never seen for this account"
        ))

    # --- Factor 3: New merchant ---
    if row["merchant_is_new"] and row["prior_txn_count"] >= MIN_PRIOR_TXNS_FOR_NOVELTY:
        triggered.append((
            WEIGHTS["new_merchant"],
            f"New merchant ({row['merchant']})"
        ))

    # --- Factor 4: Shared device ---
    if device_share >= SHARED_DEVICE_MIN_ACCOUNTS:
        triggered.append((
            WEIGHTS["shared_device"],
            f"Device {row['device']} shared by {device_share} accounts"
        ))

    # --- Factor 5: Shared merchant (unusual) ---
    if SHARED_MERCHANT_MIN_ACCOUNTS <= merchant_share <= SHARED_MERCHANT_MAX_ACCOUNTS:
        triggered.append((
            WEIGHTS["shared_merchant"],
            f"Merchant {row['merchant']} shared by {merchant_share} accounts"
        ))

    return triggered


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account", "timestamp"]).reset_index(drop=True)

    # --- Per-account chronological index ---
    df["prior_txn_count"] = df.groupby("account").cumcount()

    # --- Prior-only baselines (avoids look-ahead bias) ---
    means, stds = _compute_prior_baselines(df)
    df["acc_mean"] = means
    df["acc_std"] = stds

    # --- Per-account seen-flags ---
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
    scores, decisions, reasons, evidence_counts = [], [], [], []

    for _, row in df.iterrows():
        triggered = _evaluate_factors(
            row,
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
        evidence_counts.append(len(triggered))

    df["risk_score"] = scores
    df["decision"] = decisions
    df["reasons"] = reasons
    df["evidence_count"] = evidence_counts

    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    scored = score_transactions(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_PATH, index=False)

    print(f"✓ Scored {len(scored)} transactions")
    print(f"  Output: {OUTPUT_PATH}")
    print()

    print("Decision breakdown:")
    print(scored["decision"].value_counts().to_string())
    print()

    ring = scored[scored["device"] == "D05"]
    print(f"Ring transactions (D05): {len(ring)}")
    print(f"  Avg risk score: {ring['risk_score'].mean():.1f}")
    print(f"  Min risk score: {ring['risk_score'].min()}")
    print(f"  Max risk score: {ring['risk_score'].max()}")
    print()

    blocked = scored[scored["decision"] == "BLOCK"]
    false_blocks = blocked[blocked["device"] != "D05"]
    print(f"BLOCK total: {len(blocked)} | Of which non-ring: {len(false_blocks)}")
    print()

    print("Top 15 highest-risk transactions:")
    top = scored.nlargest(15, "risk_score")[
        ["txn_id", "account", "device", "merchant", "amount", "risk_score", "decision"]
    ]
    print(top.to_string(index=False))

    print()
    print("Sample ring transaction explanations:")
    ring_top = ring.nlargest(3, "risk_score")
    for _, r in ring_top.iterrows():
        print(f"\n  {r['txn_id']} (account {r['account']}, score {r['risk_score']}):")
        for reason in r["reasons"]:
            print(f"    {reason}")


if __name__ == "__main__":
    main()