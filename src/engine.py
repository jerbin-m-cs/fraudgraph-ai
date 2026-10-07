"""
engine.py — Pass-1 fraud detection with baseline contamination guard and velocity factor.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/transactions_with_ring.csv")
OUTPUT_PATH = Path("data/transactions_scored_pass1.csv")

WEIGHTS = {
    "unusual_amount": 25,
    "new_device": 20,
    "new_merchant": 15,
    "shared_device": 25,
    "shared_merchant": 15,
    "card_status_violation": 25,
    "international": 15,
    "non_successful_status": 20,
    "high_risk_merchant": 10,
    "off_hours": 10,
    "rapid_velocity": 20,
    "ring_signature": 50,
}


AMOUNT_ZSCORE_THRESHOLD = 2.0
MIN_PRIOR_TXNS_FOR_NOVELTY = 5
SHARED_DEVICE_MIN_ACCOUNTS = 4
SHARED_MERCHANT_MIN_ACCOUNTS = 3
SHARED_MERCHANT_MAX_ACCOUNTS = 10
OFF_HOURS_START = 6
OFF_HOURS_END = 22
VELOCITY_WINDOW_MINUTES = 60
VELOCITY_MIN_TXNS = 3

# Baseline contamination guard: a transaction whose amount exceeds
# CONTAMINATION_MULTIPLIER * median prior amount is excluded from future baselines.
CONTAMINATION_MULTIPLIER = 3.0

BLOCK_THRESHOLD = 70
REVIEW_THRESHOLD = 40


def _compute_prior_baselines(df):
    """
    Per-account rolling baseline of amounts using ONLY prior transactions.
    Transactions flagged as anomalous (amount > CONTAMINATION_MULTIPLIER * prior median)
    are excluded from subsequent baselines — otherwise the ring contaminates
    its own baseline and later ring transactions look "normal."
    """
    means = np.full(len(df), np.nan)
    stds = np.full(len(df), np.nan)
    df = df.reset_index(drop=True)

    for acc, group in df.groupby("account", sort=False):
        idxs = group.index.values
        amounts = group["amount"].values
        clean_history = []  # amounts we consider "normal" for this account

        for i, idx in enumerate(idxs):
            if clean_history:
                arr = np.array(clean_history)
                means[idx] = arr.mean()
                stds[idx] = arr.std() if len(arr) >= 2 else np.nan

                # Decide if this amount is anomalous vs. the clean history
                prior_median = np.median(arr)
                is_contaminated = (
                    prior_median > 0 and
                    amounts[i] > CONTAMINATION_MULTIPLIER * prior_median
                )
                if not is_contaminated:
                    clean_history.append(amounts[i])
            else:
                clean_history.append(amounts[i])

    return means, stds


def _compute_velocity(df):
    """For each transaction, count how many of the same account's prior
    transactions occurred within the last VELOCITY_WINDOW_MINUTES."""
    df = df.reset_index(drop=True)
    velocity = np.zeros(len(df), dtype=int)

    for acc, group in df.groupby("account", sort=False):
        idxs = group.index.values
        times = group["timestamp"].values
        for i in range(len(idxs)):
            if i == 0:
                continue
            window_start = times[i] - np.timedelta64(VELOCITY_WINDOW_MINUTES, "m")
            count = sum(1 for j in range(i) if times[j] >= window_start)
            velocity[idxs[i]] = count

    return velocity


def _evaluate_factors(row, device_share, merchant_share):
    triggered = []

    # Factor 1
    if row["acc_std"] and not np.isnan(row["acc_std"]) and row["acc_std"] > 0:
        z = (row["amount"] - row["acc_mean"]) / row["acc_std"]
        if z > AMOUNT_ZSCORE_THRESHOLD:
            triggered.append((
                WEIGHTS["unusual_amount"],
                f"Unusual amount (₹{row['amount']:.0f}, {z:.1f}σ above prior avg ₹{row['acc_mean']:.0f})"
            ))

    # Factor 2
    if row["device_is_new"] and row["prior_txn_count"] >= MIN_PRIOR_TXNS_FOR_NOVELTY:
        triggered.append((WEIGHTS["new_device"], f"New card ({row['device']})"))

    # Factor 3
    if row["merchant_is_new"] and row["prior_txn_count"] >= MIN_PRIOR_TXNS_FOR_NOVELTY:
        triggered.append((WEIGHTS["new_merchant"], f"New merchant ({row['merchant']})"))

    # Factor 4
    if device_share >= SHARED_DEVICE_MIN_ACCOUNTS:
        triggered.append((WEIGHTS["shared_device"], f"Card shared by {device_share} accounts"))

    # Factor 5
    if SHARED_MERCHANT_MIN_ACCOUNTS <= merchant_share <= SHARED_MERCHANT_MAX_ACCOUNTS:
        triggered.append((WEIGHTS["shared_merchant"], f"Merchant shared by {merchant_share} accounts"))

    # Factor 6
    card_status = row.get("Card_Status")
    if card_status in ("Expired", "Blocked", "Lost"):
        triggered.append((WEIGHTS["card_status_violation"], f"Card status: {card_status}"))

    # Factor 7
    if row.get("Is_International") == 1:
        triggered.append((WEIGHTS["international"], "International transaction"))

    # Factor 8
    status = row.get("Transaction_Status")
    if status and status != "Successful":
        triggered.append((WEIGHTS["non_successful_status"], f"Status: {status}"))

    # Factor 9
    if row.get("Merchant_Risk_Level") == "High":
        triggered.append((WEIGHTS["high_risk_merchant"], "High-risk merchant"))

    # Factor 10
    hour = row["timestamp"].hour
    if hour < OFF_HOURS_START or hour >= OFF_HOURS_END:
        triggered.append((WEIGHTS["off_hours"], f"Off-hours ({hour:02d}:xx)"))

        # Factor 11: Velocity
    if row["velocity"] >= VELOCITY_MIN_TXNS:
        triggered.append((WEIGHTS["rapid_velocity"], f"Velocity: {row['velocity']} txns in {VELOCITY_WINDOW_MINUTES}min"))

        # Factor 12: Ring signature
    # Fires whenever the transaction uses a card that is shared across
    # >=4 accounts AND the account has some history (avoid cold-start).
    # Any card shared across many accounts is a coordination signal —
    # every transaction on such a card is suspect until proven otherwise.
    if device_share >= SHARED_DEVICE_MIN_ACCOUNTS and row["prior_txn_count"] >= 2:
        triggered.append((
            WEIGHTS["ring_signature"],
            f"Ring signature: card shared by {device_share} accounts"
        ))

    return triggered


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account", "timestamp"]).reset_index(drop=True)

    df["prior_txn_count"] = df.groupby("account").cumcount()

    means, stds = _compute_prior_baselines(df)
    df["acc_mean"] = means
    df["acc_std"] = stds

    df["velocity"] = _compute_velocity(df)

    df["device_is_new"] = df.groupby("account")["device"].transform(
        lambda s: ~s.duplicated(keep="first")
    )
    df["merchant_is_new"] = df.groupby("account")["merchant"].transform(
        lambda s: ~s.duplicated(keep="first")
    )

    device_share = df.groupby("device")["account"].nunique().to_dict()
    merchant_share = df.groupby("merchant")["account"].nunique().to_dict()

    scores, decisions, reasons, evidence_counts, raw_scores = [], [], [], [], []

    for _, row in df.iterrows():
        triggered = _evaluate_factors(
            row,
            device_share.get(row["device"], 1),
            merchant_share.get(row["merchant"], 1),
        )
        raw = sum(p for p, _ in triggered)
        clamped = min(raw, 100)

        if clamped >= BLOCK_THRESHOLD:
            decision = "BLOCK"
        elif clamped >= REVIEW_THRESHOLD:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        raw_scores.append(raw)
        scores.append(clamped)
        decisions.append(decision)
        reasons.append([f"+{p} {txt}" for p, txt in triggered])
        evidence_counts.append(len(triggered))

    df["raw_score"] = raw_scores
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
    print()
    print("Decision breakdown:")
    print(scored["decision"].value_counts().to_string())
    print()

    ring = scored[scored["device"] == "CARD_INJECTED_01"]
    print(f"Injected ring transactions: {len(ring)}")
    print(f"  Avg risk score: {ring['risk_score'].mean():.1f}")
    print(f"  Min: {ring['risk_score'].min()} | Max: {ring['risk_score'].max()}")
    print(f"  Blocked: {(ring['decision'] == 'BLOCK').sum()} / {len(ring)}")
    print()

    real = scored[scored["Fraud_Reason"] != "Coordinated Ring Activity (injected)"]
    real_fraud = real[real["Fraud_Flag"] == 1]

    flagged = real[real["decision"] == "BLOCK"]
    tp = flagged[flagged["Fraud_Flag"] == 1].shape[0]
    fp = flagged[flagged["Fraud_Flag"] == 0].shape[0]
    fn = real_fraud.shape[0] - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("Real fraud detection (excluding ring):")
    print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print()

    print("Detection by fraud reason:")
    for reason in sorted(real_fraud["Fraud_Reason"].dropna().unique()):
        sub = real_fraud[real_fraud["Fraud_Reason"] == reason]
        caught = (sub["decision"] == "BLOCK").sum()
        print(f"  {reason[:48]:48s} {caught:5d}/{len(sub):5d}  ({caught/len(sub)*100:5.1f}%)")


if __name__ == "__main__":
    main()