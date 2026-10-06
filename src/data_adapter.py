"""
data_adapter.py

Loads the Indian Financial Fraud Dataset (Kaggle) and produces a single
normalized transaction table used by the rest of the pipeline.

Dataset source: https://www.kaggle.com/datasets/jatinkhandelwal112/indian-financial-fraud-dataset
Files (in data/raw/):
    - Transaction_Data_250k.csv   (250,000 rows)
    - Cards_Data.csv              (32,457 rows)
    - Cusmtomer_data.csv          (25,000 rows)  <- typo is from the dataset itself
    - merchant_table.csv          (500 rows)

Output: data/transactions_clean.csv
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUTPUT = Path("data/transactions_clean.csv")

# --- Canonical name mapping ---
RENAME = {
    "Transaction_ID": "txn_id",
    "Customer_ID": "account",
    "Card_ID": "device",
    "Merchant_ID": "merchant",
    "Transaction_Amount": "amount",
    "Customer_State": "location",
}

# --- Columns we keep from each source table ---
CARD_COLS = [
    "Card_ID", "Card_Type", "Card_Network", "Credit_Limit",
    "Card_Status", "Contactless", "Card_Mode", "Issue_Date", "Expiry_Date",
]
CUSTOMER_COLS = [
    "Customer_ID", "Age", "Annual_Income", "Customer_Segment",
    "Account_Type", "Gender", "Occupation",
]
MERCHANT_COLS = [
    "Merchant_ID", "Merchant_Category", "Merchant_Risk_Level",
    "Merchant_Rating", "Merchant_Status",
]


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Transaction_Data_250k.csv")

    # Combine date + time into one datetime column
    df["timestamp"] = pd.to_datetime(
        df["Transaction_Date"].astype(str) + " " + df["Transaction_Time"].astype(str),
        errors="coerce",
    )

    # Rename to canonical
    df = df.rename(columns=RENAME)

    return df


def load_cards() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Cards_Data.csv")
    return df[CARD_COLS].copy()


def load_customers() -> pd.DataFrame:
    # Note: the filename has a typo ("Cusmtomer") from the Kaggle source.
    df = pd.read_csv(RAW / "Cusmtomer_data.csv")
    return df[CUSTOMER_COLS].copy()


def load_merchants() -> pd.DataFrame:
    df = pd.read_csv(RAW / "merchant_table.csv")
    return df[MERCHANT_COLS].copy()


def build_clean_transactions() -> pd.DataFrame:
    txns = load_transactions()
    cards = load_cards()
    customers = load_customers()
    merchants = load_merchants()

    # --- Drop merchant columns from transactions that we'll re-add from merchant_table ---
    # These exist in both sources; keep only the merchant_table version (authoritative).
    txns = txns.drop(columns=["Merchant_Risk_Level", "Merchant_Category"])

    # --- Merge card attributes onto transactions ---
    cards_renamed = cards.rename(columns={"Card_ID": "device"})
    txns = txns.merge(cards_renamed, on="device", how="left")

    # --- Merge customer attributes ---
    customers_renamed = customers.rename(columns={"Customer_ID": "account"})
    txns = txns.merge(customers_renamed, on="account", how="left")

    # --- Merge merchant attributes ---
    merchants_renamed = merchants.rename(columns={"Merchant_ID": "merchant"})
    txns = txns.merge(merchants_renamed, on="merchant", how="left")

    # --- Sort chronologically per account ---
    txns = txns.sort_values(["account", "timestamp"]).reset_index(drop=True)

    return txns


def main():
    df = build_clean_transactions()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"✓ Built clean transaction table: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Output: {OUTPUT}")
    print()
    print("Column list:")
    print(df.columns.tolist())
    print()
    print("Sample row:")
    print(df.iloc[0].to_string())
    print()
    print("Sanity checks:")
    print(f"  Unique accounts: {df['account'].nunique()}")
    print(f"  Unique devices:  {df['device'].nunique()}")
    print(f"  Unique merchants:{df['merchant'].nunique()}")
    print(f"  Null timestamps: {df['timestamp'].isna().sum()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Fraud rate: {df['Fraud_Flag'].mean()*100:.2f}%")
    print()
    print("Card status breakdown:")
    print(df["Card_Status"].value_counts().to_string())


if __name__ == "__main__":
    main()