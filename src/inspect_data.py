"""
inspect_fraud.py — check whether the dataset contains coordinated fraud patterns.
"""
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/Transaction_Data_250k.csv")

print("=" * 70)
print("FRAUD OVERVIEW")
print("=" * 70)
print(f"Total transactions: {len(df)}")
print(f"Fraud transactions:  {df['Fraud_Flag'].sum()} ({df['Fraud_Flag'].mean()*100:.2f}%)")
print()

print("=" * 70)
print("FRAUD_REASON DISTRIBUTION (top 15)")
print("=" * 70)
print(df[df['Fraud_Flag'] == 1]['Fraud_Reason'].value_counts().head(15).to_string())
print()

print("=" * 70)
print("RING SIGNAL 1: Card shared by multiple customers")
print("=" * 70)
card_customers = df.groupby("Card_ID")["Customer_ID"].nunique()
multi = card_customers[card_customers > 1]
print(f"Cards used by >1 customer: {len(multi)}")
if len(multi) > 0:
    print("Sample:")
    print(multi.head(10).to_string())
print()

print("=" * 70)
print("RING SIGNAL 2: Merchants with unusually high fraud rate")
print("=" * 70)
merchant_stats = df.groupby("Merchant_ID").agg(
    txn_count=("Transaction_ID", "count"),
    fraud_count=("Fraud_Flag", "sum"),
).reset_index()
merchant_stats["fraud_rate"] = merchant_stats["fraud_count"] / merchant_stats["txn_count"]
merchant_stats = merchant_stats[merchant_stats["txn_count"] >= 20]  # filter noise
print("Top 15 merchants by fraud rate (min 20 txns):")
print(merchant_stats.sort_values("fraud_rate", ascending=False).head(15).to_string(index=False))
print()

print("=" * 70)
print("RING SIGNAL 3: Customers with multiple fraud transactions")
print("=" * 70)
cust_stats = df.groupby("Customer_ID").agg(
    txn_count=("Transaction_ID", "count"),
    fraud_count=("Fraud_Flag", "sum"),
).reset_index()
cust_stats["fraud_rate"] = cust_stats["fraud_count"] / cust_stats["txn_count"]
high_risk = cust_stats[cust_stats["fraud_count"] >= 3].sort_values("fraud_count", ascending=False)
print(f"Customers with >=3 fraud txns: {len(high_risk)}")
print(high_risk.head(15).to_string(index=False))
print()

print("=" * 70)
print("RING SIGNAL 4: Card status distribution (expired/blocked/lost cards)")
print("=" * 70)
cards = pd.read_csv("data/raw/Cards_Data.csv")
print(cards["Card_Status"].value_counts().to_string())
print()

print("=" * 70)
print("RING SIGNAL 5: Merchant risk level")
print("=" * 70)
print(df["Merchant_Risk_Level"].value_counts().to_string())
print()
print("Fraud rate by merchant risk level:")
print(df.groupby("Merchant_Risk_Level")["Fraud_Flag"].mean().to_string())