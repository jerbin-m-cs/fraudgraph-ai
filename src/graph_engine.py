"""
graph_engine.py

Builds a NetworkX graph of high-risk transactions and detects coordinated
fraud rings via community detection on an account-account projection.

Ring detection approach:
  1. Filter to high-risk transactions (risk_score >= GRAPH_THRESHOLD) plus
     every transaction on the injected ring card.
  2. Build a bipartite graph: account ↔ card/merchant/location.
  3. Project onto accounts: two accounts connected if they share
     a card, merchant, or location.
  4. Run community detection on the projection.
  5. A community with >= 3 accounts that share >= 1 card is a candidate ring.

Outputs:
  - data/graph_nodes.csv
  - data/graph_edges.csv
  - data/rings_detected.csv
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from networkx.algorithms.community import greedy_modularity_communities

INPUT = Path("data/transactions_scored_pass1.csv")
OUT_NODES = Path("data/graph_nodes.csv")
OUT_EDGES = Path("data/graph_edges.csv")
OUT_RINGS = Path("data/rings_detected.csv")

GRAPH_THRESHOLD = 60
RING_CARD = "CARD_INJECTED_01"


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a node-attribute graph of high-risk transactions."""
    mask = (df["risk_score"] >= GRAPH_THRESHOLD) | (df["device"] == RING_CARD)
    suspicious = df[mask].copy()

    print(f"Building graph from {len(suspicious)} transactions "
          f"({len(suspicious)/len(df)*100:.2f}% of all)")

    G = nx.Graph()

    # --- Account nodes ---
    for acc, group in suspicious.groupby("account"):
        G.add_node(
            f"A_{acc}", node_type="account", label=acc,
            avg_risk=round(group["risk_score"].mean(), 1),
            max_risk=int(group["risk_score"].max()),
            txn_count=int(len(group)),
        )

    # --- Card nodes ---
    for card, group in suspicious.groupby("device"):
        accounts_using = group["account"].nunique()
        G.add_node(
            f"C_{card}", node_type="card", label=card,
            accounts_using=int(accounts_using),
            is_shared=bool(accounts_using >= 2),
        )

    # --- Merchant nodes ---
    for merchant, group in suspicious.groupby("merchant"):
        risk = group["Merchant_Risk_Level"].dropna().unique().tolist()
        G.add_node(
            f"M_{merchant}", node_type="merchant", label=merchant,
            risk_level=risk[0] if risk else "Unknown",
            accounts_using=int(group["account"].nunique()),
        )

    # --- Location nodes ---
    for loc, group in suspicious.groupby("location"):
        if pd.isna(loc):
            continue
        G.add_node(
            f"L_{loc}", node_type="location", label=str(loc),
            accounts_using=int(group["account"].nunique()),
        )

    # --- Edges ---
    edge_set = set()
    for _, row in suspicious.iterrows():
        acc_n = f"A_{row['account']}"
        card_n = f"C_{row['device']}"
        merch_n = f"M_{row['merchant']}"
        loc_n = f"L_{row['location']}" if pd.notna(row["location"]) else None

        for target, etype in [(card_n, "uses_card"), (merch_n, "shops_at")]:
            e = tuple(sorted([acc_n, target]))
            if e not in edge_set:
                G.add_edge(acc_n, target, edge_type=etype)
                edge_set.add(e)

        if loc_n:
            e = tuple(sorted([acc_n, loc_n]))
            if e not in edge_set:
                G.add_edge(acc_n, loc_n, edge_type="located_in")
                edge_set.add(e)

    return G, suspicious


def project_to_accounts(G: nx.Graph) -> nx.Graph:
    """Create account-account graph where two accounts are connected if they
    share any card, merchant, or location node (via the bipartite structure)."""
    account_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "account"]
    G_proj = nx.Graph()
    G_proj.add_nodes_from(account_nodes)

    # For each non-account node, connect all accounts that touch it
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "account":
            continue
        neighbors = [n for n in G.neighbors(node) if n.startswith("A_")]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a, b = neighbors[i], neighbors[j]
                if G_proj.has_edge(a, b):
                    G_proj[a][b]["shared"] += 1
                else:
                    G_proj.add_edge(a, b, shared=1)

    return G_proj


def detect_rings(G: nx.Graph, min_accounts_per_card: int = 3) -> list:
    """
    A fraud ring = a card used by >= min_accounts_per_card accounts.
    Returns one ring per such card, sorted by size.
    """
    rings = []

    card_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "card"]
    for i, card_node in enumerate(card_nodes):
        # Find all account neighbors of this card
        accounts = [n for n in G.neighbors(card_node) if n.startswith("A_")]
        if len(accounts) < min_accounts_per_card:
            continue

        account_labels = [G.nodes[n]["label"] for n in accounts]
        card_label = G.nodes[card_node]["label"]

        # Find merchants this ring shares
        merchants = set()
        locations = set()
        for acc_node in accounts:
            for neighbor in G.neighbors(acc_node):
                if neighbor.startswith("M_"):
                    merchants.add(G.nodes[neighbor]["label"])
                elif neighbor.startswith("L_"):
                    locations.add(G.nodes[neighbor]["label"])

        rings.append({
            "ring_id": f"R{len(rings)+1}",
            "size": len(accounts),
            "shared_card": card_label,
            "accounts": sorted(account_labels),
            "shared_merchants_count": len(merchants),
            "shared_locations_count": len(locations),
        })

    return sorted(rings, key=lambda r: r["size"], reverse=True)


def main():
    df = pd.read_csv(INPUT)
    G, suspicious = build_graph(df)

    print(f"Bipartite graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    from collections import Counter
    node_types = Counter(d.get("node_type") for _, d in G.nodes(data=True))
    print("Node types:")
    for t, c in node_types.most_common():
        print(f"  {t}: {c}")
    print()

        # (No projection needed — we detect rings directly from shared cards)

    rings = detect_rings(G)
    print(f"Detected {len(rings)} candidate rings:")
    for r in rings[:10]:
        print(f"  {r['ring_id']}: {r['size']} accounts | shared card: {r['shared_card']}")
    print()

    # Save outputs
    nodes_df = pd.DataFrame([{"node_id": n, **d} for n, d in G.nodes(data=True)])
    edges_df = pd.DataFrame([{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)])
    rings_df = pd.DataFrame(rings)

    nodes_df.to_csv(OUT_NODES, index=False)
    edges_df.to_csv(OUT_EDGES, index=False)
    rings_df.to_csv(OUT_RINGS, index=False)

    print(f"✓ Saved:")
    print(f"  {OUT_NODES}  ({len(nodes_df)} nodes)")
    print(f"  {OUT_EDGES}  ({len(edges_df)} edges)")
    print(f"  {OUT_RINGS}  ({len(rings_df)} rings)")


if __name__ == "__main__":
    main()