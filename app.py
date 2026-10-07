"""
app.py — FraudGraph AI dashboard (Streamlit).

Tabs:
  Overview      — summary metrics and ring detection status
  Transactions  — top-scored transactions with reason breakdown
  Accounts      — accounts ranked by risk
  Network       — subgraph of the detected ring (Plotly, deterministic colors)
  Investigation — pick a transaction, see the full explanation and action
"""

import ast
import numpy as np
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# ------- Page config -------
st.set_page_config(
    page_title="FraudGraph AI",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="collapsed",
)

SCORED = "data/transactions_scored_pass1.csv"
RINGS = "data/rings_detected.csv"
RING_CARD = "CARD_INJECTED_01"

# ------- Custom CSS -------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

    h1 { font-size: 2rem !important; font-weight: 700 !important; margin-bottom: 0.1rem !important; }
    .subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; }

    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 4px;
        color: #6b7280;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #e5e7eb;
    }

    .chip {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.85rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    .badge-block { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display:inline-block;}
    .badge-review { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display:inline-block;}
    .badge-allow { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display:inline-block;}
    .badge-ok { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display:inline-block;}

    .info-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .info-row {
        display: flex;
        padding: 6px 0;
        border-bottom: 1px dashed #e5e7eb;
        font-size: 0.9rem;
    }
    .info-row:last-child { border-bottom: none; }
    .info-key { color: #6b7280; width: 180px; font-weight: 600; }
    .info-val { color: #111827; }
</style>
""", unsafe_allow_html=True)


# ------- Data -------
@st.cache_data
def load_data():
    df = pd.read_csv(SCORED)
    rings = pd.read_csv(RINGS)
    return df, rings


def safe_list(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def decision_badge(decision: str) -> str:
    cls = {"BLOCK": "badge-block", "REVIEW": "badge-review", "ALLOW": "badge-allow"}[decision]
    return f'<span class="{cls}">{decision}</span>'


df, rings = load_data()

# ------- Header -------
st.markdown("<h1>🔍 FraudGraph AI</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-Time Financial Fraud Intelligence · HackNex 2026 · PS04</div>',
    unsafe_allow_html=True,
)

# ------- Tabs -------
tab_overview, tab_txns, tab_accounts, tab_network, tab_investigate = st.tabs([
    "📊  Overview", "💳  Transactions", "👤  Accounts", "🕸️  Network", "🔎  Investigation"
])


# ================= OVERVIEW =================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-delta">250k real + 30 injected</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        blocked = (df["decision"] == "BLOCK").sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Flagged (BLOCK)</div>
            <div class="kpi-value">{blocked:,}</div>
            <div class="kpi-delta">{(blocked/len(df))*100:.2f}% of all txns</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        review = (df["decision"] == "REVIEW").sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Under Review</div>
            <div class="kpi-value">{review:,}</div>
            <div class="kpi-delta">{(review/len(df))*100:.2f}% of all txns</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud Rings</div>
            <div class="kpi-value">{len(rings)}</div>
            <div class="kpi-delta">coordinated groups detected</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎯 Injected Ring — Detection Status</div>', unsafe_allow_html=True)

    ring_txns = df[df["device"] == RING_CARD]
    if len(ring_txns) > 0:
        ring_blocked = (ring_txns["decision"] == "BLOCK").all()
        r1, r2, r3 = st.columns(3, gap="medium")
        with r1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Ring Transactions</div>
                <div class="kpi-value">{len(ring_txns)}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            status = '<span class="badge-ok">✅ 100% BLOCKED</span>' if ring_blocked else '<span class="badge-review">⚠️ Partial</span>'
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Detection</div>
                <div style="margin-top:6px">{status}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg Risk Score</div>
                <div class="kpi-value">{ring_txns['risk_score'].mean():.1f}<span style="font-size:1rem;color:#6b7280;"> / 100</span></div>
            </div>""", unsafe_allow_html=True)

        if len(rings) > 0:
            accs = safe_list(rings.iloc[0]["accounts"])
            chips = "".join(f'<span class="chip">{a}</span>' for a in accs)
            st.markdown('<div class="section-header">Ring Members (identified by graph analysis)</div>', unsafe_allow_html=True)
            st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
    else:
        st.warning("No injected ring transactions found.")

    st.markdown('<div class="section-header">📋 Detection Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <div class="info-row"><div class="info-key">Engine</div><div class="info-val">12-factor rule-based risk scoring (per-transaction)</div></div>
        <div class="info-row"><div class="info-key">Graph Layer</div><div class="info-val">Shared-card ring detection via NetworkX</div></div>
        <div class="info-row"><div class="info-key">Dataset</div><div class="info-val">Indian Financial Fraud — 250k transactions, 8 real fraud categories</div></div>
        <div class="info-row"><div class="info-key">Augmentation</div><div class="info-val">1 coordinated ring (5 accounts, 30 transactions) injected for demonstration</div></div>
        <div class="info-row"><div class="info-key">Ground Truth</div><div class="info-val"><code>Fraud_Flag</code> used for evaluation only — never used during scoring</div></div>
        <div class="info-row"><div class="info-key">Precision / Recall</div><div class="info-val">0.59 / 0.19 on real labeled frauds (precision prioritised)</div></div>
    </div>
    """, unsafe_allow_html=True)


# ================= TRANSACTIONS =================
with tab_txns:
    st.markdown('<div class="section-header">Top-Risk Transactions</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        decision_filter = st.multiselect(
            "Filter by decision",
            options=["BLOCK", "REVIEW", "ALLOW"],
            default=["BLOCK", "REVIEW"],
            label_visibility="collapsed",
        )
    with col_b:
        show_ring_only = st.checkbox("Ring only", value=False)

    view = df[df["decision"].isin(decision_filter)]
    if show_ring_only:
        view = view[view["device"] == RING_CARD]

    top = view.nlargest(200, "risk_score")[
        ["txn_id", "account", "device", "merchant", "amount",
         "risk_score", "decision", "Fraud_Flag"]
    ].rename(columns={"Fraud_Flag": "Label"})

    st.dataframe(
        top, use_container_width=True, height=520,
        column_config={
            "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.0f"),
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%d"
            ),
        }
    )
    st.caption(f"Showing top 200 of {len(view):,} matching transactions.")


# ================= ACCOUNTS =================
with tab_accounts:
    st.markdown('<div class="section-header">Accounts Ranked by Risk</div>', unsafe_allow_html=True)

    acc = df.groupby("account").agg(
        txn_count=("txn_id", "count"),
        avg_risk=("risk_score", "mean"),
        max_risk=("risk_score", "max"),
        total_amount=("amount", "sum"),
        blocked=("decision", lambda s: (s == "BLOCK").sum()),
    ).reset_index()

    acc["account_risk"] = (acc["avg_risk"] * 0.6 + acc["max_risk"] * 0.4).round(1)
    acc = acc.sort_values("account_risk", ascending=False)

    ring_accounts = safe_list(rings.iloc[0]["accounts"]) if len(rings) > 0 else []
    acc["in_ring"] = acc["account"].isin(ring_accounts)

    top = acc.head(100)[
        ["account", "txn_count", "avg_risk", "max_risk",
         "total_amount", "blocked", "account_risk", "in_ring"]
    ].rename(columns={
        "txn_count": "Txns",
        "avg_risk": "Avg Risk",
        "max_risk": "Max Risk",
        "total_amount": "Total Amount (₹)",
        "blocked": "Blocked",
        "account_risk": "Account Risk",
        "in_ring": "In Ring",
    })

    st.dataframe(
        top, use_container_width=True, height=520,
        column_config={
            "Avg Risk": st.column_config.NumberColumn(format="%.1f"),
            "Max Risk": st.column_config.NumberColumn(format="%d"),
            "Account Risk": st.column_config.ProgressColumn(
                "Account Risk", min_value=0, max_value=100, format="%.1f"
            ),
            "Total Amount (₹)": st.column_config.NumberColumn(format="₹%.0f"),
            "In Ring": st.column_config.CheckboxColumn(),
        }
    )
    st.caption(f"Showing top 100 of {len(acc):,} accounts · "
               f"{len(ring_accounts)} flagged as ring members")


# ================= NETWORK =================
with tab_network:
    st.markdown('<div class="section-header">Fraud Ring Network</div>', unsafe_allow_html=True)
    st.caption("Accounts (red) connected to the shared card (blue), merchant (green), and location (purple). "
               "Only the injected ring transactions are shown.")

    if len(rings) == 0:
        st.warning("No ring detected.")
    else:
        ring_accs = safe_list(rings.iloc[0]["accounts"])
        ring_df = df[(df["account"].isin(ring_accs)) & (df["device"] == RING_CARD)]

        # ---- Collect unique nodes with their types ----
        nodes = {}   # node_id -> {"label": ..., "type": ...}
        edges = []   # list of (source_id, target_id)

        for _, row in ring_df.iterrows():
            acc_id = f"A_{row['account']}"
            card_id = f"C_{row['device']}"
            merch_id = f"M_{row['merchant']}"

            nodes[acc_id] = {"label": row["account"], "type": "account"}
            nodes[card_id] = {"label": row["device"], "type": "card"}
            nodes[merch_id] = {"label": row["merchant"], "type": "merchant"}

            edges.append((acc_id, card_id))
            edges.append((acc_id, merch_id))

            if pd.notna(row["location"]):
                loc_id = f"L_{row['location']}"
                nodes[loc_id] = {"label": str(row["location"]), "type": "location"}
                edges.append((acc_id, loc_id))

        # ---- Layout: accounts on a circle, hub nodes near center ----
        account_ids = [nid for nid, d in nodes.items() if d["type"] == "account"]
        hub_ids = [nid for nid, d in nodes.items() if d["type"] != "account"]

        pos = {}
        n_acc = len(account_ids)
        for i, nid in enumerate(account_ids):
            angle = 2 * np.pi * i / n_acc
            pos[nid] = (np.cos(angle), np.sin(angle))
        for i, nid in enumerate(hub_ids):
            offset = (i - (len(hub_ids) - 1) / 2) * 0.18
            pos[nid] = (offset, offset * 0.4)

        # ---- Edge trace ----
        edge_x, edge_y = [], []
        for src, dst in edges:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#cbd5e1"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        # ---- Node traces (one per type for the legend) ----
        type_colors = {
            "account":  "#e74c3c",
            "card":     "#3498db",
            "merchant": "#2ecc71",
            "location": "#9b59b6",
        }
        type_labels = {
            "account":  "Account",
            "card":     "Shared Card",
            "merchant": "Merchant",
            "location": "Location",
        }
        type_sizes = {"account": 30, "card": 46, "merchant": 38, "location": 38}

        node_traces = []
        for ntype in ["account", "card", "merchant", "location"]:
            ids = [nid for nid, d in nodes.items() if d["type"] == ntype]
            if not ids:
                continue
            node_traces.append(go.Scatter(
                x=[pos[nid][0] for nid in ids],
                y=[pos[nid][1] for nid in ids],
                mode="markers+text",
                text=[nodes[nid]["label"] for nid in ids],
                textposition="bottom center",
                textfont=dict(size=13, color="#111827"),
                marker=dict(
                    size=type_sizes[ntype],
                    color=type_colors[ntype],
                    line=dict(width=2, color="#ffffff"),
                ),
                hoverinfo="text",
                hovertext=[f"{type_labels[ntype]}: {nodes[nid]['label']}" for nid in ids],
                name=type_labels[ntype],
                showlegend=True,
            ))

        fig = go.Figure(data=[edge_trace] + node_traces)
        fig.update_layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                bgcolor="rgba(255,255,255,0.9)",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Ring accounts:** {len(ring_accs)} &nbsp;·&nbsp; "
            f"**Shared card:** `{rings.iloc[0]['shared_card']}` &nbsp;·&nbsp; "
            f"**Relation:** all accounts transact on the same card within a 3-hour window"
        )


# ================= INVESTIGATION =================
with tab_investigate:
    st.markdown('<div class="section-header">Investigate a Transaction</div>', unsafe_allow_html=True)
    st.caption("Select a transaction to see the full risk breakdown and recommended action.")

    ring_txns = df[df["device"] == RING_CARD]
    default_id = ring_txns.iloc[0]["txn_id"] if len(ring_txns) > 0 else df.iloc[0]["txn_id"]

    pool = pd.concat([
        ring_txns,
        df[df["decision"] == "BLOCK"].head(500),
    ]).drop_duplicates("txn_id")

    txn_id = st.selectbox("Transaction", options=pool["txn_id"].tolist(), index=0)
    row = df[df["txn_id"] == txn_id].iloc[0]

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown(f"#### Transaction `{row['txn_id']}`")
        meta_cols = st.columns(3)
        meta_cols[0].markdown(f"**Account**<br>`{row['account']}`", unsafe_allow_html=True)
        meta_cols[1].markdown(f"**Card**<br>`{row['device']}`", unsafe_allow_html=True)
        meta_cols[2].markdown(f"**Merchant**<br>`{row['merchant']}`", unsafe_allow_html=True)

        meta_cols2 = st.columns(3)
        meta_cols2[0].markdown(f"**Amount**<br>₹{row['amount']:,.2f}", unsafe_allow_html=True)
        meta_cols2[1].markdown(f"**Location**<br>{row['location']}", unsafe_allow_html=True)
        meta_cols2[2].markdown(f"**Time**<br>{row['timestamp']}", unsafe_allow_html=True)

        st.markdown("##### Risk Factors")
        reasons = safe_list(row["reasons"])
        if reasons:
            for reason in reasons:
                st.markdown(f"- {reason}")
        else:
            st.info("No significant risk factors triggered for this transaction.")

    with right:
        st.markdown("##### Decision")
        decision = row["decision"]
        score = int(row["risk_score"])
        st.markdown(decision_badge(decision), unsafe_allow_html=True)
        st.markdown(f"**Risk Score: {score} / 100**")
        st.progress(score / 100)

        st.markdown("##### Recommended Action")
        if decision == "BLOCK":
            st.error("Freeze account · flag card for replacement · notify customer")
        elif decision == "REVIEW":
            st.warning("Step-up authentication · monitor next 3 txns · watchlist 24h")
        else:
            st.success("No action required · transaction cleared")