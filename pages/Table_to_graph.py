import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ucimlrepo import fetch_ucirepo
import os
from urllib.parse import urlparse

st.set_page_config(layout="wide")

# ------------------- Helper Functions --------------------

def minmax_scale(df):
    num = df.select_dtypes(include=[np.number])
    rng = num.max() - num.min()
    return ((num - num.min()) / rng.replace(0, np.nan)).fillna(0.0)

def _v(i):
    return fr"$v_{{{i + 1}}}$"

def build_corr_graph(corr, threshold=0.05):
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    ii, jj = np.where((np.abs(corr.values) > threshold) & mask)
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    for i, j in zip(ii, jj):
        G.add_edge(corr.index[i], corr.columns[j], weight=corr.iat[i, j])
    return G



def _pmf(series, bins_edges):
    counts, _ = np.histogram(series, bins=bins_edges)
    return counts / counts.sum() if counts.sum() else np.zeros_like(counts)

def tvd_histogram_graph(X_scaled, y, G, a, b, bins=20):
    edges = np.linspace(0.0, 1.0, bins + 1)
    mask_a, mask_b = y == a, y == b
    d_v = sum(0.5 * np.abs(_pmf(X_scaled.loc[mask_a, n], edges) - _pmf(X_scaled.loc[mask_b, n], edges)).sum() for n in G.nodes)
    d_e = 0.0
    for i, j in G.edges:
        diff_a = 0.5 * np.abs(_pmf(X_scaled.loc[mask_a, i], edges) - _pmf(X_scaled.loc[mask_a, j], edges)).sum()
        diff_b = 0.5 * np.abs(_pmf(X_scaled.loc[mask_b, i], edges) - _pmf(X_scaled.loc[mask_b, j], edges)).sum()
        d_e += abs(diff_a - diff_b)
    return d_v, d_e

def tvd_label_vs_all(X_scaled, y, G, label, bins=20):
    edges = np.linspace(0.0, 1.0, bins + 1)
    mask_label = y == label
    mask_all = pd.Series(True, index=y.index)
    d_v = sum(0.5 * np.abs(_pmf(X_scaled.loc[mask_label, n], edges) - _pmf(X_scaled[n], edges)).sum() for n in G.nodes)
    d_e = 0.0
    for i, j in G.edges:
        diff_label = 0.5 * np.abs(_pmf(X_scaled.loc[mask_label, i], edges) - _pmf(X_scaled.loc[mask_label, j], edges)).sum()
        diff_all = 0.5 * np.abs(_pmf(X_scaled[i], edges) - _pmf(X_scaled[j], edges)).sum()
        d_e += abs(diff_label - diff_all)
    return d_v, d_e

def draw_corr_graph(G, name_map):
    pos = nx.spring_layout(G, seed=42)
    edge_colors = ["red" if G[u][v]["weight"] < 0 else "gray" for u, v in G.edges]
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw(G, pos, with_labels=False, node_color="#c6dbef", edge_color=edge_colors, node_size=1300, ax=ax)
    for node, (x, y) in pos.items():
        ax.text(x, y, name_map[node], fontsize=11, ha="center", va="center")
    ax.set_title("Feature Correlation Graph")
    ax.axis("off")

    # Add legend mapping $v_i$ to feature names
    label_box = [f"{name_map[col]}: {col}" for col in G.nodes if col in name_map]
    plt.figtext(1.01, 0.5, '\n'.join(label_box), va='center', fontsize=8)

    return fig

def plot_dv_vs_de(dv, de, labels=None, title="TVD: $d_v$ vs $d_e$"):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    custom_colors = ["purple", "green", "blue"]
    for i, (x, y_) in enumerate(zip(dv, de)):
        color = custom_colors[i % len(custom_colors)]
        ax.scatter(x, y_, s=100, color=color, label=labels[i] if labels else f"Point {i+1}")
    max_val = 1.05 * max(max(dv), max(de))
    ax.plot([0, max_val], [0, max_val], "r--", lw=1)
    ax.set_xlabel("Node distance $\\theta_V$")
    ax.set_ylabel("Edge distance $\\theta_E$")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    return fig

def plot_histograms_single_class(X_scaled, y, label, max_feats=9, bins="auto"):
    import matplotlib.colors as mcolors
    pastel_colors = list(mcolors.TABLEAU_COLORS.values())[:9]
    num_cols = X_scaled.columns[:max_feats]
    fig, axes = plt.subplots(3, 3, figsize=(9, 7.5))
    mask = y.squeeze() == label
    for idx, feat in enumerate(num_cols):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        data = X_scaled.loc[mask, feat].dropna()
        cnt, bin_edges = np.histogram(data, bins=bins)
        probs = cnt / cnt.sum() if cnt.sum() else np.zeros_like(cnt)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(centers, probs, width=bin_edges[1]-bin_edges[0], alpha=0.85,
               edgecolor="gray", linewidth=0.5, color=pastel_colors[idx % len(pastel_colors)], zorder=3)
        ax.set_title(fr"$v_{{{idx+1}}}$", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    for idx in range(len(num_cols), 9):
        axes.flatten()[idx].axis("off")
    plt.suptitle(fr"Feature Distributions – label '{label}'", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

# ------------------- Dataset Loading Logic --------------------
@st.cache_data(show_spinner=False)
def load_selected_dataset(uid):
    ds = fetch_ucirepo(id=uid)
    X_full, y_full = ds.data.features, ds.data.targets
    if uid == 211:
        cols = ["pctBlack", "pctPoverty", "pctNotHSgrad", "pctUnemploy", "pctAllDivorc",
                "pctKidsBornNevrMarr", "pctHousWOphone", "pctVacantBoarded", "pctPolicBlack"]
        X = X_full[cols].copy()
        murders = y_full["murders"]
        bins = [-np.inf, murders.quantile(0.33), murders.quantile(0.66), np.inf]
        y_cat = pd.cut(murders, bins, labels=["Low", "Medium", "High"])
        return X, y_cat.to_frame(name="Crime Level"), ds.metadata['name']
    if uid == 105:
        vote_map = {'y': 1, 'n': 0, '?': np.nan}
        X_votes = X_full.replace(vote_map).astype(float).dropna()
        y_clean = y_full.loc[X_votes.index].squeeze()
        return X_votes, y_clean.to_frame(name="Party"), ds.metadata['name']
    return X_full, y_full, ds.metadata['name']

# ------------------- Main App Execution --------------------

st.title("📊 UCI Dataset Correlation Graph Explorer (v2.3)")
option = st.selectbox("Choose dataset", [
    "Adult Income (ID=2)",
    "Communities and Crime Unnormalized (ID=211)",
    "Congressional Voting Records (ID=105)"
])
uci_id = {"Adult Income (ID=2)": 2, "Communities and Crime Unnormalized (ID=211)": 211, "Congressional Voting Records (ID=105)": 105}[option]
X, y, dataset_name = load_selected_dataset(uci_id)

corr_method = st.radio("Correlation Method", ["pearson", "spearman"])
threshold = st.slider("Correlation Threshold", 0.01, 1.0, 0.05, 0.01)
bins = st.slider("TVD Histogram Bins", 5, 50, 20, 1)

if X is not None and y is not None:
    st.success(f"Loaded: {dataset_name}")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 📌 Data Preview")
        st.dataframe(pd.concat([X, y], axis=1).head())
    with col2:
        st.markdown("#### 🔢 Target Summary")
        st.dataframe(y.value_counts().rename("count"))

    X_num = X.select_dtypes(include=[np.number])
    X_scaled = minmax_scale(X_num)
    corr = X_scaled.corr(method=corr_method)
    G = build_corr_graph(corr, threshold)
    name_map = {col: _v(i) for i, col in enumerate(corr.columns)}

    col_heat, col_graph = st.columns(2)
    with col_heat:
        st.subheader("📊 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        short_lbls = [name_map[c] for c in corr.columns]
        ax.set_xticklabels(short_lbls, rotation=90)
        ax.set_yticklabels(short_lbls)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr.iloc[i,j]) > 0.5 else "black", fontsize=7)
        fig.colorbar(cax)
        st.pyplot(fig)

    with col_graph:
        st.subheader("🌐 Correlation Graph")
        st.pyplot(draw_corr_graph(G, name_map))

    st.subheader("🌟 Feature Distributions (One Class)")
    chosen_label = st.selectbox("Select a class label", y.squeeze().unique().tolist())
    plot_histograms_single_class(X_scaled, y, label=chosen_label)

    labels = y.squeeze().unique()
    if len(labels) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 TVD Between Label Pairs")
            dv_list, de_list, label_list = [], [], []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    dv, de = tvd_histogram_graph(X_scaled, y.squeeze(), G, labels[i], labels[j], bins)
                    dv_list.append(dv)
                    de_list.append(de)
                    label_list.append(f"{labels[i]} vs {labels[j]}")
            fig1 = plot_dv_vs_de(dv_list, de_list, labels=label_list, title="Inter Class Distance")
            st.pyplot(fig1)

        with col2:
            st.subheader("🔄 TVD: Each Label vs ALL")
            dv_all, de_all, tag_all = [], [], []
            for i, lab in enumerate(labels):
                dv_i, de_i = tvd_label_vs_all(X_scaled, y.squeeze(), G, lab, bins)
                dv_all.append(dv_i)
                de_all.append(de_i)
                tag_all.append(str(lab))
            fig2 = plot_dv_vs_de(dv_all, de_all, labels=tag_all, title="Each Label vs ALL")
            st.pyplot(fig2)
else:
    st.warning("Please select or load a valid dataset to begin.")
