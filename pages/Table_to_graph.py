# 📄 pages/UCI Correlation Explorer.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ucimlrepo import fetch_ucirepo
from urllib.parse import urlparse
import os
from itertools import combinations

st.set_page_config(layout="wide")
st.title("📊 UCI Dataset Correlation Graph Explorer")

FALLBACK_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv"

# --------------- Data Loading -------------------
@st.cache_data(show_spinner=False)
def load_dataset(input_text):
    try:
        if input_text.isdigit():
            ds = fetch_ucirepo(id=int(input_text))
            X, y = ds.data.features, ds.data.targets
            return X, y, f"UCI Dataset: {ds.metadata['name']}"

        elif urlparse(input_text).scheme in ("http", "https"):
            df = pd.read_csv(input_text)
            df.columns = [f"col_{i}" for i in range(df.shape[1])] if df.columns.isnull().any() else df.columns
            return df.iloc[:, :-1], df.iloc[:, -1:], f"CSV Loaded from URL"

        elif os.path.exists(input_text):
            df = pd.read_csv(input_text)
            df.columns = [f"col_{i}" for i in range(df.shape[1])] if df.columns.isnull().any() else df.columns
            return df.iloc[:, :-1], df.iloc[:, -1:], f"CSV Loaded from File"

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
    return None, None, ""

# --------------- Correlation Graph -------------
def minmax_scale(df):
    num = df.select_dtypes(include=[np.number])
    rng = num.max() - num.min()
    scaled = (num - num.min()) / rng.replace(0, np.nan)
    return scaled.fillna(0.0)

def build_corr_graph(corr, threshold=0.05):
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    ii, jj = np.where((np.abs(corr.values) > threshold) & mask)
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    for i, j in zip(ii, jj):
        G.add_edge(corr.index[i], corr.columns[j], weight=corr.iat[i, j])
    return G

def draw_corr_graph(G):
    pos = nx.spring_layout(G, seed=42)
    edge_colors = ["red" if G[u][v]["weight"] < 0 else "blue" for u, v in G.edges]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color=edge_colors, node_size=1200, ax=ax)
    ax.set_title("Feature Correlation Graph")
    ax.axis("off")
    return fig

# --------------- TVD Calculation & Plotting ----------------
def tvd_histogram_graph(X_scaled, y, G, label_a, label_b, bins=20):
    edges = list(G.edges)
    bins_edges = np.linspace(0.0, 1.0, bins + 1)

    mask_a = y == label_a
    mask_b = y == label_b

    def pmf(series):
        counts, _ = np.histogram(series, bins=bins_edges)
        return counts / counts.sum() if counts.sum() else np.zeros_like(counts)

    d_v, d_e = 0.0, 0.0
    for node in G.nodes:
        p = pmf(X_scaled.loc[mask_a, node])
        q = pmf(X_scaled.loc[mask_b, node])
        d_v += 0.5 * np.abs(p - q).sum()

    for i, j in edges:
        smooth_a = 0.5 * np.abs(pmf(X_scaled.loc[mask_a, i]) - pmf(X_scaled.loc[mask_a, j])).sum()
        smooth_b = 0.5 * np.abs(pmf(X_scaled.loc[mask_b, i]) - pmf(X_scaled.loc[mask_b, j])).sum()
        d_e += abs(smooth_a - smooth_b)

    return d_v, d_e

def tvd_label_vs_all(X_scaled, y, G, label, bins=20):
    bins_edges = np.linspace(0.0, 1.0, bins + 1)
    mask_label = y == label

    def pmf(series):
        cnt, _ = np.histogram(series, bins=bins_edges)
        return cnt / cnt.sum() if cnt.sum() else np.zeros_like(cnt)

    d_v, d_e = 0.0, 0.0
    for node in G.nodes:
        p = pmf(X_scaled.loc[mask_label, node])
        q = pmf(X_scaled[node])
        d_v += 0.5 * np.abs(p - q).sum()

    for i, j in G.edges:
        smooth_lab = 0.5 * np.abs(pmf(X_scaled.loc[mask_label, i]) - pmf(X_scaled.loc[mask_label, j])).sum()
        smooth_all = 0.5 * np.abs(pmf(X_scaled[i]) - pmf(X_scaled[j])).sum()
        d_e += abs(smooth_lab - smooth_all)

    return d_v, d_e

def plot_dv_vs_de(dv, de, labels=None, title="TVD: d_v vs d_e"):
    dv = np.atleast_1d(dv).astype(float)
    de = np.atleast_1d(de).astype(float)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(dv, de, color="steelblue", s=80)
    max_val = 1.05 * max(dv.max(), de.max())
    ax.plot([0, max_val], [0, max_val], "r--", label="d_e = d_v")
    ax.set_xlabel("Node distance d_v")
    ax.set_ylabel("Edge distance d_e")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    if labels:
        for x, y, txt in zip(dv, de, labels):
            ax.text(x + 0.01 * max_val, y, txt, fontsize=8)

    return fig, list(zip(labels, dv, de)) if labels else []

def plot_group_histograms(X, y, max_groups=5, max_features=5, bins="auto"):
    y_series = y.squeeze()
    all_groups = y_series.value_counts().index.tolist()
    show_groups = all_groups[:max_groups]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    show_feats = num_cols[:max_features]

    n_rows, n_cols = len(show_groups), len(show_feats)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.5 * n_rows), squeeze=False)

    for r, label in enumerate(show_groups):
        mask = y_series == label
        for c, feat in enumerate(show_feats):
            ax = axes[r, c]
            data = X.loc[mask, feat].dropna()
            counts, bin_edges = np.histogram(data, bins=bins)
            probs = counts / counts.sum() if counts.sum() else np.zeros_like(counts)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width = (bin_edges[1] - bin_edges[0])
            ax.bar(bin_centers, probs, width=width, alpha=0.85, edgecolor="k")
            if r == 0: ax.set_title(feat, fontsize=9)
            if c == 0: ax.set_ylabel(f"label={label}", fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    st.pyplot(fig)

# -------------------- UI -----------------------
st.markdown("### Select Dataset")
dataset_option = st.selectbox("Choose from predefined UCI datasets", [
    "Adult Income (ID=2)",
    "Communities and Crime Unnormalized (ID=211)",
    "Congressional Voting Records (ID=105)"
])

uci_id_map = {
    "Adult Income (ID=2)": 2,
    "Communities and Crime Unnormalized (ID=211)": 211,
    "Congressional Voting Records (ID=105)": 105
}

@st.cache_data(show_spinner=False)
def load_selected_dataset(dataset_label):
    try:
        uci_id = uci_id_map[dataset_label]
        ds = fetch_ucirepo(id=uci_id)
        X_full = ds.data.features
        y_full = ds.data.targets

        if uci_id == 211:
            selected_features = [
                "pctBlack",
                "pctPoverty",
                "pctNotHSgrad",
                "pctUnemploy",
                "pctAllDivorc",
                "pctKidsBornNevrMarr",
                "pctHousWOphone",
                "pctVacantBoarded",
                "pctPolicBlack",
                
            ]

            missing = [f for f in selected_features if f not in X_full.columns]
            if missing:
                st.error(f"Missing columns in dataset: {missing}")
                return None, None, ""

            X = X_full[selected_features].copy()
            y_numeric = y_full["murders"].copy()

            bins = [-np.inf, y_numeric.quantile(0.33), y_numeric.quantile(0.66), np.inf]
            labels = ["Low", "Medium", "High"]
            y_categorical = pd.cut(y_numeric, bins=bins, labels=labels)

            return X, y_categorical.to_frame(name="Crime Level"), \
                   f"UCI Dataset: {ds.metadata['name']} (target: 'murders' → 3 classes)"

        elif uci_id == 105:
            # Step 1: Replace categorical values with numeric (y=1, n=0, ?=NaN)
            vote_map = {'y': 1, 'n': 0, '?': np.nan}
            X_votes = X_full.replace(vote_map).astype(float)

            # Step 2: Drop rows with any missing values
            valid_rows = X_votes.dropna()
            y_clean = y_full.loc[valid_rows.index].squeeze()

            return valid_rows, y_clean.to_frame(name="Party"), \
            f"UCI Dataset: {ds.metadata['name']} (votes encoded; rows with missing values dropped)"

        else:
            return ds.data.features, ds.data.targets, f"UCI Dataset: {ds.metadata['name']}"

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return None, None, ""


X, y, info = load_selected_dataset(dataset_option)
corr_method = st.radio("Correlation Method", ["pearson", "spearman"])
threshold = st.slider("Correlation Threshold", 0.01, 1.0, 0.05, 0.01)
bins = st.slider("TVD Histogram Bins", 5, 50, 20, 1)

if X is not None and y is not None:
    st.success(info)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 📌 Data Preview")
        st.dataframe(pd.concat([X, y], axis=1).head())

    with col2:
        st.markdown("#### 🔢 Target Summary")
        st.dataframe(y.value_counts().rename("count"))

    X_scaled = minmax_scale(X.select_dtypes(include=[np.number]))
    corr = X_scaled.corr(method=corr_method)

    st.markdown("---")
    col_heat, col_graph = st.columns(2)
    with col_heat:
        st.subheader("📊 Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
        fig_corr.colorbar(cax)
        st.pyplot(fig_corr)

    with col_graph:
        st.subheader("🌐 Correlation Graph")
        fig_graph = draw_corr_graph(build_corr_graph(corr, threshold))
        st.pyplot(fig_graph)

    st.subheader("🌟 Feature Distributions by Label")
    with st.expander("Show Group Histograms"):
        plot_group_histograms(X_scaled, y.squeeze())

    labels = y.squeeze().unique()
    if len(labels) >= 2:
        col_pair, col_all = st.columns(2)
        with col_pair:
            st.subheader("📊 TVD Between Label Pairs")
            dv_list, de_list, tag_list = [], [], []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    d_v, d_e = tvd_histogram_graph(X_scaled, y.squeeze(), build_corr_graph(corr, threshold), labels[i], labels[j], bins=bins)
                    dv_list.append(d_v)
                    de_list.append(d_e)
                    tag_list.append(f"{labels[i]} vs {labels[j]}")
            fig1, points1 = plot_dv_vs_de(dv_list, de_list, labels=tag_list, title="All Label Pairs")
            st.pyplot(fig1)
            for lbl, dvv, dee in points1:
                st.markdown(f"- **{lbl}**: d_v = {dvv:.4f}, d_e = {dee:.4f}")

        with col_all:
            st.subheader("🔄 TVD: Each Label vs ALL")
            dv_all, de_all, point_data = [], [], []
            for lab in labels:
                dv_i, de_i = tvd_label_vs_all(X_scaled, y.squeeze(), build_corr_graph(corr, threshold), lab, bins=bins)
                dv_all.append(dv_i)
                de_all.append(de_i)
                point_data.append((lab, dv_i, de_i))
            fig2, _ = plot_dv_vs_de(dv_all, de_all, labels=[str(l) for l in labels], title="Each Label vs ALL")
            st.pyplot(fig2)
            for lbl, dvv, dee in point_data:
                st.markdown(f"- **{lbl} vs ALL**: d_v = {dvv:.4f}, d_e = {dee:.4f}")
else:
    st.warning("Enter a valid UCI ID, URL or local CSV path to get started.")
