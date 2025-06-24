# Minimal prototype for custom graph creation and distribution comparison with Streamlit
import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

st.set_page_config(layout="wide")
st.title("📊 Custom Graph Distribution Comparison")

# Distance functions
def poisson_tvd(lam_p, lam_q, support=None):
    if support is None:
        support = np.arange(0, 30)
    p = poisson.pmf(support, lam_p)
    q = poisson.pmf(support, lam_q)
    return 0.5 * np.sum(np.abs(p - q))

def gaussian_w2(mu1, mu2, sigma1=1.0, sigma2=1.0):
    return np.sqrt((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2)

left_col, right_col = st.columns([1, 2])

with st.sidebar:
    dist_mode = st.radio("Select Distribution Type", ["Poisson", "Gaussian"])

with left_col:
    with st.expander("⚙️ Graph Configuration Panel", expanded=True):
        num_nodes = st.number_input("Select number of nodes", min_value=2, max_value=10, value=3, step=1)
        nodes = list(range(num_nodes))

        st.subheader("Add edges between nodes")
        if "custom_edges" not in st.session_state:
            st.session_state.custom_edges = []

        col1, col2 = st.columns(2)
        with col1:
            node_a = st.selectbox("From node", nodes, key="node_a")
        with col2:
            node_b = st.selectbox("To node", nodes, key="node_b")

        if st.button("Add Edge"):
            edge = (min(node_a, node_b), max(node_a, node_b))
            if node_a != node_b and edge not in st.session_state.custom_edges:
                st.session_state.custom_edges.append(edge)

        st.markdown("### Current Edges")
        st.write(st.session_state.custom_edges)
        if st.button("Reset Edges"):
            st.session_state.custom_edges = []

        st.subheader("")
        original_lambdas = [0.5 + i * 0.1 for i in nodes]

        st.subheader("Assign λ values to Modified Graph")
        modified_lambdas = [
            st.slider(f"Modified λ for Node {i}", 0.0, 2.0, original_lambdas[i], 0.01, key=f"mod_lambda_{i}")
            for i in nodes
        ]

        st.subheader("Assign σ values to Modified Graph")
        pred_sigma = [
            st.slider(f"σ for Node {i}", 0.1, 2.0, 1.0, 0.1, key=f"sigma_{i}", disabled=(dist_mode != "Gaussian"))
            for i in nodes
        ]

with right_col:
    st.subheader("Graph Visualization")
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(st.session_state.custom_edges)

    col_gt, col_mod = st.columns(2)
    with col_gt:
        fig1, ax1 = plt.subplots(figsize=(3, 2))
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=600, ax=ax1)
        for n, (x, y) in pos.items():
            ax1.text(x, y, f"{original_lambdas[n]:.2f}", ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', lw=0.5))
        ax1.set_title("Original Graph")
        ax1.axis('off')
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    with col_mod:
        fig2, ax2 = plt.subplots(figsize=(3, 2))
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=600, ax=ax2)
        for n, (x, y) in pos.items():
            ax2.text(x, y, f"{modified_lambdas[n]:.2f}", ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', lw=0.5))
        ax2.set_title("Modified Graph")
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # Tabs for distribution comparison
    tab_poisson, tab_gaussian = st.tabs(["Poisson TVD", "Gaussian W₂"])

    with tab_poisson:
        st.subheader("Poisson Distribution Analysis")
        support = np.arange(0, 30)
        dv_tvd = sum(poisson_tvd(original_lambdas[i], modified_lambdas[i], support) for i in nodes)
        de_tvd = sum(poisson_tvd(abs(original_lambdas[u] - original_lambdas[v]), abs(modified_lambdas[u] - modified_lambdas[v]), support)
                     for u, v in st.session_state.custom_edges)

        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in nodes:
                ax.bar(support + i * 0.2 - 0.2, poisson.pmf(support, original_lambdas[i]), width=0.2, alpha=0.6, label=f"Node {i}")
            ax.set_title("Original PMFs")
            ax.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in nodes:
                ax.bar(support + i * 0.2 - 0.2, poisson.pmf(support, modified_lambdas[i]), width=0.2, alpha=0.6, label=f"Node {i}")
            ax.set_title("Modified PMFs")
            ax.legend()
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter([dv_tvd], [de_tvd], s=80)
            max_val = max(dv_tvd, de_tvd) * 1.2 + 1e-6
            ax.plot([0, max_val], [0, max_val], "r--", label="d_v = d_e")
            ax.set_xlabel("Node d_v (TVD)")
            ax.set_ylabel("Edge d_e (TVD)")
            ax.grid(True)
            ax.legend()
            ax.set_title("TVD: d_v vs d_e")
            st.pyplot(fig)
            st.markdown(f"**Σ Node TVD (d_v):** `{dv_tvd:.4f}`")
            st.markdown(f"**Σ Edge TVD (d_e):** `{de_tvd:.4f}`")

    with tab_gaussian:
        st.subheader("Gaussian Distribution Analysis")
        SIGMA_GT = np.ones(len(nodes))

        dv_w2 = sum(gaussian_w2(original_lambdas[i], modified_lambdas[i], SIGMA_GT[i], pred_sigma[i]) for i in nodes)
        de_w2 = sum(gaussian_w2(abs(original_lambdas[u] - original_lambdas[v]), abs(modified_lambdas[u] - modified_lambdas[v]), 1.0, 1.0)
                     for u, v in st.session_state.custom_edges)

        col1, col2, col3 = st.columns(3)
        x = np.linspace(-1, 3, 1000)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in nodes:
                ax.plot(x, norm.pdf(x, original_lambdas[i], SIGMA_GT[i]), label=f"Node {i}")
            ax.set_title("Original PDFs")
            ax.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in nodes:
                ax.plot(x, norm.pdf(x, modified_lambdas[i], pred_sigma[i]), label=f"Node {i}")
            ax.set_title("Modified PDFs")
            ax.legend()
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter([dv_w2], [de_w2], s=80, color="green")
            max_val = max(dv_w2, de_w2) * 1.2 + 1e-6
            ax.plot([0, max_val], [0, max_val], "r--", label="d_v = d_e")
            ax.set_xlabel("Node d_v (W₂)")
            ax.set_ylabel("Edge d_e (W₂)")
            ax.grid(True)
            ax.legend()
            ax.set_title("W₂: d_v vs d_e")
            st.pyplot(fig)
            st.markdown(f"**Σ Node W₂ (d_v):** `{dv_w2:.4f}`")
            st.markdown(f"**Σ Edge W₂ (d_e):** `{de_w2:.4f}`")

# Footer
st.markdown("---")
st.caption("Custom graph builder with distribution comparison (Poisson TVD and Gaussian W₂)")
