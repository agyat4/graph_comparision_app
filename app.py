import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import poisson, norm

# Set page config
st.set_page_config(layout="wide", page_title="Graph Distribution Comparison")
st.title("📊 Graph Signal Distribution Comparison")

# Define the distance functions
def poisson_tvd(lam_p, lam_q, support=np.arange(0, 20)):
    p = poisson.pmf(support, lam_p)
    q = poisson.pmf(support, lam_q)
    return 0.5 * np.sum(np.abs(p - q))

def gaussian_w2(mu1, mu2, sigma1=1.0, sigma2=1.0):
    """Wasserstein-2 distance for 1D Gaussians"""
    return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)

# Initialize graph
nodes = [0, 1]
edges = [(0, 1)]
values_GT = np.array([0.3, 0.7])
sigma_GT = np.array([1.0, 1.0])  # Default standard deviations

# Create tabs for Poisson and Gaussian
tab_poisson, tab_gaussian = st.tabs(["Poisson TVD", "Gaussian W2"])

# Function to draw graph with better labels
def draw_graph(ax, values, title, color, node_size=1200, font_size=10):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = {0: (0, 0), 1: (1, 0)}
    
    nx.draw(G, pos, with_labels=False, node_color=color, 
            node_size=node_size, edge_color='gray', ax=ax, linewidths=2)
    
    # Draw custom labels with background
    for node in G.nodes:
        x, y = pos[node]
        text = f"{values[node]:.2f}"
        ax.text(x, y, text, 
                ha='center', va='center', 
                fontsize=font_size, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", 
                          fc=(1,1,1,0.8), 
                          ec="none"))
    
    ax.set_title(title, fontweight='bold')
    ax.set_axis_off()
    return ax

# Poisson TVD Tab
with tab_poisson:
    st.header("Poisson Distribution Analysis (TVD)")
    st.markdown("**Total Variation Distance (TVD) between Poisson distributions**")
    
    # Sidebar controls for Poisson
    with st.sidebar:
        st.subheader("Poisson Controls")
        node0_val = st.slider("Node 0 Mean (λ)", 0.0, 2.0, 0.3, 0.01, key="poisson_node0")
        node1_val = st.slider("Node 1 Mean (λ)", 0.0, 2.0, 0.7, 0.01, key="poisson_node1")
    
    pred_values = np.array([node0_val, node1_val])
    
    # Calculate Poisson distances
    dv_tvd = sum(poisson_tvd(values_GT[i], pred_values[i]) for i in nodes)
    de_tvd = sum(poisson_tvd(abs(values_GT[u] - values_GT[v]), abs(pred_values[u] - pred_values[v])) for u, v in edges)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Poisson PMF Comparison")
        fig_pmf, ax_pmf = plt.subplots(1, 2, figsize=(12, 4))
        support = np.arange(0, 20)
        
        # Original Graph: Both nodes in one plot
        ax_pmf[0].bar(support - 0.2, poisson.pmf(support, values_GT[0]), 
                    width=0.4, alpha=0.7, label=f'Node 0 (λ={values_GT[0]:.2f})')
        ax_pmf[0].bar(support + 0.2, poisson.pmf(support, values_GT[1]), 
                    width=0.4, alpha=0.7, label=f'Node 1 (λ={values_GT[1]:.2f})')
        ax_pmf[0].set_title("Original Graph: Both Nodes")
        ax_pmf[0].legend()
        
        # Modified Graph: Both nodes in one plot
        ax_pmf[1].bar(support - 0.2, poisson.pmf(support, pred_values[0]), 
                    width=0.4, alpha=0.7, label=f'Node 0 (λ={pred_values[0]:.2f})')
        ax_pmf[1].bar(support + 0.2, poisson.pmf(support, pred_values[1]), 
                    width=0.4, alpha=0.7, label=f'Node 1 (λ={pred_values[1]:.2f})')
        ax_pmf[1].set_title("Modified Graph: Both Nodes")
        ax_pmf[1].legend()
        
        st.pyplot(fig_pmf)

        # Graph Visualization for Poisson - below PMF
        st.subheader("Graph Representation")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            fig_orig, ax_orig = plt.subplots(figsize=(3, 3))
            draw_graph(ax_orig, values_GT, "Original Graph", "lightblue", font_size=10)
            st.pyplot(fig_orig)

        with col_graph2:
            fig_mod, ax_mod = plt.subplots(figsize=(3, 3))
            draw_graph(ax_mod, pred_values, "Modified Graph", "lightgreen", font_size=10)
            st.pyplot(fig_mod)

    with col2:
        st.subheader("Distance Metrics")
        st.markdown(f"**Node TVD Distance (d_v):** `{dv_tvd:.4f}`")
        st.markdown(f"**Edge TVD Distance (d_e):** `{de_tvd:.4f}`")
        
        fig_dist, ax_dist = plt.subplots(figsize=(6, 6))
        ax_dist.scatter([dv_tvd], [de_tvd], color='blue', s=100)
        max_val = max(dv_tvd, de_tvd) * 1.2
        ax_dist.plot([0, max_val], [0, max_val], 'r--', label='d_v = d_e')
        ax_dist.set_xlabel("Node d_v (Poisson TVD)")
        ax_dist.set_ylabel("Edge d_e (Poisson TVD)")
        ax_dist.set_title(f"TVD: d_v vs d_e")
        ax_dist.legend()
        ax_dist.grid(True)
        st.pyplot(fig_dist)
        
        # Display distance explanation
        st.markdown("### Distance Interpretation")
        st.info("""
        - **TVD (Total Variation Distance)**: Measures the largest possible difference in probability 
        assigned to the same event by two distributions
        - Lower values indicate more similar distributions
        - The red line shows where node and edge distances would be equal
        """)

# Gaussian W2 Tab
with tab_gaussian:
    st.header("Gaussian Distribution Analysis (W2)")
    st.markdown("**Wasserstein-2 Distance (W2) between Gaussian distributions**")
    
    # Sidebar controls for Gaussian
    with st.sidebar:
        st.subheader("Gaussian Controls")
        gauss_node0_mean = st.slider("Node 0 Mean (μ)", 0.0, 2.0, 0.3, 0.01, key="gauss_node0_mean")
        gauss_node0_std = st.slider("Node 0 Std Dev (σ)", 0.1, 2.0, 1.0, 0.1, key="gauss_node0_std")
        gauss_node1_mean = st.slider("Node 1 Mean (μ)", 0.0, 2.0, 0.7, 0.01, key="gauss_node1_mean")
        gauss_node1_std = st.slider("Node 1 Std Dev (σ)", 0.1, 2.0, 1.0, 0.1, key="gauss_node1_std")
    
    pred_means = np.array([gauss_node0_mean, gauss_node1_mean])
    pred_stds = np.array([gauss_node0_std, gauss_node1_std])
    
    # Calculate Gaussian distances
    dv_w2 = sum(gaussian_w2(values_GT[i], pred_means[i], sigma_GT[i], pred_stds[i]) for i in nodes)
    de_w2 = sum(gaussian_w2(
        abs(values_GT[u] - values_GT[v]), 
        abs(pred_means[u] - pred_means[v]),
        sigma_GT[0],  # Fixed sigma for edge comparison
        sigma_GT[0]
    ) for u, v in edges)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gaussian PDF Comparison")
        fig_pdf, ax_pdf = plt.subplots(1, 2, figsize=(12, 4))
        x = np.linspace(-3, 3, 1000)
        
        # Original Graph: Both nodes in one plot
        ax_pdf[0].plot(x, norm.pdf(x, values_GT[0], sigma_GT[0]), 'b-', alpha=0.8, 
                     label=f'Node 0 (μ={values_GT[0]:.2f}, σ={sigma_GT[0]:.2f})')
        ax_pdf[0].plot(x, norm.pdf(x, values_GT[1], sigma_GT[1]), 'r-', alpha=0.8, 
                     label=f'Node 1 (μ={values_GT[1]:.2f}, σ={sigma_GT[1]:.2f})')
        ax_pdf[0].fill_between(x, 0, norm.pdf(x, values_GT[0], sigma_GT[0]), color='blue', alpha=0.1)
        ax_pdf[0].fill_between(x, 0, norm.pdf(x, values_GT[1], sigma_GT[1]), color='red', alpha=0.1)
        ax_pdf[0].set_title("Original Graph: Both Nodes")
        ax_pdf[0].legend()
        
        # Modified Graph: Both nodes in one plot
        ax_pdf[1].plot(x, norm.pdf(x, pred_means[0], pred_stds[0]), 'b-', alpha=0.8, 
                     label=f'Node 0 (μ={pred_means[0]:.2f}, σ={pred_stds[0]:.2f})')
        ax_pdf[1].plot(x, norm.pdf(x, pred_means[1], pred_stds[1]), 'r-', alpha=0.8, 
                     label=f'Node 1 (μ={pred_means[1]:.2f}, σ={pred_stds[1]:.2f})')
        ax_pdf[1].fill_between(x, 0, norm.pdf(x, pred_means[0], pred_stds[0]), color='blue', alpha=0.1)
        ax_pdf[1].fill_between(x, 0, norm.pdf(x, pred_means[1], pred_stds[1]), color='red', alpha=0.1)
        ax_pdf[1].set_title("Modified Graph: Both Nodes")
        ax_pdf[1].legend()
        
        st.pyplot(fig_pdf)

        # Graph Visualization for Gaussian -just the below PDF
        st.subheader("Graph Representation")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            fig_orig, ax_orig = plt.subplots(figsize=(3, 3))
            draw_graph(ax_orig, values_GT, "Original Graph", "lightblue", font_size=10)
            st.pyplot(fig_orig)

        with col_graph2:
            fig_mod, ax_mod = plt.subplots(figsize=(3, 3))
            draw_graph(ax_mod, pred_means, "Modified Graph", "lightgreen", font_size=10)
            st.pyplot(fig_mod)

    with col2:
        st.subheader("Distance Metrics")
        st.markdown(f"**Node W2 Distance (d_v):** `{dv_w2:.4f}`")
        st.markdown(f"**Edge W2 Distance (d_e):** `{de_w2:.4f}`")
        
        fig_dist, ax_dist = plt.subplots(figsize=(6, 6))
        ax_dist.scatter([dv_w2], [de_w2], color='green', s=100)
        max_val = max(dv_w2, de_w2) * 1.2
        ax_dist.plot([0, max_val], [0, max_val], 'r--', label='d_v = d_e')
        ax_dist.set_xlabel("Node d_v (Gaussian W2)")
        ax_dist.set_ylabel("Edge d_e (Gaussian W2)")
        ax_dist.set_title(f"Wasserstein-2: d_v vs d_e")
        ax_dist.legend()
        ax_dist.grid(True)
        st.pyplot(fig_dist)
        
        # Display distance explanation
        st.markdown("### Distance Interpretation")
        st.info("""
        - **W2 (Wasserstein-2 Distance)**: Measures the "earth mover's distance" between distributions
        - Accounts for both mean and standard deviation differences
        - The red line shows where node and edge distances would be equal
        """)

# Footer
st.markdown("---")
st.caption("Interactive Graph Signal Distribution Comparison | Switch tabs to compare different distance metrics")