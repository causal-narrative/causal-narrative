"""
Visualization module: static, interactive, hierarchical, themes, and utilities.

This module combines all visualization functionality:
- VizTheme and theme management
- Static high-resolution network visualization (matplotlib)
- Interactive network visualization (pyvis, plotly)
- Hierarchical/layered network visualization
- Visualization utility functions
"""

# CRITICAL: Apply nest_asyncio at module level for Jupyter notebook compatibility
# This must be done BEFORE any playwright imports anywhere in the code
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Will warn later if PNG export is attempted

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import seaborn as sns
from loguru import logger

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False


# =============================================================================
# Themes
# =============================================================================

@dataclass
class VizTheme:
    """Unified visualization theme configuration."""

    name: str = "default"

    # Figure settings
    figsize: Tuple[float, float] = (16.0, 12.0)
    dpi: int = 300

    # Font settings
    font_family: str = "sans-serif"
    font_size_base: int = 10
    font_size_title: int = 14
    font_size_label: int = 10
    font_size_legend: int = 9

    # Color palettes
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ])

    # Transparency settings
    edge_alpha: float = 0.6
    node_alpha: float = 0.9

    # Colormaps
    edge_colormap: str = "YlOrRd"
    node_colormap: str = "viridis"

    # Background and grid
    background_color: str = "white"
    grid_color: str = "#e0e0e0"
    grid_alpha: float = 0.3

    # Spines
    spine_color: str = "#333333"
    spine_width: float = 0.5

    def apply_mpl_style(self) -> None:
        """Apply theme to matplotlib rcParams."""
        plt.rcParams["font.family"] = self.font_family
        plt.rcParams["font.size"] = self.font_size_base
        plt.rcParams["axes.titlesize"] = self.font_size_title
        plt.rcParams["axes.labelsize"] = self.font_size_label
        plt.rcParams["legend.fontsize"] = self.font_size_legend
        plt.rcParams["xtick.labelsize"] = self.font_size_label - 1
        plt.rcParams["ytick.labelsize"] = self.font_size_label - 1

        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["figure.facecolor"] = self.background_color
        plt.rcParams["axes.facecolor"] = self.background_color

        plt.rcParams["axes.edgecolor"] = self.spine_color
        plt.rcParams["axes.linewidth"] = self.spine_width

        plt.rcParams["grid.color"] = self.grid_color
        plt.rcParams["grid.alpha"] = self.grid_alpha
        plt.rcParams["grid.linewidth"] = 0.5

        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False

        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.framealpha"] = 0.8
        plt.rcParams["legend.fancybox"] = True

    def get_color_palette(self, n_colors: Optional[int] = None) -> List[str]:
        """Get color palette with n_colors."""
        if n_colors is None:
            return self.color_palette

        if n_colors <= len(self.color_palette):
            return self.color_palette[:n_colors]

        palette = sns.color_palette("husl", n_colors)
        return [mpl.colors.rgb2hex(c) for c in palette]

    def get_node_color_by_community(
        self,
        n_communities: int,
        community_id: int
    ) -> str:
        """Get color for a node based on community."""
        palette = self.get_color_palette(n_communities)
        return palette[community_id % len(palette)]

    def get_edge_color_by_weight(
        self,
        weight: float,
        min_weight: float,
        max_weight: float
    ) -> str:
        """Get color for edge based on weight."""
        if max_weight == min_weight:
            normalized = 0.5
        else:
            normalized = (weight - min_weight) / (max_weight - min_weight)

        cmap = plt.get_cmap(self.edge_colormap)
        rgba = cmap(normalized)
        return mpl.colors.rgb2hex(rgba[:3])

    def get_node_size_by_centrality(
        self,
        centrality: float,
        min_centrality: float,
        max_centrality: float,
        min_size: float = 100.0,
        max_size: float = 3000.0
    ) -> float:
        """Get node size based on centrality."""
        if max_centrality == min_centrality:
            return (min_size + max_size) / 2

        normalized = (centrality - min_centrality) / (max_centrality - min_centrality)
        return min_size + normalized * (max_size - min_size)


def get_default_theme() -> VizTheme:
    """Get default visualization theme."""
    return VizTheme(name="default")


def get_publication_theme() -> VizTheme:
    """Get publication-ready theme with high contrast."""
    theme = VizTheme(
        name="publication",
        figsize=(12.0, 9.0),
        dpi=300,
        font_family="serif",
        font_size_base=11,
        font_size_title=16,
        font_size_label=12,
        font_size_legend=10,
        background_color="white",
        edge_alpha=0.7,
        node_alpha=1.0,
        spine_color="black",
        spine_width=1.0,
    )
    return theme


def get_presentation_theme() -> VizTheme:
    """Get theme optimized for presentations (dark background)."""
    theme = VizTheme(
        name="presentation",
        figsize=(16.0, 9.0),
        dpi=150,
        font_size_base=14,
        font_size_title=20,
        font_size_label=16,
        font_size_legend=14,
        background_color="#2e2e2e",
        grid_color="#555555",
        spine_color="#cccccc",
        edge_alpha=0.7,
        node_alpha=0.95,
        color_palette=[
            "#ff6b6b", "#4ecdc4", "#45b7d1", "#fdcb6e",
            "#6c5ce7", "#a29bfe", "#fd79a8", "#00b894",
        ],
    )
    return theme


def get_minimal_theme() -> VizTheme:
    """Get minimal theme with clean aesthetics."""
    theme = VizTheme(
        name="minimal",
        figsize=(14.0, 10.0),
        font_size_base=9,
        font_size_title=12,
        font_size_label=9,
        font_size_legend=8,
        background_color="#fafafa",
        grid_color="#e8e8e8",
        edge_alpha=0.5,
        node_alpha=0.85,
        spine_width=0.3,
        color_palette=[
            "#34495e", "#95a5a6", "#7f8c8d", "#bdc3c7",
            "#2c3e50", "#ecf0f1", "#95a5a6", "#7f8c8d",
        ],
    )
    return theme


THEME_REGISTRY: Dict[str, VizTheme] = {
    "default": get_default_theme(),
    "publication": get_publication_theme(),
    "presentation": get_presentation_theme(),
    "minimal": get_minimal_theme(),
}


def get_theme(name: str = "default") -> VizTheme:
    """Get theme by name."""
    if name not in THEME_REGISTRY:
        raise ValueError(
            f"Unknown theme: {name}. "
            f"Available themes: {', '.join(THEME_REGISTRY.keys())}"
        )

    return THEME_REGISTRY[name]


def get_community_color(community_id: int, n_communities: int) -> str:
    """Get color for a community.

    Args:
        community_id: Community index
        n_communities: Total number of communities

    Returns:
        Color hex code
    """
    theme = get_default_theme()
    return theme.get_node_color_by_community(n_communities, community_id)


def apply_theme(name: str) -> None:
    """Apply a named theme to matplotlib.

    Args:
        name: Theme name
    """
    theme = get_theme(name)
    theme.apply_mpl_style()


# =============================================================================
# Static Visualization
# =============================================================================

def plot_network_static(
    G: nx.DiGraph,
    output_path: Path,
    layout: str = "spring",
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 300,
    node_size_attr: str = "degree",
    edge_width_attr: str = "weight",
    color_by_community: bool = True,
    label_top_k: Optional[int] = 20,
    title: Optional[str] = None,
    theme: str = "academic",
    **kwargs,
) -> None:
    """Create static high-resolution network visualization."""
    if len(G.nodes()) == 0:
        logger.warning("Empty graph, skipping visualization")
        return

    apply_theme(theme if theme in THEME_REGISTRY else "default")
    theme_config = get_theme(theme if theme in THEME_REGISTRY else "default")

    fig, ax = plt.subplots(figsize=figsize, facecolor=theme_config.background_color)
    ax.set_facecolor(theme_config.background_color)
    ax.axis("off")

    logger.info(f"Computing {layout} layout for {len(G.nodes())} nodes...")
    if layout == "spring":
        pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    if node_size_attr == "degree":
        sizes = dict(G.degree())
    elif node_size_attr == "pagerank":
        sizes = nx.pagerank(G)
    elif node_size_attr == "betweenness":
        sizes = nx.betweenness_centrality(G)
    else:
        sizes = {n: 1 for n in G.nodes()}

    size_values = list(sizes.values())
    min_size, max_size = 300, 3000
    if max(size_values) > 0:
        node_sizes = [min_size + (sizes[n] / max(size_values)) * (max_size - min_size)
                      for n in G.nodes()]
    else:
        node_sizes = [min_size] * len(G.nodes())

    if color_by_community:
        try:
            import community.community_louvain as community_louvain
            communities = community_louvain.best_partition(G.to_undirected())
            unique_communities = set(communities.values())
            node_colors = [get_community_color(communities[n], len(unique_communities))
                          for n in G.nodes()]
        except ImportError:
            logger.warning("python-louvain not installed, using default colors")
            node_colors = [theme_config.color_palette[0]] * len(G.nodes())
    else:
        node_colors = [theme_config.color_palette[0]] * len(G.nodes())

    edge_weights = [G[u][v].get(edge_width_attr, 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (w / max_weight) * 3 for w in edge_weights]

    min_alpha, max_alpha = 0.2, 0.8
    edge_alphas = [min_alpha + (w / max_weight) * (max_alpha - min_alpha) for w in edge_weights]

    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                   lw=width,
                                   alpha=alpha,
                                   color="gray",
                                   connectionstyle="arc3,rad=0.1"))

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.9, linewidths=1, edgecolors="white", ax=ax)

    if label_top_k is not None:
        centrality = sizes
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:label_top_k]
        labels_to_draw = {n: G.nodes[n].get("label", str(n)) for n, _ in top_nodes}
    else:
        labels_to_draw = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}

    if HAS_ADJUSTTEXT and len(labels_to_draw) > 0:
        texts = []
        for node, label in labels_to_draw.items():
            x, y = pos[node]
            t = ax.text(x, y, label, fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="gray", alpha=0.8))
            texts.append(t)

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    else:
        nx.draw_networkx_labels(G, pos, labels_to_draw, font_size=8, ax=ax)

    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)

    if color_by_community and 'communities' in locals():
        legend_elements = []
        for comm_id in sorted(unique_communities)[:10]:
            color = get_community_color(comm_id, len(unique_communities))
            legend_elements.append(mpatches.Patch(color=color, label=f'Community {comm_id}'))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()

    if output_path.suffix == '.svg':
        plt.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')

    plt.close()

    logger.info(f"Saved static visualization to {output_path}")


# =============================================================================
# Interactive Visualization
# =============================================================================

def plot_network_interactive_pyvis(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Causal Narrative Network",
    height: str = "800px",
    width: str = "100%",
    notebook: bool = False,
    top_n: Optional[int] = None,
    font_size: int = 14,
    **kwargs,
) -> None:
    """Create interactive network visualization using pyvis.
    
    Args:
        G: NetworkX directed graph
        output_path: Path to save the HTML file
        title: Title of the visualization
        height: Height of the visualization (e.g., '800px')
        width: Width of the visualization (e.g., '100%')
        notebook: Whether to display in Jupyter notebook
        top_n: Keep only top N nodes by degree (None = all nodes)
        font_size: Font size for node labels (default: 14)
        **kwargs: Additional arguments
    
    Example:
        >>> plot_network_interactive_pyvis(
        ...     G, 
        ...     'output/network.html',
        ...     top_n=100,  # Only show top 100 nodes
        ...     font_size=20  # Larger font
        ... )
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("pyvis not installed. Install with: pip install pyvis")
        return

    if len(G.nodes()) == 0:
        logger.warning("Empty graph, skipping visualization")
        return
    
    # Filter to top N nodes by degree if specified
    if top_n is not None and G.number_of_nodes() > top_n:
        logger.info(f"Filtering graph to top {top_n} nodes by degree...")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_ids).copy()
        logger.info(f"Filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    net = Network(height=height, width=width, directed=True, notebook=notebook)
    net.barnes_hut()

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      }
    }
    """)

    for node in G.nodes():
        label = G.nodes[node].get("label", str(node))
        title_text = f"<b>{label}</b><br>"

        if "size" in G.nodes[node]:
            title_text += f"Size: {G.nodes[node]['size']}<br>"
        if "exemplars" in G.nodes[node]:
            exemplars = G.nodes[node]["exemplars"][:3]
            title_text += f"Examples:<br>{'<br>'.join(exemplars)}"

        size = G.degree(node) * 3 + 10

        net.add_node(
            node,
            label=label,
            title=title_text,
            size=size,
            color="#2E86AB",
            font={"size": font_size},
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        title_text = f"Weight: {weight}"

        if "examples" in data:
            examples = data["examples"][:2]
            title_text += f"<br>Examples:<br>{'<br>'.join(examples)}"

        net.add_edge(u, v, value=weight, title=title_text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    logger.info(f"Saved interactive pyvis visualization to {output_path}")


def plot_network_interactive_plotly(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Causal Narrative Network",
    top_n: Optional[int] = None,
    **kwargs,
) -> None:
    """Create interactive network visualization using plotly.
    
    Args:
        G: NetworkX directed graph
        output_path: Path to save the HTML file
        title: Title of the visualization
        top_n: Keep only top N nodes by degree (None = all nodes)
        **kwargs: Additional arguments
    
    Example:
        >>> plot_network_interactive_plotly(
        ...     G, 
        ...     'output/network.html',
        ...     top_n=100  # Only show top 100 nodes
        ... )
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.error("plotly not installed. Install with: pip install plotly")
        return

    if len(G.nodes()) == 0:
        logger.warning("Empty graph, skipping visualization")
        return
    
    # Filter to top N nodes by degree if specified
    if top_n is not None and G.number_of_nodes() > top_n:
        logger.info(f"Filtering graph to top {top_n} nodes by degree...")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_ids).copy()
        logger.info(f"Filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    edge_x = []
    edge_y = []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Edges'
    )

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        label = G.nodes[node].get("label", str(node))
        degree = G.degree(node)
        node_text.append(f"{label}<br>Degree: {degree}")
        node_sizes.append(degree * 5 + 10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n].get("label", str(n)) for n in G.nodes()],
        hovertext=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=[G.degree(n) for n in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        name='Nodes'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))

    logger.info(f"Saved interactive plotly visualization to {output_path}")


# =============================================================================
# Hierarchical Visualization
# =============================================================================

def _hierarchical_layout(G: nx.DiGraph) -> Dict:
    """Compute hierarchical layout using longest path layering."""
    try:
        layers = list(nx.topological_generations(G))
    except nx.NetworkXError:
        logger.info("Graph is not a DAG, using SCC-based layering")
        scc = list(nx.strongly_connected_components(G))
        node_to_layer = {}
        for i, component in enumerate(scc):
            for node in component:
                node_to_layer[node] = i

        max_layer = max(node_to_layer.values())
        layers = [[] for _ in range(max_layer + 1)]
        for node, layer in node_to_layer.items():
            layers[layer].append(node)

    pos = {}
    y_gap = 1.0 / (len(layers) + 1)

    for layer_idx, layer_nodes in enumerate(layers):
        y = 1.0 - (layer_idx + 1) * y_gap
        x_gap = 1.0 / (len(layer_nodes) + 1) if layer_nodes else 0.5

        for node_idx, node in enumerate(layer_nodes):
            x = (node_idx + 1) * x_gap
            pos[node] = (x, y)

    return pos


def plot_network_hierarchical(
    G: nx.DiGraph,
    output_path: Path,
    method: str = "auto",
    figsize: tuple = (16, 12),
    dpi: int = 300,
    **kwargs,
) -> None:
    """Create hierarchical/layered network visualization."""
    if len(G.nodes()) == 0:
        logger.warning("Empty graph, skipping visualization")
        return

    if method == "auto":
        if shutil.which("dot"):
            method = "graphviz"
        else:
            method = "hierarchy"
            logger.info("Graphviz not found, using networkx hierarchical layout")

    if method == "graphviz" and shutil.which("dot"):
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception as e:
            logger.warning(f"Graphviz layout failed: {e}, falling back to hierarchy")
            pos = _hierarchical_layout(G)
    else:
        pos = _hierarchical_layout(G)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True,
                          arrowsize=15, edge_color="gray", ax=ax)

    node_sizes = [G.degree(n) * 50 + 200 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color="#2E86AB", alpha=0.9, ax=ax)

    labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved hierarchical visualization to {output_path}")


# =============================================================================
# Visualization Utilities
# =============================================================================

def filter_network_for_viz(
    G: nx.DiGraph,
    top_k: Optional[int] = None,
    min_weight: Optional[float] = None,
    k_core: Optional[int] = None,
) -> nx.DiGraph:
    """Filter network for better visualization."""
    G_filtered = G.copy()

    if min_weight is not None:
        edges_to_remove = [(u, v) for u, v, d in G_filtered.edges(data=True)
                          if d.get('weight', 1) < min_weight]
        G_filtered.remove_edges_from(edges_to_remove)
        logger.info(f"Removed {len(edges_to_remove)} edges with weight < {min_weight}")

    isolated = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolated)

    if k_core is not None:
        try:
            core_nodes = nx.k_core(G_filtered.to_undirected(), k=k_core).nodes()
            G_filtered = G_filtered.subgraph(core_nodes).copy()
            logger.info(f"K-core({k_core}) filtering: {len(core_nodes)} nodes remaining")
        except Exception as e:
            logger.warning(f"K-core filtering failed: {e}")

    if top_k is not None and len(G_filtered.nodes()) > top_k:
        degrees = dict(G_filtered.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_node_set = {n for n, _ in top_nodes}
        G_filtered = G_filtered.subgraph(top_node_set).copy()
        logger.info(f"Kept top {top_k} nodes by degree")

    return G_filtered


def compute_node_sizes(
    G: nx.DiGraph,
    method: str = 'degree',
    min_size: float = 300,
    max_size: float = 3000,
) -> Dict:
    """Compute node sizes based on centrality."""
    if method == 'degree':
        values = dict(G.degree())
    elif method == 'pagerank':
        values = nx.pagerank(G)
    elif method == 'betweenness':
        values = nx.betweenness_centrality(G)
    else:
        return {n: min_size for n in G.nodes()}

    max_val = max(values.values()) if values else 1
    if max_val == 0:
        return {n: min_size for n in G.nodes()}

    sizes = {n: min_size + (v / max_val) * (max_size - min_size)
             for n, v in values.items()}

    return sizes


def assign_community_colors(
    G: nx.DiGraph,
    method: str = 'louvain',
) -> Dict:
    """Assign colors to nodes based on community detection."""
    try:
        if method == 'louvain':
            import community.community_louvain as community_louvain
            communities = community_louvain.best_partition(G.to_undirected())
        elif method == 'label_prop':
            communities_gen = nx.community.label_propagation_communities(G.to_undirected())
            communities = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    communities[node] = i
        else:
            communities = {n: 0 for n in G.nodes()}

        n_communities = len(set(communities.values()))
        colors = {n: get_community_color(communities[n], n_communities)
                 for n in G.nodes()}

        return colors

    except ImportError as e:
        logger.warning(f"Community detection failed: {e}")
        return {n: "#2E86AB" for n in G.nodes()}


def get_top_nodes_by_centrality(
    G: nx.DiGraph,
    k: int = 20,
    method: str = 'degree',
) -> List:
    """Get top K nodes by centrality measure."""
    if method == 'degree':
        centrality = dict(G.degree())
    elif method == 'pagerank':
        centrality = nx.pagerank(G)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(G)
    else:
        centrality = dict(G.degree())

    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:k]
    return [n for n, _ in top_nodes]


def compute_edge_bundling_alpha(
    G: nx.DiGraph,
    weight_attr: str = 'weight',
    min_alpha: float = 0.2,
    max_alpha: float = 0.8,
) -> Dict:
    """Compute edge alpha values based on weights."""
    weights = [d.get(weight_attr, 1) for u, v, d in G.edges(data=True)]

    if not weights:
        return {}

    max_weight = max(weights)
    if max_weight == 0:
        return {(u, v): min_alpha for u, v in G.edges()}

    alphas = {(u, v): min_alpha + (d.get(weight_attr, 1) / max_weight) * (max_alpha - min_alpha)
              for u, v, d in G.edges(data=True)}

    return alphas


# =============================================================================
# Convenience wrapper classes (for backward compatibility)
# =============================================================================

class StaticVisualizer:
    """Wrapper for static visualization."""
    
    @staticmethod
    def plot(G, output_path, **kwargs):
        return plot_network_static(G, output_path, **kwargs)


class InteractiveVisualizer:
    """Wrapper for interactive visualization."""
    
    @staticmethod
    def plot_pyvis(G, output_path, **kwargs):
        return plot_network_interactive_pyvis(G, output_path, **kwargs)
    
    @staticmethod
    def plot_plotly(G, output_path, **kwargs):
        return plot_network_interactive_plotly(G, output_path, **kwargs)


class HierarchicalVisualizer:
    """Wrapper for hierarchical visualization."""
    
    @staticmethod
    def plot(G, output_path, **kwargs):
        return plot_network_hierarchical(G, output_path, **kwargs)


# =============================================================================
# Pyvis Interactive Visualization (Based on causal_network_viz.py)
# =============================================================================

def plot_interactive_network_pyvis(
    networkx_graph,
    output_html="graph.html",
    notebook=False,
    show_buttons=False,
    only_physics_buttons=False,
    gravity=-100,
    options=None,
    width="1000px",
    height="1000px",
):
    """
    Plot interactive network using pyvis.
    
    Args:
        networkx_graph: NetworkX graph (can be directed or undirected)
        output_html: Output HTML file path (str or Path)
        notebook: If True, display in Jupyter notebook
        show_buttons: Show interactive controls
        only_physics_buttons: Show only physics controls
        gravity: Gravity strength (negative = repulsion)
        options: Custom pyvis options (JavaScript object string)
        width: Canvas width
        height: Canvas height
        
    Returns:
        str: Path to saved HTML file
        
    Example:
        >>> G = nx.DiGraph()
        >>> G.add_edge("A", "B", weight=1)
        >>> plot_interactive_network_pyvis(G, "network.html")
        'network.html'
    """
    # Convert Path to string
    output_html = str(output_html)
    
    try:
        from pyvis import network as net
    except ImportError:
        raise ImportError(
            "pyvis is required for interactive visualization. "
            "Install with: pip install pyvis"
        )
    
    # Create pyvis network
    is_directed = isinstance(networkx_graph, (nx.DiGraph, nx.MultiDiGraph))
    pyvis_graph = net.Network(notebook=notebook, directed=is_directed)
    pyvis_graph.width = width
    pyvis_graph.height = height
    
    # Add nodes with attributes
    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node, **node_attrs)
    
    # Add edges with attributes
    for source, target, edge_attrs in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(source, target, **edge_attrs)
    
    # Controls
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=["physics"])
        else:
            pyvis_graph.show_buttons()
    
    # Edge smoothing
    pyvis_graph.set_edge_smooth("dynamic")
    
    # Physics
    pyvis_graph.barnes_hut(gravity=gravity, overlap=0.1)
    
    # Custom options
    if options:
        pyvis_graph.set_options(options)
    
    # Save
    pyvis_graph.save_graph(output_html)
    
    return output_html


def export_pyvis_to_png(
    html_path,
    png_path,
    width=3000,
    height=3000,
    wait_ms=15000
):
    """
    Export pyvis HTML to high-resolution PNG using playwright.
    
    Args:
        html_path: Path to pyvis HTML file (str or Path)
        png_path: Output PNG path (str or Path)
        width: Canvas width in pixels
        height: Canvas height in pixels
        wait_ms: Time to wait for physics stabilization (ms)
        
    Returns:
        str: Path to saved PNG file
        
    Example:
        >>> plot_interactive_network_pyvis(G, "network.html")
        >>> export_pyvis_to_png("network.html", "network.png")
        'network.png'
        
    Note:
        Requires: pip install playwright nest-asyncio && playwright install chromium
    """
    # Convert Path objects to strings
    html_path = str(html_path)
    png_path = str(png_path)
    
    import os
    
    # Import playwright (nest_asyncio already applied at module level)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for PNG export. "
            "Install with: pip install playwright && playwright install chromium"
        )
    
    print(f"Exporting to PNG:")
    print(f"  HTML: {html_path}")
    print(f"  PNG:  {png_path}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=2  # High DPI
            )
            
            # Load HTML
            page.goto(f"file://{os.path.abspath(html_path)}")
            
            # Wait for canvas
            print("  → Waiting for canvas...")
            page.wait_for_selector("canvas", timeout=30000)
            
            # Wait for physics stabilization
            print("  → Waiting for physics stabilization...")
            page.evaluate(
                """
                () => new Promise(resolve => {
                    const interval = setInterval(() => {
                        if (window.network && network.physics && !network.physics.physicsEnabled) {
                            clearInterval(interval);
                            resolve();
                        }
                    }, 500);
                    setTimeout(resolve, 15000);
                })
                """
            )
            
            # Extra buffer
            page.wait_for_timeout(wait_ms)
            
            # Screenshot canvas
            canvas = page.query_selector("canvas")
            canvas.screenshot(path=png_path)
            
            browser.close()
        
        print(f"  ✓ PNG saved: {png_path}")
        
    except Exception as e:
        logger.error(f"Failed to export PNG: {e}")
        logger.info("PNG export skipped. HTML file is still available.")
        return None
    
    return png_path


def build_causal_network_from_clusters(
    df,
    cause_cluster_col='cause_srl_cluster_id',
    cause_event_col='cause_srl_event',
    effect_cluster_col='effect_srl_cluster_id',
    effect_event_col='effect_srl_event',
    top_n=None,
    min_frequency=1,
    prune_network=True
):
    """
    Build a causal network from clustered events.
    
    Args:
        df: DataFrame with clustering results
        cause_cluster_col: Column name for cause cluster IDs
        cause_event_col: Column name for cause event names
        effect_cluster_col: Column name for effect cluster IDs
        effect_event_col: Column name for effect event names
        top_n: Keep only top N most frequent narratives (None = all)
        min_frequency: Minimum frequency threshold
        prune_network: Keep only largest connected component
        
    Returns:
        nx.MultiDiGraph: NetworkX graph with nodes and edges
        
    Example:
        >>> G = build_causal_network_from_clusters(
        ...     df,
        ...     cause_cluster_col='cause_srl_cluster_id',
        ...     cause_event_col='cause_srl_event',
        ...     effect_cluster_col='effect_srl_cluster_id',
        ...     effect_event_col='effect_srl_event',
        ...     top_n=100
        ... )
        >>> G.number_of_nodes(), G.number_of_edges()
        (45, 78)
    """
    import pandas as pd
    
    # Filter valid rows
    valid_mask = (
        (df[cause_cluster_col] >= 0) & 
        (df[effect_cluster_col] >= 0) &
        (df[cause_event_col] != '') &
        (df[effect_event_col] != '')
    )
    
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        logger.warning("No valid causal relations found")
        return nx.MultiDiGraph()
    
    # Create narrative strings
    df_valid['narrative'] = df_valid[cause_event_col] + ' -> ' + df_valid[effect_event_col]
    
    # Count frequencies
    narrative_counts = df_valid['narrative'].value_counts().reset_index()
    narrative_counts.columns = ['narrative', 'frequency']
    
    # Filter by minimum frequency
    narrative_counts = narrative_counts[narrative_counts['frequency'] >= min_frequency]
    
    # Keep top N
    if top_n is not None:
        narrative_counts = narrative_counts.head(top_n)
    
    print(f"Building network: {len(narrative_counts)} unique narratives")
    
    # Quantile-based edge width
    if len(narrative_counts) > 30:
        narrative_counts['q_freq'] = pd.qcut(
            narrative_counts['frequency'], 
            30, 
            duplicates='drop', 
            labels=False
        ) + 1
    else:
        # Linear scaling for small datasets
        max_freq = narrative_counts['frequency'].max()
        min_freq = narrative_counts['frequency'].min()
        narrative_counts['q_freq'] = (
            (narrative_counts['frequency'] - min_freq) / 
            (max_freq - min_freq + 1e-6) * 29 + 1
        )
    
    # Build network
    G = nx.MultiDiGraph()
    
    for _, row in narrative_counts.iterrows():
        narrative = row['narrative']
        frequency = row['frequency']
        q_freq = row['q_freq']
        
        if ' -> ' not in narrative:
            continue
        
        cause, effect = narrative.split(' -> ', 1)
        cause = cause.strip()
        effect = effect.strip()
        
        # Add edge with proper attributes for pyvis
        G.add_edge(
            cause,
            effect,
            value=float(q_freq),  # pyvis uses 'value' for edge width
            width=float(q_freq / 3),
            title=f"Frequency: {frequency}",  # Hover text
            frequency=frequency,
            color="#95a5a6"  # Use hex color
        )
    
    # Set node attributes
    degrees = dict(G.degree())
    
    # Calculate node sizes (scale to reasonable range)
    if degrees:
        max_degree = max(degrees.values())
        min_size = 10
        max_size = 50
        
        for node in G.nodes():
            degree = degrees[node]
            # Scale degree to size range
            if max_degree > 0:
                size = min_size + (degree / max_degree) * (max_size - min_size)
            else:
                size = min_size
            
            G.nodes[node]["size"] = size
            G.nodes[node]["color"] = "#3498db"  # Use hex color instead of rgba
            G.nodes[node]["title"] = f"{node}\nDegree: {degree}\nFrequency: {degree}"
            G.nodes[node]["label"] = node[:50]  # Truncate long labels
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Prune to largest component
    if prune_network and G.number_of_nodes() > 0:
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        print(f"  After pruning: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def visualize_causal_network(
    G,
    output_html="causal_network.html",
    output_png=None,
    notebook=False,
    gravity=-500,
    width="1500px",
    height="1500px",
    show_buttons=False,
    top_n=None,
    min_edge_weight=1,
    png_width=3000,
    png_height=3000,
    wait_ms=15000
):
    """
    Visualize causal network with pyvis (HTML + optional PNG).
    
    Based on causalis_ver_4_1.py plot_interactive_network().
    
    Args:
        G: NetworkX graph
        output_html: Output HTML file path
        output_png: Optional PNG output path (requires playwright)
        notebook: Display in Jupyter notebook
        gravity: Gravity strength (default: -500, negative = repulsion)
        width: HTML canvas width
        height: HTML canvas height
        show_buttons: Show interactive controls
        top_n: Keep only top N nodes by degree (None = all nodes)
        min_edge_weight: Minimum edge weight to display (default: 1)
        png_width: PNG width in pixels
        png_height: PNG height in pixels
        wait_ms: Wait time for physics stabilization (ms)
        
    Returns:
        dict: {'html': str, 'png': str or None}
        
    Example:
        >>> G = build_causal_network_from_clusters(df, top_n=100)
        >>> result = visualize_causal_network(
        ...     G,
        ...     output_html="network.html",
        ...     top_n=50,
        ...     min_edge_weight=2
        ... )
        >>> print(result['html'])
    """
    import copy
    
    # Convert Path objects to strings
    output_html = str(output_html)
    if output_png is not None:
        output_png = str(output_png)
    
    # Filter graph by top_n nodes and min_edge_weight
    G_filtered = G.copy()
    
    if top_n is not None and G_filtered.number_of_nodes() > top_n:
        # Get top N nodes by degree
        degrees = dict(G_filtered.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node for node, _ in top_nodes]
        G_filtered = G_filtered.subgraph(top_node_ids).copy()
        print(f"  Filtered to top {top_n} nodes (by degree)")
    
    if min_edge_weight > 1:
        # Remove edges below threshold
        edges_to_remove = []
        for u, v, data in G_filtered.edges(data=True):
            weight = data.get('frequency', data.get('value', 1))
            if weight < min_edge_weight:
                edges_to_remove.append((u, v))
        
        G_filtered.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        isolated = list(nx.isolates(G_filtered))
        G_filtered.remove_nodes_from(isolated)
        
        print(f"  Filtered edges (weight >= {min_edge_weight}): {G_filtered.number_of_edges()} edges")
        print(f"  Nodes after filtering: {G_filtered.number_of_nodes()}")
    
    if G_filtered.number_of_nodes() == 0:
        logger.warning("Graph is empty after filtering!")
        return {'html': None, 'png': None}
    
    # Options matching causalis_ver_4_1.py
    default_options = """
    var options = {
      "nodes": {
        "font": {
          "size": 16
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "font": {
          "size": 12
        },
        "smooth": {
          "forceDirection": "none"
        }
      },
      "physics": {
        "barnesHut": {
          "centralGravity": 0.3,
          "springLength": 100,
          "springConstant": 0.05,
          "avoidOverlap": 0.3
        },
        "minVelocity": 0.75
      }
    }
    """
    
    # Generate HTML
    print(f"\nGenerating interactive network...")
    print(f"  Nodes: {G_filtered.number_of_nodes()}")
    print(f"  Edges: {G_filtered.number_of_edges()}")
    
    html_path = plot_interactive_network_pyvis(
        G_filtered,
        output_html=output_html,
        notebook=notebook,
        show_buttons=show_buttons,
        only_physics_buttons=True,
        gravity=gravity,
        options=default_options,
        width=width,
        height=height
    )
    print(f"  ✓ HTML saved: {html_path}")
    
    result = {'html': html_path, 'png': None}
    
    # Export PNG if requested
    if output_png:
        png_path = export_pyvis_to_png(
            html_path=html_path,
            png_path=output_png,
            width=png_width,
            height=png_height,
            wait_ms=wait_ms
        )
        result['png'] = png_path
    
    return result

