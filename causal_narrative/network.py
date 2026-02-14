"""
Causal narrative network: builder, filters, and statistics.

This module combines:
- CausalNetworkBuilder for constructing directed causal graphs
- Network filtering utilities (weight, degree, k-core, top-k)
- Network statistics and analysis functions
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from loguru import logger

from causal_narrative.models import CausalEdge, ClusterSummary


# =============================================================================
# Network Builder
# =============================================================================

class CausalNetworkBuilder:
    """Builder for causal narrative networks.

    Constructs a directed graph where nodes are event clusters and edges
    represent causal relationships between clusters.

    Attributes:
        graph: NetworkX directed graph
        cluster_summaries: Dictionary mapping cluster_id -> ClusterSummary
        edge_examples: Dictionary storing example causal pairs per edge
    """

    def __init__(self):
        """Initialize network builder."""
        self.graph = nx.DiGraph()
        self.cluster_summaries: Dict[int, ClusterSummary] = {}
        self.edge_examples: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(
            list
        )

    def add_causal_pair(
        self,
        cause_cluster_id: int,
        effect_cluster_id: int,
        cause_text: Optional[str] = None,
        effect_text: Optional[str] = None,
    ) -> None:
        """Add a causal relationship pair to the network."""
        if self.graph.has_edge(cause_cluster_id, effect_cluster_id):
            self.graph[cause_cluster_id][effect_cluster_id]["weight"] += 1
        else:
            self.graph.add_edge(cause_cluster_id, effect_cluster_id, weight=1)

        if cause_text and effect_text:
            edge_key = (cause_cluster_id, effect_cluster_id)
            self.edge_examples[edge_key].append((cause_text, effect_text))

    def add_causal_pairs_batch(
        self,
        pairs: List[Tuple[int, int]],
        texts: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Add multiple causal pairs."""
        logger.debug(f"Adding batch of {len(pairs)} causal pairs to network")
        for i, (cause_id, effect_id) in enumerate(pairs):
            cause_text = None
            effect_text = None

            if texts and i < len(texts):
                cause_text, effect_text = texts[i]

            self.add_causal_pair(cause_id, effect_id, cause_text, effect_text)

        logger.info(f"Added {len(pairs)} causal pairs to network (nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})")

    def set_cluster_summaries(self, summaries: List[ClusterSummary]) -> None:
        """Set cluster summaries as node attributes."""
        for summary in summaries:
            self.cluster_summaries[summary.cluster_id] = summary

            if not self.graph.has_node(summary.cluster_id):
                self.graph.add_node(summary.cluster_id)

            self.graph.nodes[summary.cluster_id].update(
                {
                    "label": summary.label or f"Cluster {summary.cluster_id}",
                    "size": summary.size,
                    "exemplars": summary.exemplars[:3],
                    "keywords": summary.keywords[:5],
                }
            )

        logger.info(f"Set summaries for {len(summaries)} clusters")

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        cause_col: str = "cause_cluster_id",
        effect_col: str = "effect_cluster_id",
        cause_text_col: Optional[str] = None,
        effect_text_col: Optional[str] = None,
    ) -> nx.DiGraph:
        """Build network from DataFrame."""
        logger.info(f"Building network from DataFrame with {len(df)} causal pairs...")
        logger.debug(f"Using columns: cause={cause_col}, effect={effect_col}")

        for idx, row in df.iterrows():
            cause_id = int(row[cause_col])
            effect_id = int(row[effect_col])

            cause_text = row.get(cause_text_col) if cause_text_col else None
            effect_text = row.get(effect_text_col) if effect_text_col else None

            self.add_causal_pair(cause_id, effect_id, cause_text, effect_text)
            
            if (idx + 1) % 1000 == 0:
                logger.debug(f"Processed {idx + 1}/{len(df)} pairs")

        logger.info(
            f"Network built successfully: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        return self.graph

    def get_edges(self, min_weight: int = 1) -> List[CausalEdge]:
        """Get edges as CausalEdge objects."""
        edges = []

        for source, target, data in self.graph.edges(data=True):
            weight = data.get("weight", 1)

            if weight < min_weight:
                continue

            cause_label = self.graph.nodes[source].get("label")
            effect_label = self.graph.nodes[target].get("label")

            edge_key = (source, target)
            example_pairs = self.edge_examples.get(edge_key, [])[:5]

            edge = CausalEdge(
                cause_cluster_id=source,
                effect_cluster_id=target,
                weight=weight,
                example_pairs=example_pairs,
                cause_label=cause_label,
                effect_label=effect_label,
            )

            edges.append(edge)

        return edges

    def export_to_csv(
        self,
        output_dir: Path,
        nodes_file: str = "nodes.csv",
        edges_file: str = "edges.csv",
    ) -> Tuple[Path, Path]:
        """Export network to CSV files."""
        logger.info(f"Exporting network to CSV files in {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export nodes
        nodes_data = []
        for node_id in self.graph.nodes():
            node_attrs = self.graph.nodes[node_id]

            nodes_data.append(
                {
                    "cluster_id": node_id,
                    "label": node_attrs.get("label", f"Cluster {node_id}"),
                    "size": node_attrs.get("size", 0),
                    "in_degree": self.graph.in_degree(node_id),
                    "out_degree": self.graph.out_degree(node_id),
                    "degree": self.graph.degree(node_id),
                    "keywords": ", ".join(node_attrs.get("keywords", [])),
                    "exemplars": " | ".join(node_attrs.get("exemplars", [])),
                }
            )

        nodes_df = pd.DataFrame(nodes_data)
        nodes_path = output_dir / nodes_file
        nodes_df.to_csv(nodes_path, index=False)
        logger.info(f"Exported {len(nodes_df)} nodes to {nodes_path}")

        # Export edges
        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edges_data.append(
                {
                    "source": source,
                    "target": target,
                    "weight": data.get("weight", 1),
                    "source_label": self.graph.nodes[source].get("label"),
                    "target_label": self.graph.nodes[target].get("label"),
                }
            )

        edges_df = pd.DataFrame(edges_data)
        edges_path = output_dir / edges_file
        edges_df.to_csv(edges_path, index=False)
        logger.info(f"Exported {len(edges_df)} edges to {edges_path}")

        return nodes_path, edges_path

    def export_to_graphml(self, output_path: Path) -> Path:
        """Export network to GraphML format."""
        logger.info(f"Exporting network to GraphML format: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for node_id in self.graph.nodes():
            attrs = self.graph.nodes[node_id]
            if "keywords" in attrs and isinstance(attrs["keywords"], list):
                attrs["keywords"] = ", ".join(attrs["keywords"])
            if "exemplars" in attrs and isinstance(attrs["exemplars"], list):
                attrs["exemplars"] = " | ".join(attrs["exemplars"])

        nx.write_graphml(self.graph, output_path)
        logger.info(f"Successfully exported network to {output_path}")

        return output_path

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get network summary statistics."""
        stats = {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_in_degree": sum(dict(self.graph.in_degree()).values())
            / max(self.graph.number_of_nodes(), 1),
            "avg_out_degree": sum(dict(self.graph.out_degree()).values())
            / max(self.graph.number_of_nodes(), 1),
        }

        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        if in_degrees:
            stats["max_in_degree"] = max(in_degrees)
            stats["max_out_degree"] = max(out_degrees)

        if self.graph.number_of_nodes() > 0:
            stats["is_weakly_connected"] = nx.is_weakly_connected(self.graph)
            stats["n_weakly_connected_components"] = nx.number_weakly_connected_components(
                self.graph
            )

        return stats


# =============================================================================
# Network Filtering
# =============================================================================

def filter_by_weight(
    graph: nx.DiGraph,
    min_weight: int = 2,
    inplace: bool = False,
) -> nx.DiGraph:
    """Filter edges by minimum weight."""
    if not inplace:
        graph = graph.copy()

    edges_to_remove = [
        (u, v) for u, v, data in graph.edges(data=True) if data.get("weight", 1) < min_weight
    ]

    graph.remove_edges_from(edges_to_remove)

    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)

    logger.info(
        f"Filtered by weight >= {min_weight}: "
        f"removed {len(edges_to_remove)} edges, {len(isolated)} isolated nodes"
    )

    return graph


def filter_by_degree(
    graph: nx.DiGraph,
    min_degree: int = 1,
    degree_type: str = "total",
    inplace: bool = False,
) -> nx.DiGraph:
    """Filter nodes by minimum degree."""
    if not inplace:
        graph = graph.copy()

    if degree_type == "in":
        degrees = dict(graph.in_degree())
    elif degree_type == "out":
        degrees = dict(graph.out_degree())
    else:
        degrees = dict(graph.degree())

    nodes_to_remove = [node for node, degree in degrees.items() if degree < min_degree]

    graph.remove_nodes_from(nodes_to_remove)

    logger.info(
        f"Filtered by {degree_type} degree >= {min_degree}: "
        f"removed {len(nodes_to_remove)} nodes"
    )

    return graph


def filter_top_k_nodes(
    graph: nx.DiGraph,
    k: int,
    criterion: str = "degree",
    inplace: bool = False,
) -> nx.DiGraph:
    """Keep only top-k nodes by a criterion."""
    if not inplace:
        graph = graph.copy()

    if criterion == "degree":
        scores = dict(graph.degree())
    elif criterion == "in_degree":
        scores = dict(graph.in_degree())
    elif criterion == "out_degree":
        scores = dict(graph.out_degree())
    elif criterion == "pagerank":
        try:
            scores = nx.pagerank(graph)
        except:
            logger.warning("PageRank failed, using degree instead")
            scores = dict(graph.degree())
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k_nodes = [node for node, score in sorted_nodes[:k]]

    nodes_to_remove = set(graph.nodes()) - set(top_k_nodes)
    graph.remove_nodes_from(nodes_to_remove)

    logger.info(
        f"Kept top {k} nodes by {criterion}: " f"removed {len(nodes_to_remove)} nodes"
    )

    return graph


def apply_kcore(
    graph: nx.DiGraph,
    k: int = 2,
    inplace: bool = False,
) -> nx.DiGraph:
    """Apply k-core decomposition to filter graph."""
    if not inplace:
        graph = graph.copy()

    undirected = graph.to_undirected()

    try:
        core_nodes = nx.k_core(undirected, k=k).nodes()

        nodes_to_remove = set(graph.nodes()) - set(core_nodes)
        graph.remove_nodes_from(nodes_to_remove)

        logger.info(
            f"Applied {k}-core filter: "
            f"kept {len(core_nodes)} nodes, removed {len(nodes_to_remove)} nodes"
        )

    except nx.NetworkXError as e:
        logger.warning(f"K-core decomposition failed: {e}")

    return graph


# =============================================================================
# Network Statistics & Analysis
# =============================================================================

def get_strongly_connected_components(
    graph: nx.DiGraph,
    min_size: int = 2,
) -> List[Set[int]]:
    """Get strongly connected components."""
    components = list(nx.strongly_connected_components(graph))
    components = [comp for comp in components if len(comp) >= min_size]
    components.sort(key=len, reverse=True)

    logger.info(
        f"Found {len(components)} strongly connected components " f"(min_size={min_size})"
    )

    return components


def get_weakly_connected_components(
    graph: nx.DiGraph,
    min_size: int = 2,
) -> List[Set[int]]:
    """Get weakly connected components."""
    components = list(nx.weakly_connected_components(graph))
    components = [comp for comp in components if len(comp) >= min_size]
    components.sort(key=len, reverse=True)

    logger.info(
        f"Found {len(components)} weakly connected components " f"(min_size={min_size})"
    )

    return components


def compute_network_stats(graph: nx.DiGraph) -> Dict[str, Any]:
    """Compute comprehensive network statistics."""
    logger.info(f"Computing network statistics for graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    stats = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
    }

    if graph.number_of_nodes() == 0:
        logger.warning("Graph has no nodes, returning basic stats")
        return stats

    in_degrees = [d for _, d in graph.in_degree()]
    out_degrees = [d for _, d in graph.out_degree()]
    degrees = [d for _, d in graph.degree()]

    stats.update(
        {
            "avg_in_degree": sum(in_degrees) / len(in_degrees),
            "avg_out_degree": sum(out_degrees) / len(out_degrees),
            "avg_degree": sum(degrees) / len(degrees),
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }
    )

    stats["is_strongly_connected"] = nx.is_strongly_connected(graph)
    stats["is_weakly_connected"] = nx.is_weakly_connected(graph)
    stats["n_strongly_connected_components"] = nx.number_strongly_connected_components(graph)
    stats["n_weakly_connected_components"] = nx.number_weakly_connected_components(graph)

    if graph.number_of_nodes() <= 1000:
        try:
            pagerank = nx.pagerank(graph)
            stats["avg_pagerank"] = sum(pagerank.values()) / len(pagerank)
            stats["max_pagerank_node"] = max(pagerank.items(), key=lambda x: x[1])[0]

            betweenness = nx.betweenness_centrality(graph)
            stats["avg_betweenness"] = sum(betweenness.values()) / len(betweenness)
            stats["max_betweenness_node"] = max(betweenness.items(), key=lambda x: x[1])[0]

        except Exception as e:
            logger.warning(f"Failed to compute centrality measures: {e}")

    weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]
    if weights:
        stats["total_weight"] = sum(weights)
        stats["avg_weight"] = sum(weights) / len(weights)
        stats["max_weight"] = max(weights)

    logger.info(f"Network statistics computed: {stats['n_nodes']} nodes, {stats['n_edges']} edges, density={stats['density']:.4f}")
    return stats


def find_cycles(graph: nx.DiGraph, max_length: Optional[int] = None) -> List[List[int]]:
    """Find cycles in the graph."""
    try:
        cycles = list(nx.simple_cycles(graph))

        if max_length:
            cycles = [c for c in cycles if len(c) <= max_length]

        cycles.sort(key=len)

        logger.info(f"Found {len(cycles)} cycles")
        return cycles

    except Exception as e:
        logger.error(f"Failed to find cycles: {e}")
        return []


def get_hub_nodes(
    graph: nx.DiGraph,
    top_k: int = 10,
    criterion: str = "out_degree",
) -> List[tuple]:
    """Get top hub nodes (high out-degree)."""
    if criterion == "out_degree":
        scores = dict(graph.out_degree())
    elif criterion == "pagerank":
        scores = nx.pagerank(graph)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]


def get_authority_nodes(
    graph: nx.DiGraph,
    top_k: int = 10,
    criterion: str = "in_degree",
) -> List[tuple]:
    """Get top authority nodes (high in-degree)."""
    if criterion == "in_degree":
        scores = dict(graph.in_degree())
    elif criterion == "pagerank":
        scores = nx.pagerank(graph)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]
