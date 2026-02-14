#!/usr/bin/env python3
"""
Causal Event Clustering Module

This module provides two event clustering strategies:
1. Role-based Event Embedding: Role-aware embedding clustering based on semantic role labeling structure (ARG0/V/ARG1)
2. Phrase-based: Direct embedding clustering on raw phrases

Main features:
- Support for DP-Means, K-Means, and HDBSCAN clustering algorithms
- Automatic filtering of procedural statements
- Support for merging small clusters
- Generation of causal narratives

Usage example:
    from causal_narrative.event_clustering import EventClusterer
    
    # Create clusterer
    clusterer = EventClusterer(
        method='role_based',  # 'role_based' or 'phrase'
        algorithm='dpmeans',
        delta=0.2
    )
    
    # Run clustering
    results = clusterer.fit_transform(df)
    
    # Export results
    clusterer.export_results(output_dir)

Author: Causal Narrative Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import re
import gc
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import embedding functions from embedding module
from causal_narrative.embedding import (
    load_embedder,
    generate_role_based_embeddings,
    generate_phrase_embeddings,
    role_based_pca,
    apply_role_based_pca,
    phrase_pca,
)


# =============================================================================
# Constants
# =============================================================================

# Default embedding model
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Default PCA dimensions
DEFAULT_PCA_DIM_A0 = 10  # ARG0 (Subject/Agent)
DEFAULT_PCA_DIM_V = 5    # Verb
DEFAULT_PCA_DIM_A1 = 10  # ARG1 (Object/Patient)
DEFAULT_TEXT_PCA_DIM = 15  # PCA dimension for phrase-based clustering

# DP-Means default parameters
DEFAULT_DPMEANS_DELTA = 0.2
DEFAULT_DPMEANS_BATCH_SIZE = 100

# Minimum cluster size
DEFAULT_MIN_CLUSTER_SIZE = 5


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClusteringConfig:
    """
    Clustering configuration class
    
    Stores all clustering-related configuration parameters.
    
    Attributes:
        method: Clustering method, 'role_based' (Role-based Event Embedding) or 'phrase' (phrase-level)
        algorithm: Clustering algorithm, 'dpmeans', 'kmeans', or 'hdbscan'
        model_name: Sentence Transformer model name
        dim_a0: PCA dimension for ARG0 role
        dim_v: PCA dimension for V (verb) role
        dim_a1: PCA dimension for ARG1 role
        phrase_pca_dim: PCA dimension for phrase-based clustering
        delta: DP-Means delta threshold
        batch_size: DP-Means batch size
        n_clusters: Number of clusters for K-Means
        min_cluster_size: HDBSCAN minimum cluster size / threshold for merging small clusters
        random_state: Random seed
    """
    method: str = 'role_based'  # 'role_based' or 'phrase'
    algorithm: str = 'dpmeans'  # 'dpmeans', 'kmeans', 'hdbscan'
    model_name: str = DEFAULT_MODEL_NAME
    dim_a0: int = DEFAULT_PCA_DIM_A0
    dim_v: int = DEFAULT_PCA_DIM_V
    dim_a1: int = DEFAULT_PCA_DIM_A1
    phrase_pca_dim: int = DEFAULT_TEXT_PCA_DIM
    delta: float = DEFAULT_DPMEANS_DELTA
    batch_size: int = DEFAULT_DPMEANS_BATCH_SIZE
    n_clusters: Optional[int] = None  # for kmeans
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    random_state: int = 42


@dataclass
class ClusterInfo:
    """
    Information for a single cluster
    
    Attributes:
        cluster_id: Cluster ID
        name: Representative cluster name (most common text)
        size: Cluster size
        texts: All texts contained in the cluster
        text_counts: Occurrence count for each text
    """
    cluster_id: int
    name: str
    size: int
    texts: List[str] = field(default_factory=list)
    text_counts: Counter = field(default_factory=Counter)


@dataclass
class ClusteringResult:
    """
    Clustering result
    
    Attributes:
        labels: Cluster label array
        cluster_stats: Statistical information for each cluster
        n_clusters: Number of clusters
        method: Clustering method used
        algorithm: Clustering algorithm used
    """
    labels: np.ndarray
    cluster_stats: Dict[int, ClusterInfo]
    n_clusters: int
    method: str
    algorithm: str


# =============================================================================
# SRL Parsing Functions
# =============================================================================

def parse_srl_json(srl_str: str) -> Tuple[str, str, str]:
    """
    Parse SRL JSON string to extract A0, V, A1 components
    
    Extract semantic roles from AllenNLP or spaCy generated SRL JSON.
    
    Args:
        srl_str: SRL JSON string, format like:
            {"verbs": [{"verb": "run", "description": "[ARG0: the dog] [V: runs] [ARG1: fast]"}]}
    
    Returns:
        Tuple[str, str, str]: (A0, V, A1) tuple
            - A0: Agent (Subject)
            - V: Verb
            - A1: Patient (Object)
        
        Returns ('', '', '') if parsing fails
    
    Example:
        >>> parse_srl_json('{"verbs": [{"verb": "run", "description": "[ARG0: dog] [V: runs] [ARG1: fast]"}]}')
        ('dog', 'runs', 'fast')
    """
    if pd.isna(srl_str) or not srl_str:
        return '', '', ''
    
    try:
        srl_data = json.loads(srl_str)
    except json.JSONDecodeError:
        return '', '', ''
    
    verbs = srl_data.get('verbs', [])
    if not verbs:
        return '', '', ''
    
    verb_frame = verbs[0]  # Use the first verb frame
    description = verb_frame.get('description', '')
    verb = verb_frame.get('verb', '')
    
    # Extract components using regular expressions
    a0_match = re.search(r'\[ARG0:\s*([^\]]+)\]', description)
    v_match = re.search(r'\[V:\s*([^\]]+)\]', description)
    a1_match = re.search(r'\[ARG1:\s*([^\]]+)\]', description)
    
    a0 = a0_match.group(1).strip() if a0_match else ''
    v = v_match.group(1).strip() if v_match else verb
    a1 = a1_match.group(1).strip() if a1_match else ''
    
    return a0, v, a1


def is_valid_srl(a0: str, v: str, a1: str) -> bool:
    """
    Check if SRL structure is valid
    
    A valid SRL structure must have:
    1. Verb V must exist
    2. At least one of A0 or A1 must exist
    
    Args:
        a0: ARG0 agent
        v: Verb
        a1: ARG1 patient
    
    Returns:
        bool: True if structure is valid
    
    Example:
        >>> is_valid_srl('dog', 'runs', '')
        True
        >>> is_valid_srl('', '', '')
        False
        >>> is_valid_srl('', 'runs', '')
        False
    """
    if not v or v.strip() == '':
        return False
    has_a0 = bool(a0 and a0.strip())
    has_a1 = bool(a1 and a1.strip())
    return has_a0 or has_a1


def make_svo_text(a0: str, v: str, a1: str) -> str:
    """
    Combine A0, V, A1 into SVO text
    
    Args:
        a0: ARG0 agent
        v: Verb
        a1: ARG1 patient
    
    Returns:
        str: Combined SVO text, parts separated by spaces
    
    Example:
        >>> make_svo_text('dog', 'runs', 'fast')
        'dog runs fast'
        >>> make_svo_text('', 'runs', 'fast')
        'runs fast'
    """
    parts = [p for p in [a0, v, a1] if p and p.strip()]
    return ' '.join(parts)


# =============================================================================
# Embedding Generation Functions
# =============================================================================

# Note: The following functions have been moved to causal_narrative.embedding:
# - load_embedder()
# - generate_role_based_embeddings()
# - generate_phrase_embeddings()
# - role_based_pca()
# - apply_role_based_pca()
# - phrase_pca()
# They are imported at the top of this file for backward compatibility.


# =============================================================================
# Clustering Algorithms
# =============================================================================

def run_dpmeans(
    embeddings: np.ndarray,
    delta: float = DEFAULT_DPMEANS_DELTA,
    batch_size: int = DEFAULT_DPMEANS_BATCH_SIZE,
    random_state: int = 42
) -> Tuple[np.ndarray, Any]:
    """
    Run DP-Means clustering
    
    DP-Means is a clustering algorithm that automatically determines the number of clusters.
    A new cluster is created when the distance from a sample to the nearest cluster center exceeds delta.
    
    Args:
        embeddings: Input with shape (n_samples, n_features)
        delta: Distance threshold, smaller value means more clusters
        batch_size: Mini-batch size
        random_state: Random seed
    
    Returns:
        Tuple[np.ndarray, MiniBatchDPMeans]:
            - labels: Cluster labels
            - model: Trained DP-Means model
    
    Raises:
        ImportError: If pdc_dp_means is not installed
    """
    from pdc_dp_means import MiniBatchDPMeans
    
    print(f"    DP-Means (delta={delta}, batch_size={batch_size}, "
          f"n={embeddings.shape[0]}, dim={embeddings.shape[1]})...")
    
    # Ensure data type and memory layout are consistent (to match reference implementation)
    X = np.ascontiguousarray(embeddings, dtype=np.float64)
    
    dpmeans = MiniBatchDPMeans(
        n_clusters=1,
        batch_size=batch_size,
        delta=delta,
        max_iter=30,
        n_init=1,
        random_state=random_state,
        verbose=0
    )
    
    # Run in background and show progress
    result = [None]
    done = threading.Event()
    
    def fit_predict():
        result[0] = dpmeans.fit_predict(X)
        done.set()
    
    thread = threading.Thread(target=fit_predict)
    thread.start()
    
    with tqdm(desc="      DP-Means", unit="iter", leave=True) as pbar:
        while not done.is_set():
            time.sleep(0.5)
            pbar.update(1)
    
    thread.join()
    labels = result[0]
    
    n_clusters = len(np.unique(labels))
    print(f"      ✓ {n_clusters} clusters")
    
    return labels, dpmeans


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, Any]:
    """
    Run K-Means clustering
    
    Args:
        embeddings: Input with shape (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Tuple[np.ndarray, KMeans]:
            - labels: Cluster labels
            - model: Trained K-Means model
    """
    from sklearn.cluster import KMeans
    
    print(f"    K-Means (k={n_clusters}, n={embeddings.shape[0]})...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    
    labels = kmeans.fit_predict(embeddings)
    print(f"      ✓ {n_clusters} clusters")
    
    return labels, kmeans


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = 'euclidean',
    cluster_selection_epsilon: float = 0.0
) -> Tuple[np.ndarray, Any]:
    """
    Run HDBSCAN clustering
    
    HDBSCAN is a density-based clustering algorithm that automatically determines the number of clusters.
    
    Args:
        embeddings: Input with shape (n_samples, n_features)
        min_cluster_size: Minimum cluster size (default: 5)
        min_samples: Minimum samples for core points. If None, defaults to min_cluster_size (default: None)
        metric: Distance metric to use. Options:
                - 'euclidean' (default): Euclidean distance
                - 'cosine': Cosine distance (uses precomputed distance matrix)
                - 'manhattan': Manhattan distance
                - Other metrics supported by sklearn
        cluster_selection_epsilon: Distance threshold for cluster merging (default: 0.0)
    
    Returns:
        Tuple[np.ndarray, HDBSCAN]:
            - labels: Cluster labels (-1 indicates noise points)
            - model: Trained HDBSCAN model
    
    Raises:
        ImportError: If hdbscan is not installed
    
    Note:
        When using 'cosine' metric, the function automatically computes a precomputed
        distance matrix, which may require more memory for large datasets.
    
    Example:
        >>> # Euclidean distance
        >>> labels, model = run_hdbscan(embeddings, min_cluster_size=5, min_samples=3)
        >>> 
        >>> # Cosine distance (good for text embeddings)
        >>> labels, model = run_hdbscan(embeddings, min_cluster_size=5, metric='cosine')
        >>> 
        >>> # Check results
        >>> n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        >>> n_noise = (labels == -1).sum()
    """
    from loguru import logger
    import hdbscan
    
    # Set min_samples to min_cluster_size if not specified
    if min_samples is None:
        min_samples = min_cluster_size
    
    logger.info(
        f"Running HDBSCAN: min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples}, metric={metric}, "
        f"n_samples={embeddings.shape[0]}"
    )
    
    # Handle cosine metric specially
    if metric == 'cosine':
        # For cosine distance, we need to use precomputed distance matrix
        from sklearn.metrics.pairwise import cosine_distances
        
        logger.debug("Computing cosine distance matrix...")
        distance_matrix = cosine_distances(embeddings)
        
        # Ensure the distance matrix is float64 (required by HDBSCAN)
        distance_matrix = distance_matrix.astype(np.float64)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            cluster_selection_epsilon=cluster_selection_epsilon,
            core_dist_n_jobs=-1
        )
        
        labels = clusterer.fit_predict(distance_matrix)
    else:
        # For other metrics, use directly
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            core_dist_n_jobs=-1
        )
        
        labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    logger.info(
        f"HDBSCAN completed: {n_clusters} clusters, {n_noise} noise points "
        f"({n_noise/len(labels)*100:.1f}%)"
    )
    
    return labels, clusterer


# =============================================================================
# Clustering Statistics Functions
# =============================================================================

def count_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """
    Count the number of samples in each cluster
    
    This is a simple helper function for quickly viewing cluster size distribution.
    
    Args:
        labels: Cluster label array (shape: [n_samples])
    
    Returns:
        Dict[int, int]: Dictionary where key is cluster ID, value is number of samples in the cluster
    
    Example:
        >>> labels = np.array([0, 0, 1, 1, 1, 2])
        >>> count_cluster_sizes(labels)
        {0: 2, 1: 3, 2: 1}
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def generate_cluster_names_from_srl(
    labels: np.ndarray,
    srl_results: List[Dict[str, Any]],
    fallback_texts: Optional[List[str]] = None
) -> Dict[int, str]:
    """
    Generate cluster names from SRL results (based on most frequent SVO)
    
    Args:
        labels: Cluster label array
        srl_results: List of SRL results (dictionary format)
        fallback_texts: Optional, fallback texts to use when SVO is empty
        
    Returns:
        Dictionary where key is cluster ID, value is representative name for that cluster
        
    Example:
        >>> labels = np.array([0, 0, 1])
        >>> srl_results = [
        ...     {"verbs": [{"description": "[ARG0: dog] [V: chase] [ARG1: cat]"}]},
        ...     {"verbs": [{"description": "[ARG0: dog] [V: chase] [ARG1: cat]"}]},
        ...     {"verbs": [{"description": "[V: run]"}]}
        ... ]
        >>> generate_cluster_names_from_srl(labels, srl_results)
        {0: 'dog chase cat', 1: 'run'}
    """
    from causal_narrative import extract_roles
    from collections import Counter
    
    cluster_names = {}
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_srl_dicts = [srl_results[i] for i in range(len(srl_results)) if mask[i]]
        
        # Extract SVO texts
        svo_texts = []
        for i, srl_dict in enumerate(cluster_srl_dicts):
            roles = extract_roles(srl_dict)
            # Use 'or' to handle None values
            svo = ' '.join([
                (roles.get('ARG0') or '').strip(),
                (roles.get('V') or '').strip(),
                (roles.get('ARG1') or '').strip()
            ]).strip()
            
            # If SVO is empty, try using fallback_texts
            if not svo and fallback_texts is not None:
                original_idx = np.where(mask)[0][i]
                if original_idx < len(fallback_texts):
                    svo = fallback_texts[original_idx]
            
            if svo:
                svo_texts.append(svo)
        
        # Use most frequent SVO as name
        if svo_texts:
            most_common = Counter(svo_texts).most_common(1)[0][0]
            cluster_names[cluster_id] = most_common
        else:
            cluster_names[cluster_id] = f'Cluster_{cluster_id}'
    
    return cluster_names


def generate_cluster_names_from_texts(
    labels: np.ndarray,
    texts: List[str]
) -> Dict[int, str]:
    """
    Generate cluster names from texts (based on most frequent text)
    
    Args:
        labels: Cluster label array
        texts: List of texts
        
    Returns:
        Dictionary where key is cluster ID, value is representative name for that cluster
        
    Example:
        >>> labels = np.array([0, 0, 1, 1])
        >>> texts = ['stop job', 'stop job', 'run fast', 'run']
        >>> generate_cluster_names_from_texts(labels, texts)
        {0: 'stop job', 1: 'run fast'}
    """
    from collections import Counter
    
    cluster_names = {}
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        
        # Use most frequent text as name
        if cluster_texts:
            most_common = Counter(cluster_texts).most_common(1)[0][0]
            cluster_names[cluster_id] = most_common
        else:
            cluster_names[cluster_id] = f'Cluster_{cluster_id}'
    
    return cluster_names


def merge_small_clusters(
    labels: np.ndarray,
    min_size: int = DEFAULT_MIN_CLUSTER_SIZE
) -> Tuple[np.ndarray, int]:
    """
    Merge small clusters
    
    Merge clusters with size < min_size into an "OTHER" cluster.
    
    Args:
        labels: Original cluster labels
        min_size: Minimum cluster size
    
    Returns:
        Tuple[np.ndarray, int]:
            - new_labels: Merged labels
            - merged_count: Number of small clusters merged
    """
    label_counts = Counter(labels)
    small_clusters = {cid for cid, count in label_counts.items() if count < min_size}
    
    if not small_clusters:
        return labels, 0
    
    other_cluster_id = max(labels) + 1
    new_labels = np.array([
        other_cluster_id if l in small_clusters else l 
        for l in labels
    ])
    
    return new_labels, len(small_clusters)


def get_cluster_stats(
    texts: List[str],
    labels: np.ndarray,
    svo_texts: Optional[List[str]] = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
) -> Tuple[Dict[int, ClusterInfo], np.ndarray]:
    """
    Get clustering statistics and merge small clusters
    
    Args:
        texts: List of original texts
        labels: Cluster labels
        svo_texts: List of SVO texts (for naming clusters), use texts if None
        min_cluster_size: Minimum cluster size
    
    Returns:
        Tuple[Dict[int, ClusterInfo], np.ndarray]:
            - cluster_stats: Statistics for each cluster
            - merged_labels: Labels after merging small clusters
    """
    # Merge small clusters
    merged_labels, merged_count = merge_small_clusters(labels, min_cluster_size)
    
    if merged_count > 0:
        print(f"      Merged {merged_count} small clusters (size < {min_cluster_size})")
    
    # Collect statistics
    cluster_data = {}
    name_texts = svo_texts if svo_texts is not None else texts
    
    for i, label in enumerate(merged_labels):
        if label not in cluster_data:
            cluster_data[label] = {'texts': [], 'name_texts': []}
        cluster_data[label]['texts'].append(texts[i])
        cluster_data[label]['name_texts'].append(name_texts[i])
    
    # Generate ClusterInfo
    cluster_stats = {}
    for cid, data in cluster_data.items():
        # Use most common non-empty text as name
        non_empty = [n for n in data['name_texts'] if n and n.strip()]
        
        if non_empty:
            text_counts = Counter(non_empty)
            most_common = text_counts.most_common(1)[0][0]
        else:
            non_empty_texts = [t for t in data['texts'] if t and t.strip()]
            if non_empty_texts:
                text_counts = Counter(non_empty_texts)
                most_common = text_counts.most_common(1)[0][0]
                if len(most_common) > 80:
                    most_common = most_common[:80] + "..."
            else:
                most_common = f"[empty_{cid}]"
        
        cluster_stats[cid] = ClusterInfo(
            cluster_id=cid,
            name=most_common,
            size=len(data['texts']),
            texts=data['texts'],
            text_counts=Counter(data['texts'])
        )
    
    return cluster_stats, merged_labels


def compute_cluster_centers(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute center vectors for each cluster
    
    Args:
        embeddings: Embedding matrix
        labels: Cluster labels
    
    Returns:
        Tuple[np.ndarray, List[int]]:
            - centers: Cluster center matrix
            - cluster_ids: List of corresponding cluster IDs
    """
    unique_labels = np.unique(labels)
    centers = []
    cluster_ids = []
    
    for label in unique_labels:
        mask = labels == label
        center = embeddings[mask].mean(axis=0)
        centers.append(center)
        cluster_ids.append(label)
    
    return np.array(centers), cluster_ids


def assign_to_nearest_cluster(
    text_embeddings: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_ids: List[int]
) -> np.ndarray:
    """
    Assign texts to nearest cluster centers
    
    Use cosine similarity to find nearest cluster center for each text.
    
    Args:
        text_embeddings: Embeddings of texts to be assigned
        cluster_centers: Cluster center vectors
        cluster_ids: List of cluster IDs
    
    Returns:
        np.ndarray: Assigned cluster labels
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"    Assigning {text_embeddings.shape[0]} texts to {len(cluster_ids)} clusters...")
    
    similarities = cosine_similarity(text_embeddings, cluster_centers)
    nearest_indices = np.argmax(similarities, axis=1)
    assigned_labels = np.array([cluster_ids[i] for i in nearest_indices])
    
    return assigned_labels


# =============================================================================
# Export Functions
# =============================================================================

def export_cluster_details(
    cluster_stats: Dict[int, ClusterInfo],
    output_file: Path,
    title: str,
    total_texts: int
) -> None:
    """
    Export cluster detailed information to txt file
    
    Args:
        cluster_stats: Cluster statistics information
        output_file: Output file path
        title: Report title
        total_texts: Total number of texts
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"{title}\n")
        f.write(f"Total texts: {total_texts}\n")
        f.write(f"Number of clusters: {len(cluster_stats)}\n")
        f.write("=" * 100 + "\n\n")
        
        # Sort by size
        sorted_clusters = sorted(
            cluster_stats.items(),
            key=lambda x: x[1].size,
            reverse=True
        )
        
        for cluster_id, info in sorted_clusters:
            f.write(f"\n{'='*80}\n")
            f.write(f"Cluster {cluster_id} (size={info.size})\n")
            f.write(f"Representative name: {info.name}\n")
            f.write(f"{'='*80}\n")
            f.write("Contained texts:\n")
            
            for i, (text, count) in enumerate(info.text_counts.most_common(), 1):
                f.write(f"  {i}. [{count}x] {text}\n")
            
            f.write("-" * 80 + "\n")
    
    print(f"  Saved: {output_file.name}")


# =============================================================================
# Main Clustering Class
# =============================================================================

class EventClusterer:
    """
    Causal event clusterer
    
    Supports two types of clustering methods:
    - role_based: Role-based Event Embedding clustering based on semantic role labeling
    - phrase: Phrase-level embedding clustering
    
    Supports three types of clustering algorithms:
    - dpmeans: Automatically determine number of clusters
    - kmeans: Specify number of clusters
    - hdbscan: Density-based clustering
    
    Attributes:
        config: Clustering configuration
        embedder: Sentence Transformer model
        embed_dim: Embedding dimension
        cause_stats: Cause cluster statistics
        effect_stats: Effect cluster statistics
        df_result: Clustering result DataFrame
    
    Example:
        >>> clusterer = EventClusterer(
        ...     method='role_based',
        ...     algorithm='dpmeans',
        ...     delta=0.2
        ... )
        >>> results = clusterer.fit_transform(df)
        >>> clusterer.export_results(Path('./output'))
    """
    
    def __init__(
        self,
        method: str = 'role_based',
        algorithm: str = 'dpmeans',
        model_name: str = DEFAULT_MODEL_NAME,
        dim_a0: int = DEFAULT_PCA_DIM_A0,
        dim_v: int = DEFAULT_PCA_DIM_V,
        dim_a1: int = DEFAULT_PCA_DIM_A1,
        phrase_pca_dim: int = DEFAULT_TEXT_PCA_DIM,
        delta: float = DEFAULT_DPMEANS_DELTA,
        batch_size: int = DEFAULT_DPMEANS_BATCH_SIZE,
        n_clusters: Optional[int] = None,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        random_state: int = 42
    ):
        """
        Initialize clusterer
        
        Args:
            method: Clustering method, 'role_based' (Role-based Event Embedding) or 'phrase'
            algorithm: Clustering algorithm, 'dpmeans', 'kmeans', or 'hdbscan'
            model_name: Sentence Transformer model name
            dim_a0: ARG0 PCA dimension (role_based only)
            dim_v: V PCA dimension (role_based only)
            dim_a1: ARG1 PCA dimension (role_based only)
            phrase_pca_dim: Phrase-level PCA dimension (phrase only)
            delta: DP-Means delta threshold
            batch_size: DP-Means batch size
            n_clusters: Number of clusters for K-Means
            min_cluster_size: Minimum cluster size
            random_state: Random seed
        """
        self.config = ClusteringConfig(
            method=method,
            algorithm=algorithm,
            model_name=model_name,
            dim_a0=dim_a0,
            dim_v=dim_v,
            dim_a1=dim_a1,
            phrase_pca_dim=phrase_pca_dim,
            delta=delta,
            batch_size=batch_size,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            random_state=random_state
        )
        
        self.embedder = None
        self.embed_dim = None
        self.cause_stats = None
        self.effect_stats = None
        self.df_result = None
        
        # PCA model (for text->svo mapping)
        self._cause_pca_models = None
        self._effect_pca_models = None
        self._cause_cluster_centers = None
        self._effect_cluster_centers = None
        self._cause_cluster_ids = None
        self._effect_cluster_ids = None
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        cause_col: str = 'cause',
        effect_col: str = 'effect',
        cause_srl_col: str = 'cause_srl',
        effect_srl_col: str = 'effect_srl'
    ) -> pd.DataFrame:
        """
        Execute clustering and return results
        
        Args:
            df: Input DataFrame, must contain cause/effect text and SRL columns
            cause_col: Cause text column name
            effect_col: Effect text column name
            cause_srl_col: Cause SRL column name
            effect_srl_col: Effect SRL column name
        
        Returns:
            pd.DataFrame: DataFrame with added clustering results, containing:
                - cause_srl_event: Cause event name obtained from SRL clustering
                - effect_srl_event: Effect event name obtained from SRL clustering
                - cause_text_event: Cause event name obtained from text clustering
                - effect_text_event: Effect event name obtained from text clustering
                - cause_event: Final merged cause cluster representative name (priority: srl_event > text_event > original text)
                - effect_event: Final merged effect cluster representative name (priority: srl_event > text_event > original text)
                - cause_cluster: Cause cluster ID
                - effect_cluster: Effect cluster ID
                - cause_cluster_type: Clustering type ('svo', 'text', 'text->svo', 'none')
                - effect_cluster_type: Clustering type
                - causal_narr: Causal narrative "cause_event -> effect_event"
        """
        cfg = self.config
        
        print("=" * 100)
        print(f"Event Clustering (method={cfg.method}, algorithm={cfg.algorithm})")
        print("=" * 100)
        
        # 1. Load model
        print("\n[1/5] Loading embedding model...")
        self.embedder = load_embedder(cfg.model_name)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"  Model: {cfg.model_name}, dim: {self.embed_dim}")
        
        # 2. Parse SRL
        print("\n[2/5] Parsing SRL...")
        df_filtered = df.copy().reset_index(drop=True)
        cause_data, effect_data = [], []
        
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="  Parsing"):
            a0, v, a1 = parse_srl_json(row.get(cause_srl_col, ''))
            cause_data.append({
                'valid': bool(is_valid_srl(a0, v, a1)),
                'svo': make_svo_text(a0, v, a1)
            })
            
            a0, v, a1 = parse_srl_json(row.get(effect_srl_col, ''))
            effect_data.append({
                'valid': bool(is_valid_srl(a0, v, a1)),
                'svo': make_svo_text(a0, v, a1)
            })
        
        df_filtered['cause_valid_srl'] = [d['valid'] for d in cause_data]
        df_filtered['effect_valid_srl'] = [d['valid'] for d in effect_data]
        df_filtered['cause_svo'] = [d['svo'] for d in cause_data]
        df_filtered['effect_svo'] = [d['svo'] for d in effect_data]
        
        cause_valid = df_filtered['cause_valid_srl'].sum()
        effect_valid = df_filtered['effect_valid_srl'].sum()
        print(f"  Cause valid SRL: {cause_valid} ({100*cause_valid/len(df_filtered):.1f}%)")
        print(f"  Effect valid SRL: {effect_valid} ({100*effect_valid/len(df_filtered):.1f}%)")
        
        # 3 & 4. cluster
        if cfg.method == 'role_based':
            self._cluster_role_based(df_filtered, cause_col, effect_col, cause_srl_col, effect_srl_col)
        else:
            self._cluster_phrase_based(df_filtered, cause_col, effect_col)
        
        # 5. mergeresult
        print("\n[5/5] Merging results...")
        self._merge_results(df_filtered, cause_col, effect_col)
        
        self.df_result = df_filtered
        return df_filtered
    
    def _cluster_role_based(self, df, cause_col, effect_col, cause_srl_col, effect_srl_col):
        """execute Role-based Event Embedding cluster"""
        cfg = self.config
        
        # Cause
        print("\n[3/5] Cause clustering (role-based)...")
        df_cause_valid = df[df['cause_valid_srl']].copy().reset_index(drop=True)
        
        if len(df_cause_valid) > 0:
            print(f"  Valid samples: {len(df_cause_valid)}")
            
            # Embedding
            # Extract SRL result
            cause_srl_dicts = []
            for srl_str in df_cause_valid[cause_srl_col]:
                try:
                    srl_dict = json.loads(srl_str) if isinstance(srl_str, str) else srl_str
                    cause_srl_dicts.append(srl_dict)
                except:
                    cause_srl_dicts.append({})
            
            cause_emb = generate_role_based_embeddings(
                cause_srl_dicts, self.embedder, show_progress=True
            )
            
            # Generate SVO textfornaming
            from causal_narrative import extract_roles
            cause_svo_texts = []
            for srl_dict in cause_srl_dicts:
                roles = extract_roles(srl_dict)
                # Use or '' process None value
                svo = ' '.join([
                    (roles.get('ARG0') or '').strip(),
                    (roles.get('V') or '').strip(),
                    (roles.get('ARG1') or '').strip()
                ]).strip()
                cause_svo_texts.append(svo if svo else df_cause_valid.iloc[len(cause_svo_texts)][cause_col])
            
            # PCA
            cause_reduced, self._cause_pca_models = role_based_pca(
                cause_emb, cfg.dim_a0, cfg.dim_v, cfg.dim_a1, self.embed_dim, cfg.random_state
            )
            del cause_emb
            gc.collect()
            
            # Clustering
            cause_labels = self._run_clustering(cause_reduced)
            
            # Stats
            self.cause_stats, cause_labels = get_cluster_stats(
                df_cause_valid[cause_col].tolist(),
                cause_labels,
                cause_svo_texts,
                cfg.min_cluster_size
            )
            
            # saveclustercenter
            self._cause_cluster_centers, self._cause_cluster_ids = compute_cluster_centers(
                cause_reduced, cause_labels
            )
            
            df_cause_valid['cause_cluster'] = cause_labels
            df_cause_valid['cause_srl_event'] = [self.cause_stats[l].name for l in cause_labels]
            
            # Mergereturn to main DataFrame
            df.loc[df['cause_valid_srl'], 'cause_cluster'] = df_cause_valid['cause_cluster'].values
            df.loc[df['cause_valid_srl'], 'cause_srl_event'] = df_cause_valid['cause_srl_event'].values
            df.loc[df['cause_valid_srl'], 'cause_cluster_type'] = 'svo'
            
            del cause_reduced
            gc.collect()
            
            print(f"  Clusters: {len(self.cause_stats)}")
        
        # Effect
        print("\n[4/5] Effect clustering (role-based)...")
        df_effect_valid = df[df['effect_valid_srl']].copy().reset_index(drop=True)
        
        if len(df_effect_valid) > 0:
            print(f"  Valid samples: {len(df_effect_valid)}")
            
            # Extract SRL result
            effect_srl_dicts = []
            for srl_str in df_effect_valid[effect_srl_col]:
                try:
                    srl_dict = json.loads(srl_str) if isinstance(srl_str, str) else srl_str
                    effect_srl_dicts.append(srl_dict)
                except:
                    effect_srl_dicts.append({})
            
            effect_emb = generate_role_based_embeddings(
                effect_srl_dicts, self.embedder, show_progress=True
            )
            
            # Generate SVO textfornaming
            effect_svo_texts = []
            for srl_dict in effect_srl_dicts:
                roles = extract_roles(srl_dict)
                # Use or '' process None value
                svo = ' '.join([
                    (roles.get('ARG0') or '').strip(),
                    (roles.get('V') or '').strip(),
                    (roles.get('ARG1') or '').strip()
                ]).strip()
                effect_svo_texts.append(svo if svo else df_effect_valid.iloc[len(effect_svo_texts)][effect_col])
            
            effect_reduced, self._effect_pca_models = role_based_pca(
                effect_emb, cfg.dim_a0, cfg.dim_v, cfg.dim_a1, self.embed_dim, cfg.random_state
            )
            del effect_emb
            gc.collect()
            
            effect_labels = self._run_clustering(effect_reduced)
            
            self.effect_stats, effect_labels = get_cluster_stats(
                df_effect_valid[effect_col].tolist(),
                effect_labels,
                effect_svo_texts,
                cfg.min_cluster_size
            )
            
            self._effect_cluster_centers, self._effect_cluster_ids = compute_cluster_centers(
                effect_reduced, effect_labels
            )
            
            df_effect_valid['effect_cluster'] = effect_labels
            df_effect_valid['effect_srl_event'] = [self.effect_stats[l].name for l in effect_labels]
            
            df.loc[df['effect_valid_srl'], 'effect_cluster'] = df_effect_valid['effect_cluster'].values
            df.loc[df['effect_valid_srl'], 'effect_srl_event'] = df_effect_valid['effect_srl_event'].values
            df.loc[df['effect_valid_srl'], 'effect_cluster_type'] = 'svo'
            
            del effect_reduced
            gc.collect()
            
            print(f"  Clusters: {len(self.effect_stats)}")
        
        # processinvalid SRL data (assign tonearest SVO cluster)
        self._assign_invalid_srl_to_svo(df, cause_col, effect_col)
    
    def _cluster_phrase_based(self, df, cause_col, effect_col):
        """executephrase-levelcluster"""
        cfg = self.config
        
        # Cause
        print("\n[3/5] Cause clustering (phrase-based)...")
        cause_texts = df[cause_col].fillna('').tolist()
        cause_texts_clean = [t.strip() for t in cause_texts]
        valid_mask = [bool(t) for t in cause_texts_clean]
        
        if any(valid_mask):
            valid_texts = [t for t, m in zip(cause_texts_clean, valid_mask) if m]
            print(f"  Valid texts: {len(valid_texts)}")
            
            cause_emb = generate_phrase_embeddings(valid_texts, self.embedder)
            cause_reduced, _ = phrase_pca(cause_emb, cfg.phrase_pca_dim, cfg.random_state)
            del cause_emb
            gc.collect()
            
            cause_labels = self._run_clustering(cause_reduced)
            
            self.cause_stats, cause_labels = get_cluster_stats(
                valid_texts, cause_labels, None, cfg.min_cluster_size
            )
            
            # Map back to original DataFrame
            label_iter = iter(cause_labels)
            df['cause_cluster'] = [next(label_iter) if m else -1 for m in valid_mask]
            df['cause_text_event'] = df['cause_cluster'].apply(
                lambda l: self.cause_stats[l].name if l in self.cause_stats else ''
            )
            df['cause_cluster_type'] = ['phrase' if m else 'none' for m in valid_mask]
            
            del cause_reduced
            gc.collect()
            
            print(f"  Clusters: {len(self.cause_stats)}")
        
        # Effect
        print("\n[4/5] Effect clustering (phrase-based)...")
        effect_texts = df[effect_col].fillna('').tolist()
        effect_texts_clean = [t.strip() for t in effect_texts]
        valid_mask = [bool(t) for t in effect_texts_clean]
        
        if any(valid_mask):
            valid_texts = [t for t, m in zip(effect_texts_clean, valid_mask) if m]
            print(f"  Valid texts: {len(valid_texts)}")
            
            effect_emb = generate_phrase_embeddings(valid_texts, self.embedder)
            effect_reduced, _ = phrase_pca(effect_emb, cfg.phrase_pca_dim, cfg.random_state)
            del effect_emb
            gc.collect()
            
            effect_labels = self._run_clustering(effect_reduced)
            
            self.effect_stats, effect_labels = get_cluster_stats(
                valid_texts, effect_labels, None, cfg.min_cluster_size
            )
            
            label_iter = iter(effect_labels)
            df['effect_cluster'] = [next(label_iter) if m else -1 for m in valid_mask]
            df['effect_text_event'] = df['effect_cluster'].apply(
                lambda l: self.effect_stats[l].name if l in self.effect_stats else ''
            )
            df['effect_cluster_type'] = ['phrase' if m else 'none' for m in valid_mask]
            
            del effect_reduced
            gc.collect()
            
            print(f"  Clusters: {len(self.effect_stats)}")
    
    def _assign_invalid_srl_to_svo(self, df, cause_col, effect_col):
        """Assign invalid SRL data to nearest SVO clusters"""
        cfg = self.config
        
        # Cause
        if self._cause_cluster_centers is not None:
            invalid_mask = ~df['cause_valid_srl']
            invalid_texts = df.loc[invalid_mask, cause_col].fillna('').tolist()
            
            if invalid_texts:
                print(f"  Assigning {sum(invalid_mask)} invalid cause texts to SVO clusters...")
                
                invalid_emb = generate_phrase_embeddings(invalid_texts, self.embedder)
                invalid_slot_emb = np.tile(invalid_emb, (1, 3))
                del invalid_emb
                gc.collect()
                
                invalid_reduced = apply_role_based_pca(
                    invalid_slot_emb, self._cause_pca_models, self.embed_dim
                )
                del invalid_slot_emb
                gc.collect()
                
                assigned_labels = assign_to_nearest_cluster(
                    invalid_reduced, self._cause_cluster_centers, self._cause_cluster_ids
                )
                del invalid_reduced
                gc.collect()
                
                df.loc[invalid_mask, 'cause_cluster'] = assigned_labels
                df.loc[invalid_mask, 'cause_srl_event'] = [
                    self.cause_stats[l].name for l in assigned_labels
                ]
                df.loc[invalid_mask, 'cause_cluster_type'] = 'text->svo'
        
        # Effect
        if self._effect_cluster_centers is not None:
            invalid_mask = ~df['effect_valid_srl']
            invalid_texts = df.loc[invalid_mask, effect_col].fillna('').tolist()
            
            if invalid_texts:
                print(f"  Assigning {sum(invalid_mask)} invalid effect texts to SVO clusters...")
                
                invalid_emb = generate_phrase_embeddings(invalid_texts, self.embedder)
                invalid_slot_emb = np.tile(invalid_emb, (1, 3))
                del invalid_emb
                gc.collect()
                
                invalid_reduced = apply_role_based_pca(
                    invalid_slot_emb, self._effect_pca_models, self.embed_dim
                )
                del invalid_slot_emb
                gc.collect()
                
                assigned_labels = assign_to_nearest_cluster(
                    invalid_reduced, self._effect_cluster_centers, self._effect_cluster_ids
                )
                del invalid_reduced
                gc.collect()
                
                df.loc[invalid_mask, 'effect_cluster'] = assigned_labels
                df.loc[invalid_mask, 'effect_srl_event'] = [
                    self.effect_stats[l].name for l in assigned_labels
                ]
                df.loc[invalid_mask, 'effect_cluster_type'] = 'text->svo'
    
    def _run_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """runclusteralgorithm"""
        cfg = self.config
        
        if cfg.algorithm == 'dpmeans':
            labels, _ = run_dpmeans(embeddings, cfg.delta, cfg.batch_size, cfg.random_state)
        elif cfg.algorithm == 'kmeans':
            if cfg.n_clusters is None:
                raise ValueError("n_clusters must be specified for kmeans")
            labels, _ = run_kmeans(embeddings, cfg.n_clusters, cfg.random_state)
        elif cfg.algorithm == 'hdbscan':
            labels, _ = run_hdbscan(embeddings, cfg.min_cluster_size)
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")
        
        return labels
    
    def _merge_results(self, df, cause_col, effect_col):
        """Merge results and generate causal_narr"""
        # Fill missing values
        if 'cause_cluster' not in df.columns:
            df['cause_cluster'] = -1
        if 'effect_cluster' not in df.columns:
            df['effect_cluster'] = -1
        if 'cause_srl_event' not in df.columns:
            df['cause_srl_event'] = ''
        if 'effect_srl_event' not in df.columns:
            df['effect_srl_event'] = ''
        if 'cause_text_event' not in df.columns:
            df['cause_text_event'] = ''
        if 'effect_text_event' not in df.columns:
            df['effect_text_event'] = ''
        if 'cause_cluster_type' not in df.columns:
            df['cause_cluster_type'] = 'none'
        if 'effect_cluster_type' not in df.columns:
            df['effect_cluster_type'] = 'none'
        
        # Merge srl_event and text_event into final cause_event and effect_event
        # Priority: use srl_event, if empty use text_event, last fallback to original text
        df['cause_event'] = df.apply(
            lambda row: (row.get('cause_srl_event', '') or 
                        row.get('cause_text_event', '') or 
                        row[cause_col][:50] if pd.notna(row[cause_col]) else ''),
            axis=1
        )
        df['effect_event'] = df.apply(
            lambda row: (row.get('effect_srl_event', '') or 
                        row.get('effect_text_event', '') or 
                        row[effect_col][:50] if pd.notna(row[effect_col]) else ''),
            axis=1
        )
        
        # Generate causal_narr
        df['causal_narr'] = df.apply(
            lambda row: f"{row['cause_event']} -> {row['effect_event']}"
            if row['cause_event'] and row['effect_event'] else '',
            axis=1
        )
        
        # statistics
        print(f"\n  Cause cluster types:")
        print(f"    SVO: {(df['cause_cluster_type'] == 'svo').sum()}")
        print(f"    Text->SVO: {(df['cause_cluster_type'] == 'text->svo').sum()}")
        print(f"    Text: {(df['cause_cluster_type'] == 'text').sum()}")
        print(f"    None: {(df['cause_cluster_type'] == 'none').sum()}")
        
        print(f"\n  Effect cluster types:")
        print(f"    SVO: {(df['effect_cluster_type'] == 'svo').sum()}")
        print(f"    Text->SVO: {(df['effect_cluster_type'] == 'text->svo').sum()}")
        print(f"    Text: {(df['effect_cluster_type'] == 'text').sum()}")
        print(f"    None: {(df['effect_cluster_type'] == 'none').sum()}")
    
    def export_results(
        self,
        output_dir: Union[str, Path],
        prefix: str = ''
    ) -> None:
        """
        exportclusterresult
        
        Args:
            output_dir: outputdirectory
            prefix: filename prefix
        
        Outputs:
            - {prefix}_clustering.csv: completeclusterresult
            - {prefix}_causal_narr.csv: simplified version (core column)
            - {prefix}_cause_clusters.txt: cause clusterdetails
            - {prefix}_effect_clusters.txt: effect clusterdetails
            - {prefix}_causal_narr_details.txt: causal_narr details
            - {prefix}_causal_narr_frequency.txt: frequencystatistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cfg = self.config
        if not prefix:
            prefix = f"{cfg.method}_{cfg.algorithm}"
        
        print("\n" + "=" * 100)
        print("Exporting results")
        print("=" * 100)
        
        # CSV
        csv_path = output_dir / f'{prefix}_clustering.csv'
        self.df_result.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")
        
        # simplified version CSV
        simple_cols = ['cause', 'effect', 
                       'cause_srl_event', 'effect_srl_event',
                       'cause_text_event', 'effect_text_event',
                       'cause_event', 'effect_event', 'causal_narr',
                       'cause_cluster', 'effect_cluster', 
                       'cause_cluster_type', 'effect_cluster_type']
        simple_cols = [c for c in simple_cols if c in self.df_result.columns]
        simple_path = output_dir / f'{prefix}_causal_narr.csv'
        self.df_result[simple_cols].to_csv(simple_path, index=False)
        print(f"  Saved: {simple_path.name}")
        
        # clusterdetails
        if self.cause_stats:
            export_cluster_details(
                self.cause_stats,
                output_dir / f'{prefix}_cause_clusters.txt',
                'Cause Clusters',
                len(self.df_result)
            )
        
        if self.effect_stats:
            export_cluster_details(
                self.effect_stats,
                output_dir / f'{prefix}_effect_clusters.txt',
                'Effect Clusters',
                len(self.df_result)
            )
        
        # Causal Narr details
        self._export_causal_narr_details(output_dir, prefix)
        
        print("\n" + "=" * 100)
        print("Done!")
        print("=" * 100)
    
    def _export_causal_narr_details(self, output_dir: Path, prefix: str):
        """export causal_narr details"""
        # Collectdetails
        narr_details = {}
        for idx, row in self.df_result.iterrows():
            narr = row['causal_narr']
            if not narr:
                continue
            if narr not in narr_details:
                narr_details[narr] = []
            narr_details[narr].append((row['cause'], row['effect']))
        
        sorted_narrs = sorted(narr_details.items(), key=lambda x: len(x[1]), reverse=True)
        
        # detailsfile
        details_path = output_dir / f'{prefix}_causal_narr_details.txt'
        with open(details_path, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("CAUSAL NARRATIVE detailedinformation\n")
            f.write(f"unique count: {len(narr_details)}\n")
            f.write(f"total entries: {sum(len(v) for v in narr_details.values())}\n")
            f.write("=" * 120 + "\n\n")
            
            for rank, (narr, units) in enumerate(sorted_narrs, 1):
                f.write("=" * 120 + "\n")
                f.write(f"[Rank {rank}] {narr}\n")
                f.write(f"frequency: {len(units)}\n")
                f.write("=" * 120 + "\n")
                
                unit_counts = Counter(units)
                f.write(f"contain {len(unit_counts)} different types cause->effect for:\n")
                f.write("-" * 120 + "\n")
                
                for i, ((cause, effect), count) in enumerate(unit_counts.most_common(), 1):
                    f.write(f"\n  [{count}x] Unit {i}:\n")
                    f.write(f"    CAUSE:  {cause}\n")
                    f.write(f"    EFFECT: {effect}\n")
                f.write("\n")
        
        print(f"  Saved: {details_path.name}")
        
        # frequencyfile
        freq_path = output_dir / f'{prefix}_causal_narr_frequency.txt'
        with open(freq_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("CAUSAL NARRATIVE frequencystatistics\n")
            f.write(f"unique count: {len(narr_details)}\n")
            f.write(f"total entries: {sum(len(v) for v in narr_details.values())}\n")
            f.write("=" * 100 + "\n\n")
            
            for rank, (narr, units) in enumerate(sorted_narrs, 1):
                f.write(f"{rank:5d}. [{len(units):5d}x] {narr}\n")
        
        print(f"  Saved: {freq_path.name}")
        
        # Print Top 10
        print(f"\n  Top 10 Causal Narratives:")
        for rank, (narr, units) in enumerate(sorted_narrs[:10], 1):
            narr_display = narr[:70] + '...' if len(narr) > 70 else narr
            print(f"    {rank:2d}. [{len(units):4d}x] {narr_display}")


# =============================================================================
# command line interface
# =============================================================================

def main():
    """commandentries"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="causal eventcluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  # Role-based Event Embedding cluster
  python event_clustering.py --csv data.csv --output ./results --method role_based
  
  # phrase-levelcluster
  python event_clustering.py --csv data.csv --output ./results --method phrase
  
  # customparameter
  python event_clustering.py --csv data.csv --output ./results \\
      --method role_based --algorithm dpmeans --delta 0.2 \\
      --dim-a0 10 --dim-v 5 --dim-a1 10
        """
    )
    
    parser.add_argument('--csv', type=str, required=True, help='input CSV filepath')
    parser.add_argument('--output', type=str, required=True, help='outputdirectory')
    parser.add_argument('--method', type=str, choices=['role_based', 'phrase'],
                        default='role_based', help='clustermethod (default: role_based)')
    parser.add_argument('--algorithm', type=str, choices=['dpmeans', 'kmeans', 'hdbscan'],
                        default='dpmeans', help='clusteralgorithm (default: dpmeans)')
    parser.add_argument('--dim-a0', type=int, default=DEFAULT_PCA_DIM_A0,
                        help=f'A0 PCA dimension (default: {DEFAULT_PCA_DIM_A0})')
    parser.add_argument('--dim-v', type=int, default=DEFAULT_PCA_DIM_V,
                        help=f'V PCA dimension (default: {DEFAULT_PCA_DIM_V})')
    parser.add_argument('--dim-a1', type=int, default=DEFAULT_PCA_DIM_A1,
                        help=f'A1 PCA dimension (default: {DEFAULT_PCA_DIM_A1})')
    parser.add_argument('--phrase-pca', type=int, default=DEFAULT_TEXT_PCA_DIM,
                        help=f'phrase-level PCA dimension (default: {DEFAULT_TEXT_PCA_DIM})')
    parser.add_argument('--delta', type=float, default=DEFAULT_DPMEANS_DELTA,
                        help=f'DP-Means delta (default: {DEFAULT_DPMEANS_DELTA})')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters for K-Means (required for kmeans)')
    parser.add_argument('--min-cluster-size', type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                        help=f'minimumclustersize (default: {DEFAULT_MIN_CLUSTER_SIZE})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_DPMEANS_BATCH_SIZE,
                        help=f'DP-Means batch size (default: {DEFAULT_DPMEANS_BATCH_SIZE})')
    
    args = parser.parse_args()
    
    # loaddata
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv, dtype=str, low_memory=False)
    print(f"  Rows: {len(df)}")
    
    # Create clusterer
    clusterer = EventClusterer(
        method=args.method,
        algorithm=args.algorithm,
        dim_a0=args.dim_a0,
        dim_v=args.dim_v,
        dim_a1=args.dim_a1,
        phrase_pca_dim=args.phrase_pca,
        delta=args.delta,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        batch_size=args.batch_size
    )
    
    # runcluster
    clusterer.fit_transform(df)
    
    # exportresult
    clusterer.export_results(args.output)


if __name__ == "__main__":
    main()

