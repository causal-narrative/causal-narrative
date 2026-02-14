"""
Causal Narrative - A toolkit for extracting and analyzing causal narratives from text.

Main Features:
- Causal Relationship Detection (heuristic / LLM / BERT)
- Cause/Effect Span Extraction (pattern / LLM / BERT)
- Semantic Role Labeling (spaCy / AllenNLP)
- Event Clustering (DP-Means / K-Means / HDBSCAN)
- Causal Network Construction & Visualization

Usage:
    # Detection
    from causal_narrative import CausalDetector, CausalSpanExtractor
    detector = CausalDetector(method='heuristic')
    result = detector.detect("The storm caused flooding.")

    # Clustering
    from causal_narrative import EventClusterer
    clusterer = EventClusterer(method='role_based', algorithm='dpmeans', delta=0.2)
    results = clusterer.fit_transform(df)
    clusterer.export_results('./output')
"""

__version__ = "0.1.2"

# Core data models
from causal_narrative.models import (
    SentenceRecord,
    CausalDecision,
    CausalSpan,
    EventTuple,
    ClusteredEvent,
    CausalEdge,
    ClusterSummary,
    ValidationResult,
)

# Causal detection & span extraction (unified API)
from causal_narrative.detection import (
    CausalDetector,
    DetectionResult,
    DEFAULT_CAUSAL_PATTERNS,
    # BERT detection backend
    BERTCausalDetector,
    DEFAULT_BERT_DETECTION_MODEL,
)

from causal_narrative.extraction import (
    CausalSpanExtractor,
    SpanExtractionResult,
    DEFAULT_SPAN_PATTERNS,
    # BERT span extraction backend
    BERTSpanExtractor,
    DEFAULT_BERT_SPAN_EXTRACTION_MODEL,
)

# Datasets
from causal_narrative.datasets import (
    list_datasets,
    get_dataset_info,
    load_data,
)

# Semantic Role Labeling
from causal_narrative.semantic_role_labeling import (
    SpacySRL,
    AllenNLPSRL,
    get_srl,
    is_allennlp_available,
    extract_roles,
    has_verb,
    is_event_srl,
    process_to_dataframe,
)

# Event clustering
from causal_narrative.event_clustering import (
    EventClusterer,
    ClusteringConfig,
    ClusterInfo,
    ClusteringResult,
    # SRL parsing
    parse_srl_json,
    is_valid_srl,
    make_svo_text,
    # Embedding
    load_embedder,
    generate_role_based_embeddings,
    generate_phrase_embeddings,
    # PCA
    role_based_pca,
    phrase_pca,
    # Clustering algorithms
    run_dpmeans,
    run_kmeans,
    run_hdbscan,
    count_cluster_sizes,
    generate_cluster_names_from_srl,
    generate_cluster_names_from_texts,
    get_cluster_stats,
)

# Visualization
from causal_narrative.viz import (
    build_causal_network_from_clusters,
    visualize_causal_network,
    plot_interactive_network_pyvis,
    export_pyvis_to_png,
)

__all__ = [
    # Version
    "__version__",
    # Data models
    "SentenceRecord",
    "CausalDecision",
    "CausalSpan",
    "EventTuple",
    "ClusteredEvent",
    "CausalEdge",
    "ClusterSummary",
    "ValidationResult",
    # Datasets
    "list_datasets",
    "get_dataset_info",
    "load_data",
    # Semantic Role Labeling
    "SpacySRL",
    "AllenNLPSRL",
    "get_srl",
    "is_allennlp_available",
    "extract_roles",
    "has_verb",
    "process_to_dataframe",
    # Detection & Extraction (unified API)
    "CausalDetector",
    "CausalSpanExtractor",
    "DetectionResult",
    "SpanExtractionResult",
    "DEFAULT_CAUSAL_PATTERNS",
    "DEFAULT_SPAN_PATTERNS",
    # BERT backends
    "BERTCausalDetector",
    "BERTSpanExtractor",
    "DEFAULT_BERT_DETECTION_MODEL",
    "DEFAULT_BERT_SPAN_EXTRACTION_MODEL",
    # Clustering
    "EventClusterer",
    "ClusteringConfig",
    "ClusterInfo",
    "ClusteringResult",
    # SRL
    "parse_srl_json",
    "is_valid_srl",
    "make_svo_text",
    # Embedding
    "load_embedder",
    "generate_role_based_embeddings",
    "generate_phrase_embeddings",
    # PCA
    "role_based_pca",
    "phrase_pca",
    # Clustering algorithms
    "run_dpmeans",
    "run_kmeans",
    "run_hdbscan",
    "count_cluster_sizes",
    "get_cluster_stats",
    # Visualization
    "build_causal_network_from_clusters",
    "visualize_causal_network",
    "plot_interactive_network_pyvis",
    "export_pyvis_to_png",
]
