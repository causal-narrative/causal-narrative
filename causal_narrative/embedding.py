"""Sentence embedding using Sentence-BERT models."""

import pickle
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

# Default model name for embeddings
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """Sentence embedder using sentence-transformers.

    Provides efficient embedding generation with caching support.

    Attributes:
        model_name: Name of sentence-transformers model
        model: Loaded SentenceTransformer model
        cache_dir: Directory for caching embeddings
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize sentence embedder.

        Args:
            model_name: Sentence-transformers model identifier
            cache_dir: Directory for caching embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model_name = model_name
            self.model = SentenceTransformer(model_name, device=device)
            self.cache_dir = Path(cache_dir) if cache_dir else None

            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Loaded embedding model: {model_name} "
                f"(dimension: {self.model.get_sentence_embedding_dimension()})"
            )

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def get_embedding_dim(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings

    def embed_with_cache(
        self,
        texts: List[str],
        cache_key: str,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings with caching.

        Args:
            texts: List of texts to embed
            cache_key: Unique key for this embedding set
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Embeddings array
        """
        if not self.cache_dir:
            # No caching, just embed
            return self.embed(texts, batch_size, show_progress)

        cache_file = self.cache_dir / f"{cache_key}_embeddings.pkl"

        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                # Verify cache is valid
                if (
                    isinstance(cached_data, dict)
                    and cached_data.get("texts") == texts
                    and cached_data.get("model") == self.model_name
                ):
                    logger.info(f"Loaded embeddings from cache: {cache_file}")
                    return cached_data["embeddings"]
                else:
                    logger.warning("Cache invalid, regenerating embeddings")

            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embed(texts, batch_size, show_progress)

        # Save to cache
        try:
            cache_data = {
                "texts": texts,
                "embeddings": embeddings,
                "model": self.model_name,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved embeddings to cache: {cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return embeddings

    def embed_batch_with_ids(
        self,
        items: List[tuple],
        text_index: int = 1,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> tuple[List[any], np.ndarray]:
        """Embed texts with associated IDs.

        Args:
            items: List of tuples containing (id, text, ...) or similar
            text_index: Index of text in each tuple
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Tuple of (ids, embeddings)
        """
        if not items:
            return [], np.array([])

        # Extract texts and IDs
        ids = []
        texts = []

        for item in items:
            if isinstance(item, (list, tuple)) and len(item) > text_index:
                ids.append(item[0])
                texts.append(item[text_index])
            else:
                logger.warning(f"Skipping invalid item: {item}")

        # Generate embeddings
        embeddings = self.embed(texts, batch_size, show_progress)

        return ids, embeddings

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (n, d)
            embeddings2: Second set of embeddings (m, d), or None for self-similarity

        Returns:
            Similarity matrix of shape (n, m) or (n, n)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if embeddings2 is None:
            embeddings2 = embeddings1

        return cosine_similarity(embeddings1, embeddings2)

    def find_nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors for a query embedding.

        Args:
            query_embedding: Query embedding vector (d,)
            candidate_embeddings: Candidate embeddings (n, d)
            top_k: Number of neighbors to return

        Returns:
            Tuple of (indices, similarities) for top-k neighbors
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)
        similarities = similarities.flatten()

        # Get top-k
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]

        return top_indices, top_similarities

    def clear_cache(self) -> int:
        """Clear cached embeddings.

        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir or not self.cache_dir.exists():
            return 0

        deleted = 0
        for cache_file in self.cache_dir.glob("*_embeddings.pkl"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file: {e}")

        logger.info(f"Deleted {deleted} embedding cache files")
        return deleted


# =============================================================================
# Helper Functions for Loading Embedders
# =============================================================================

def load_embedder(model_name: str = DEFAULT_MODEL_NAME):
    """Load Sentence Transformer model.
    
    Args:
        model_name: Model name, e.g., 'all-MiniLM-L6-v2'
    
    Returns:
        SentenceTransformer: Loaded model instance
    
    Raises:
        ImportError: If sentence_transformers is not installed
    
    Example:
        >>> embedder = load_embedder('all-MiniLM-L6-v2')
        >>> embeddings = embedder.encode(['Hello world'])
    """
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = 128
    logger.info(
        f"Model loaded successfully, embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )
    return model


# =============================================================================
# Role-Based and Phrase Embedding Functions
# =============================================================================

def generate_role_based_embeddings(
    srl_results: List[Dict[str, Any]],
    embedder,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """Generate Role-based Event Embeddings.
    
    Encode each record's SRL parsing result (ARG0, V, ARG1) separately,
    then concatenate into a joint vector: [embed(ARG0), embed(V), embed(ARG1)]
    
    Args:
        srl_results: List of SRL parsing result dictionaries
        embedder: SentenceEmbedder or SentenceTransformer model instance
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
    
    Returns:
        np.ndarray: Embedding matrix with shape (n_samples, 3 * embed_dim)
    
    Example:
        >>> from causal_narrative import extract_roles
        >>> from causal_narrative.embedding import SentenceEmbedder
        >>> srl_results = [extract_roles(srl_dict) for srl_dict in srl_list]
        >>> embedder = SentenceEmbedder(model_name='all-MiniLM-L6-v2')
        >>> embeddings = generate_role_based_embeddings(
        ...     srl_results, embedder
        ... )
        >>> embeddings.shape
        (1000, 1152)  # 3 * 384
    """
    from causal_narrative import extract_roles
    
    logger.info(
        f"Generating role-based embeddings for {len(srl_results)} SRL results"
    )
    a0_list, v_list, a1_list = [], [], []
    
    for srl_dict in srl_results:
        roles = extract_roles(srl_dict)
        # Use 'or' to handle None values (extract_roles may return None)
        a0_list.append(roles.get('ARG0') or '')
        v_list.append(roles.get('V') or '')
        a1_list.append(roles.get('ARG1') or '')
    
    # Check if embedder is SentenceEmbedder or SentenceTransformer
    if isinstance(embedder, SentenceEmbedder):
        # Use SentenceEmbedder.embed() method
        if show_progress:
            print(f"    Embedding ARG0 ({len(a0_list)} texts)...")
        a0_emb = embedder.embed(
            a0_list, batch_size=batch_size, show_progress=show_progress
        ).astype(np.float32)
        
        if show_progress:
            print(f"    Embedding V ({len(v_list)} texts)...")
        v_emb = embedder.embed(
            v_list, batch_size=batch_size, show_progress=show_progress
        ).astype(np.float32)
        
        if show_progress:
            print(f"    Embedding ARG1 ({len(a1_list)} texts)...")
        a1_emb = embedder.embed(
            a1_list, batch_size=batch_size, show_progress=show_progress
        ).astype(np.float32)
    else:
        # Use SentenceTransformer.encode() method
        if show_progress:
            print(f"    Embedding ARG0 ({len(a0_list)} texts)...")
        a0_emb = embedder.encode(
            a0_list, show_progress_bar=show_progress, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        if show_progress:
            print(f"    Embedding V ({len(v_list)} texts)...")
        v_emb = embedder.encode(
            v_list, show_progress_bar=show_progress, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        if show_progress:
            print(f"    Embedding ARG1 ({len(a1_list)} texts)...")
        a1_emb = embedder.encode(
            a1_list, show_progress_bar=show_progress, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
    
    # Concatenate
    role_emb = np.concatenate([a0_emb, v_emb, a1_emb], axis=1)
    if show_progress:
        print(f"    Role-based embedding shape: {role_emb.shape}")
    
    logger.info(f"Role-based embeddings generated: shape={role_emb.shape}")
    return role_emb


def generate_phrase_embeddings(
    texts: List[str],
    embedder,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """Generate phrase-level embeddings.
    
    Args:
        texts: List of phrases
        embedder: SentenceEmbedder or SentenceTransformer model instance
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
    
    Returns:
        np.ndarray: Embedding matrix with shape (n_samples, embed_dim)
    
    Example:
        >>> from causal_narrative.embedding import SentenceEmbedder
        >>> embedder = SentenceEmbedder(model_name='all-MiniLM-L6-v2')
        >>> texts = ['The storm', 'The government', 'The people']
        >>> embeddings = generate_phrase_embeddings(texts, embedder)
        >>> embeddings.shape
        (3, 384)
    """
    logger.info(f"Generating phrase embeddings for {len(texts)} texts")
    
    # Check if embedder is SentenceEmbedder or SentenceTransformer
    if isinstance(embedder, SentenceEmbedder):
        # Use SentenceEmbedder.embed() method
        embeddings = embedder.embed(
            texts, batch_size=batch_size, show_progress=show_progress
        ).astype(np.float32)
    else:
        # Use SentenceTransformer.encode() method
        embeddings = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            normalize_embeddings=True
        ).astype(np.float32)
    
    logger.info(f"Phrase embeddings generated: shape={embeddings.shape}")
    return embeddings


# =============================================================================
# PCA Dimensionality Reduction Functions
# =============================================================================

def role_based_pca(
    embeddings: np.ndarray,
    dim_a0: int,
    dim_v: int,
    dim_a1: int,
    embed_dim: int = 384,
    random_state: int = 42
) -> Tuple[np.ndarray, Tuple]:
    """Apply role-wise PCA dimensionality reduction on Role-based Event Embeddings.
    
    Apply PCA separately to the embeddings of three semantic roles (ARG0, V, ARG1),
    then concatenate the results. This preserves the semantic structure of each role.
    
    Args:
        embeddings: Input with shape (n_samples, 3 * embed_dim)
        dim_a0: Target dimension for ARG0 role after reduction
        dim_v: Target dimension for V (verb) role after reduction
        dim_a1: Target dimension for ARG1 role after reduction
        embed_dim: Original single-role embedding dimension (default 384)
        random_state: Random seed
    
    Returns:
        Tuple[np.ndarray, Tuple]:
            - reduced: Reduced result with shape (n_samples, dim_a0 + dim_v + dim_a1)
            - pca_models: (pca_a0, pca_v, pca_a1) PCA models tuple, for transform
    
    Example:
        >>> reduced, pca_models = role_based_pca(embeddings, 10, 5, 10)
        >>> reduced.shape
        (1000, 25)
    """
    from sklearn.decomposition import PCA
    
    logger.info(
        f"Applying role-based PCA: A0({embed_dim}->{dim_a0}), "
        f"V({embed_dim}->{dim_v}), A1({embed_dim}->{dim_a1})"
    )
    
    # Split each slot
    embeddings_a0 = embeddings[:, :embed_dim]
    embeddings_v = embeddings[:, embed_dim:2*embed_dim]
    embeddings_a1 = embeddings[:, 2*embed_dim:]
    
    # PCA separately
    logger.debug(f"PCA: A0 {embed_dim} -> {dim_a0}")
    pca_a0 = PCA(n_components=dim_a0, random_state=random_state)
    reduced_a0 = pca_a0.fit_transform(embeddings_a0)
    
    logger.debug(f"PCA: V {embed_dim} -> {dim_v}")
    pca_v = PCA(n_components=dim_v, random_state=random_state)
    reduced_v = pca_v.fit_transform(embeddings_v)
    
    logger.debug(f"PCA: A1 {embed_dim} -> {dim_a1}")
    pca_a1 = PCA(n_components=dim_a1, random_state=random_state)
    reduced_a1 = pca_a1.fit_transform(embeddings_a1)
    
    # Concatenate
    reduced = np.concatenate(
        [reduced_a0, reduced_v, reduced_a1], axis=1
    ).astype(np.float32)
    
    # Calculate variance ratio retained
    var_a0 = pca_a0.explained_variance_ratio_.sum()
    var_v = pca_v.explained_variance_ratio_.sum()
    var_a1 = pca_a1.explained_variance_ratio_.sum()
    
    logger.info(
        f"Role-based PCA completed: output shape={reduced.shape}, "
        f"variance retained: A0={var_a0:.3f}, V={var_v:.3f}, A1={var_a1:.3f}"
    )
    
    return reduced, (pca_a0, pca_v, pca_a1)


def apply_role_based_pca(
    embeddings: np.ndarray,
    pca_models: Tuple,
    embed_dim: int = 384
) -> np.ndarray:
    """Apply trained PCA models for Role-based dimensionality reduction.
    
    Args:
        embeddings: Input with shape (n_samples, 3 * embed_dim)
        pca_models: (pca_a0, pca_v, pca_a1) PCA models tuple
        embed_dim: Original single-role embedding dimension
    
    Returns:
        np.ndarray: Reduced embedding
    
    Example:
        >>> # First train PCA
        >>> reduced, pca_models = role_based_pca(train_embeddings, 10, 5, 10)
        >>> # Then apply to new data
        >>> new_reduced = apply_role_based_pca(test_embeddings, pca_models)
    """
    pca_a0, pca_v, pca_a1 = pca_models
    
    embeddings_a0 = embeddings[:, :embed_dim]
    embeddings_v = embeddings[:, embed_dim:2*embed_dim]
    embeddings_a1 = embeddings[:, 2*embed_dim:]
    
    reduced_a0 = pca_a0.transform(embeddings_a0)
    reduced_v = pca_v.transform(embeddings_v)
    reduced_a1 = pca_a1.transform(embeddings_a1)
    
    return np.concatenate(
        [reduced_a0, reduced_v, reduced_a1], axis=1
    ).astype(np.float32)


def phrase_pca(
    embeddings: np.ndarray,
    n_components: int,
    random_state: int = 42
) -> Tuple[np.ndarray, Any]:
    """Apply PCA dimensionality reduction on phrase-level embeddings.
    
    Args:
        embeddings: Input with shape (n_samples, embed_dim)
        n_components: Target dimension
        random_state: Random seed
    
    Returns:
        Tuple[np.ndarray, PCA]:
            - reduced: Reduced embedding
            - pca: PCA model instance
    
    Example:
        >>> reduced, pca = phrase_pca(embeddings, n_components=50)
        >>> reduced.shape
        (1000, 50)
    """
    from sklearn.decomposition import PCA
    
    logger.info(f"Applying phrase PCA: {embeddings.shape[1]} -> {n_components}")
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    variance = pca.explained_variance_ratio_.sum()
    
    logger.info(
        f"Phrase PCA completed: output shape={reduced.shape}, "
        f"variance retained={variance:.3f}"
    )
    
    return reduced, pca
