"""Sentence segmentation for English text.

This module provides sentence tokenization functionality using NLTK.
"""

import re
from typing import List, Optional

from loguru import logger


def ensure_nltk_data() -> None:
    """Ensure NLTK punkt tokenizer data is available.

    Downloads punkt data if not already present. This function will:
    1. Check if NLTK is installed
    2. Check if punkt tokenizer data exists
    3. Automatically download if missing
    4. Verify successful download
    
    Raises:
        ImportError: If NLTK is not installed
    """
    try:
        import nltk
        
        logger.debug("Checking NLTK punkt tokenizer data availability...")
        
        try:
            # Try to find the punkt tokenizer data
            nltk.data.find("tokenizers/punkt")
            logger.debug("NLTK punkt tokenizer data found")
        except LookupError:
            # Data not found, download it
            logger.info("NLTK punkt tokenizer data not found. Downloading...")
            logger.info("This is a one-time download and may take a few moments...")
            
            try:
                # Download punkt tokenizer
                success = nltk.download("punkt", quiet=False)
                
                if success:
                    logger.info("✓ NLTK punkt data downloaded successfully")
                    
                    # Verify the download
                    try:
                        nltk.data.find("tokenizers/punkt")
                        logger.info("✓ NLTK punkt data verified and ready to use")
                    except LookupError:
                        logger.error("✗ Downloaded data verification failed")
                        raise RuntimeError("NLTK punkt data download verification failed")
                else:
                    logger.error("✗ NLTK punkt data download failed")
                    raise RuntimeError("Failed to download NLTK punkt data")
                    
            except Exception as e:
                logger.error(f"Error downloading NLTK data: {e}")
                logger.info("You can manually download by running: python -m nltk.downloader punkt")
                raise
                
    except ImportError:
        logger.error("NLTK is not installed")
        logger.info("Install NLTK with: pip install nltk")
        raise ImportError(
            "NLTK is required for sentence tokenization. "
            "Install it with: pip install nltk"
        )


def normalize_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """Normalize text by removing control characters and extra whitespace.

    Args:
        text: Input text
        remove_extra_whitespace: Remove extra whitespace and newlines

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    if remove_extra_whitespace:
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def segment_text(
    text: str,
    language: str = "english",
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> List[str]:
    """Segment text into sentences using NLTK.

    This function performs sentence tokenization with optional length filtering
    and splitting of overly long sentences.

    Args:
        text: Input text to segment
        language: Language for tokenization (default: "english")
            Supported: "english", "german", "french", etc. (NLTK languages)
        min_length: Minimum sentence length in characters (default: None = no filter)
            If set, sentences shorter than this will be filtered out
        max_length: Maximum sentence length in characters (default: None = no filter)
            If set, sentences longer than this will be split on commas/semicolons

    Returns:
        List of sentence strings

    Raises:
        ImportError: If NLTK is not available

    Example:
        >>> # Basic segmentation
        >>> sentences = segment_text("The storm caused flooding. Many homes were damaged.")
        >>> print(sentences)
        ['The storm caused flooding.', 'Many homes were damaged.']

        >>> # With length filtering
        >>> sentences = segment_text(
        ...     "Hi. The storm caused flooding. Many homes were damaged.",
        ...     min_length=10
        ... )
        >>> print(sentences)  # "Hi." is filtered out
        ['The storm caused flooding.', 'Many homes were damaged.']

        >>> # Batch processing
        >>> texts = ["Text 1. Sentence 2.", "Text A. Sentence B."]
        >>> results = [segment_text(t) for t in texts]
    """
    if not text or not text.strip():
        return []

    try:
        import nltk

        ensure_nltk_data()

        # Normalize text
        text = normalize_text(text, remove_extra_whitespace=True)

        # Tokenize using NLTK
        sentences = nltk.sent_tokenize(text, language=language)

        # Filter out very short sentences (likely errors)
        # Always apply a minimum threshold of 3 characters
        min_threshold = max(3, min_length) if min_length else 3
        sentences = [s.strip() for s in sentences if len(s.strip()) >= min_threshold]

        # Apply length filters if specified
        if min_length is not None or max_length is not None:
            filtered = []
            for sent in sentences:
                sent_len = len(sent)

                # Check minimum length
                if min_length is not None and sent_len < min_length:
                    continue

                # Check maximum length
                if max_length is not None and sent_len > max_length:
                    # Split very long sentences on commas or semicolons
                    sub_sentences = re.split(r"[,;]", sent)
                    for sub in sub_sentences:
                        sub = sub.strip()
                        # Only add if it meets length requirements
                        sub_len = len(sub)
                        if (min_length is None or sub_len >= min_length) and \
                           (max_length is None or sub_len <= max_length):
                            filtered.append(sub)
                else:
                    # Sentence is within acceptable range
                    filtered.append(sent)

            sentences = filtered

        return sentences

    except ImportError:
        logger.error("NLTK is required for sentence tokenization")
        raise ImportError(
            "NLTK is not installed. Install it with: pip install nltk"
        )
