"""
Unified Causal Relationship Detection Module

This module provides a unified interface for detecting causal relationships
in text, supporting three backend methods:

1. **heuristic**: Fast, rule-based pattern matching using causal keywords
2. **llm**: LLM-based detection via OpenAI-compatible APIs (supports parallel calls)
3. **bert**: Supervised ML using fine-tuned BERT/RoBERTa models

Note: Span extraction (CausalSpanExtractor) has been moved to extraction.py

Usage:
    # Heuristic detection
    detector = CausalDetector(method='heuristic')
    result = detector.detect("The storm caused flooding.")

    # LLM-based detection with parallel processing
    detector = CausalDetector(
        method='llm',
        api_keys=['key1', 'key2'],
        base_url='https://api.siliconflow.cn/v1',
        model_name='deepseek-ai/DeepSeek-V3',
        max_workers=4
    )
    results = detector.detect_batch(sentences, show_progress=True)

    # BERT-based detection (auto-downloads to ./model/ on first use)
    detector = CausalDetector(method='bert')
    result = detector.detect("The storm caused flooding.")
"""

from __future__ import annotations

import re
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from loguru import logger
from tqdm import tqdm

# Try to import transformers for BERT support
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None


# =============================================================================
# Default model names
# =============================================================================

# BERT model defaults
DEFAULT_BERT_DETECTION_MODEL = "causal-narrative/roberta-causal-narrative-classifier"


# =============================================================================
# Default causal patterns for heuristic detection
# =============================================================================

DEFAULT_CAUSAL_PATTERNS = [
    r'\bbecause\b',
    r'\bcause[sd]?\b',
    r'\bresult(?:s|ed)?\s+in\b',
    r'\bled?\s+to\b',
    r'\bleads?\s+to\b',
    r'\bdue\s+to\b',
    r'\btherefore\b',
    r'\bthus\b',
    r'\bconsequently\b',
    r'\bas\s+a\s+result\b',
    r'\bso\s+that\b',
    r'\bin\s+order\s+to\b',
    r'\bmake[sd]?\b.*\b(possible|happen)\b',
    r'\bif\b.*\bthen\b',
    r'\bforced?\b',
    r'\benabled?\b',
    r'\bprevented?\b',
    r'\bstopped?\b',
    r'\bwhy\b',
    r'\bresponsible\s+for\b',
    r'\bcontribute[sd]?\s+to\b',
    r'\bblamed?\s+for\b',
]


# =============================================================================
# Causal Detection - Prompts and Parsers
# =============================================================================

# Causal detection prompts
DETECTION_SYSTEM_PROMPT = """You are an expert linguist specializing in causal relation extraction.

Your task is to determine whether a given sentence expresses a causal relationship. A causal relationship means one event or state causes, leads to, results in, or influences another event or state.

Consider these types of causality:
- Direct causation (X causes Y)
- Enabling/preventing (X enables/prevents Y)
- Conditional causation (if X then Y)
- Correlational causation with implied mechanism

Respond ONLY with valid JSON. Do not include any markdown formatting or code blocks."""

DETECTION_USER_TEMPLATE = """Analyze this sentence for causal relationships:

Sentence: "{sentence}"

Respond in this exact JSON format:
{{
  "has_causality": true or false,
  "score": 0.0 to 1.0,
  "rationale": "Brief explanation",
  "causal_type": "direct|enabling|conditional|correlational|none"
}}"""


def format_detection_prompt(sentence: str) -> str:
    """Format detection prompt for causal presence detection.

    Args:
        sentence: Input sentence text

    Returns:
        Formatted user prompt
    """
    return DETECTION_USER_TEMPLATE.format(sentence=sentence)


# JSON response parser
def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling common formatting issues.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not response_text or not response_text.strip():
        logger.warning("Empty response text")
        return None

    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"`(.*?)`",
    ]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

    # Try to find JSON object with regex
    # Match { ... } pattern
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    if matches:
        # Try each match (in case of multiple JSON objects)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Last resort: try to clean and parse
    cleaned = response_text.strip()
    # Remove common markdown artifacts
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response text: {response_text[:200]}...")
        return None


def parse_detection_response(
    response_text: str,
) -> Optional[Dict[str, Any]]:
    """Parse causal detection response.

    Expected format:
    {
        "has_causality": bool,
        "score": float,
        "rationale": str,
        "causal_type": str
    }

    Args:
        response_text: Raw response text

    Returns:
        Parsed dictionary with validated fields or None
    """
    data = parse_json_response(response_text)
    if not data:
        return None

    # Validate required fields
    required_fields = ["has_causality", "score"]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return None

    # Validate types and ranges
    try:
        has_causality = bool(data["has_causality"])
        score = float(data["score"])

        # Clamp score to valid range
        score = max(0.0, min(1.0, score))

        # Extract optional fields
        rationale = data.get("rationale", "")
        causal_type = data.get("causal_type", "unknown")

        return {
            "has_causality": has_causality,
            "score": score,
            "rationale": rationale,
            "causal_type": causal_type,
        }

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid field types in detection response: {e}")
        return None


# =============================================================================
# BERT helper functions
# =============================================================================

def _resolve_bert_model_source(
    model_name: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    model_class: Any,  # AutoModelForSequenceClassification or AutoModelForTokenClassification
) -> str:
    """Resolve BERT model source: local path > cached download > HuggingFace Hub.
    
    If the model is downloaded from HuggingFace and ``cache_dir`` is set,
    it will be saved locally with ``save_pretrained`` for future offline use.
    
    Args:
        model_name: HuggingFace model name
        model_path: Local path to pre-downloaded model
        cache_dir: Directory to cache downloaded models
        model_class: Model class to use for downloading
        
    Returns:
        Path or name to load model from
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and torch required for BERT models. "
            "Install with: pip install transformers torch"
        )
    
    # 1. If explicit local path is provided and exists, use it directly
    if model_path and Path(model_path).is_dir():
        logger.info(f"Loading BERT model from local path: {model_path}")
        return model_path
    
    # 2. Check if model is already cached in cache_dir
    if cache_dir and model_name:
        safe_name = model_name.replace("/", "_")
        local_dir = Path(cache_dir) / safe_name
        if local_dir.is_dir() and any(local_dir.iterdir()):
            logger.info(f"Loading BERT model from local cache: {local_dir}")
            return str(local_dir)
    
    # 3. Download from HuggingFace Hub
    hub_name = model_name or model_path
    logger.info(f"Downloading BERT model from HuggingFace Hub: {hub_name}")
    
    # Download first (this uses HF's internal cache)
    tokenizer = AutoTokenizer.from_pretrained(hub_name)
    model = model_class.from_pretrained(hub_name)
    
    # Save to local cache_dir for future offline use
    if cache_dir:
        safe_name = hub_name.replace("/", "_")
        local_dir = Path(cache_dir) / safe_name
        local_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(local_dir))
        model.save_pretrained(str(local_dir))
        logger.info(f"BERT model saved to local cache: {local_dir}")
        return str(local_dir)
    
    # No cache_dir — return hub name (HF handles its own cache)
    return hub_name


# =============================================================================
# BERT Detection Backend
# =============================================================================

class BERTCausalDetector:
    """
    BERT-based Causality Detector
    
    Classification: Determine if sentence contains causality.
    
    Supports auto-downloading from HuggingFace Hub and caching the model
    locally so subsequent loads are fast and offline-friendly.
    
    Usage:
        detector = BERTCausalDetector()
        has_causal = detector.has_causality("The storm caused flooding.")
        score = detector.predict_score("The storm caused flooding.")
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = "model",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize BERT detector.
        
        Args:
            model_name: HuggingFace model name (default: roberta-causal-narrative-classifier)
            model_path: Local path to pre-downloaded model
            cache_dir: Directory to cache models (default: "model")
            device: Device ('cuda', 'cpu', or None for auto)
            threshold: Classification threshold (default 0.5)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: "
                "pip install transformers torch"
            )
        
        # Default model name
        if model_name is None and model_path is None:
            model_name = DEFAULT_BERT_DETECTION_MODEL
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Resolve model source
        model_source = _resolve_bert_model_source(
            model_name=model_name,
            model_path=model_path,
            cache_dir=cache_dir,
            model_class=AutoModelForSequenceClassification,
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = threshold
        self._model_source = model_source
        
        logger.info(f"BERTCausalDetector loaded from: {model_source}")
        logger.info(f"Device: {self.device}")
    
    def predict_score(self, text: str) -> float:
        """Predict causality score [0-1]"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            causal_prob = probs[0][1].item()  # Probability of label 1 (causal)
        
        return causal_prob
    
    def has_causality(self, text: str) -> bool:
        """Check if text contains causality"""
        score = self.predict_score(text)
        return score >= self.threshold
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """Batch prediction"""
        results = []
        for text in texts:
            score = self.predict_score(text)
            has_causal = score >= self.threshold
            results.append((has_causal, score))
        return results


# =============================================================================
# Default causal patterns for heuristic detection
# =============================================================================

DEFAULT_CAUSAL_PATTERNS = [
    r'\bbecause\b',
    r'\bcause[sd]?\b',
    r'\bresult(?:s|ed)?\s+in\b',
    r'\bled?\s+to\b',
    r'\bleads?\s+to\b',
    r'\bdue\s+to\b',
    r'\btherefore\b',
    r'\bthus\b',
    r'\bconsequently\b',
    r'\bas\s+a\s+result\b',
    r'\bso\s+that\b',
    r'\bin\s+order\s+to\b',
    r'\bmake[sd]?\b.*\b(possible|happen)\b',
    r'\bif\b.*\bthen\b',
    r'\bforced?\b',
    r'\benabled?\b',
    r'\bprevented?\b',
    r'\bstopped?\b',
    r'\bwhy\b',
    r'\bresponsible\s+for\b',
    r'\bcontribute[sd]?\s+to\b',
    r'\bblamed?\s+for\b',
]


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class DetectionResult:
    """Result of causal relationship detection.

    Attributes:
        has_causality: Whether causal relationship was detected
        score: Confidence score [0.0, 1.0]
        rationale: Explanation for the decision
        causal_type: Type of causality (direct, enabling, conditional, etc.)
        method: Detection method used ('heuristic', 'llm', 'bert')
    """
    has_causality: bool
    score: float
    rationale: str = ""
    causal_type: str = "none"
    method: str = "heuristic"


# =============================================================================
# CausalDetector
# =============================================================================

class CausalDetector:
    """Unified causal relationship detector supporting multiple backends.

    This class provides a consistent API for detecting causal relationships in
    text using different methods:

    - **'heuristic'**: Fast, rule-based pattern matching using regex patterns.
      No external dependencies required. Good for quick analysis.
    - **'llm'**: Uses LLM APIs (OpenAI-compatible) for semantic understanding.
      Supports parallel API calls with multiple keys for throughput.
      More accurate but requires API access.
    - **'bert'**: Uses fine-tuned BERT/RoBERTa models for classification.
      Runs locally, good accuracy without API costs.

    Attributes:
        method: Detection method ('heuristic', 'llm', 'bert')
        causal_patterns: Regex patterns for heuristic detection
        llm_client: LLM client instance (for 'llm' method)
        bert_detector: BERT detector instance (for 'bert' method)
        system_prompt: System prompt for LLM detection
        user_template: User prompt template for LLM detection

    Example:
        >>> # Simple heuristic detection (single sentence)
        >>> detector = CausalDetector(method='heuristic')
        >>> result = detector.detect("The storm caused flooding.")
        >>> print(result.has_causality)  # True
        >>> print(result.score)  # 0.333...

        >>> # Batch detection (multiple sentences)
        >>> detector = CausalDetector(method='heuristic')
        >>> results = detector.detect([
        ...     "The storm caused flooding.",
        ...     "The sun is shining.",
        ... ])
        >>> [r.has_causality for r in results]
        [True, False]

        >>> # LLM-based detection with parallel processing
        >>> detector = CausalDetector(
        ...     method='llm',
        ...     api_keys=['key1', 'key2', 'key3'],
        ...     base_url='https://api.siliconflow.cn/v1',
        ...     model_name='deepseek-ai/DeepSeek-V3',
        ... )
        >>> # Single or batch - same method
        >>> result = detector.detect("The storm caused flooding.")
        >>> results = detector.detect(multiple_sentences)

        >>> # BERT-based detection (default model, auto-downloads to ./model/)
        >>> detector = CausalDetector(method='bert')
        >>> result = detector.detect("The storm caused flooding.")
    """

    def __init__(
        self,
        method: str = "heuristic",
        # Heuristic parameters
        causal_patterns: Optional[List[str]] = None,
        # LLM parameters
        api_keys: Optional[List[str]] = None,
        model_name: str = "deepseek-ai/DeepSeek-V3",
        base_url: Optional[str] = None,
        max_workers: Optional[int] = None,
        timeout: int = 60,
        max_retries: int = 3,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        # BERT parameters
        bert_model_name: Optional[str] = None,
        bert_model_path: Optional[str] = None,
        bert_cache_dir: Optional[str] = "model",
        bert_device: Optional[str] = None,
        bert_threshold: float = 0.5,
    ):
        """Initialize CausalDetector.

        Args:
            method: Detection method. One of:
                - 'heuristic': Rule-based pattern matching (default)
                - 'llm': LLM-based detection via API
                - 'bert': Fine-tuned BERT classification
            causal_patterns: Custom regex patterns for heuristic method.
                If None, uses DEFAULT_CAUSAL_PATTERNS.
            api_keys: List of API keys for LLM method (required for 'llm').
                Multiple keys enable parallel processing with key rotation.
            model_name: LLM model identifier (e.g., 'deepseek-ai/DeepSeek-V3',
                'gpt-4o-mini'). Used with 'llm' method.
            base_url: API base URL (required for 'llm' method).
                Examples:
                - SiliconFlow: "https://api.siliconflow.cn/v1"
                - OpenAI: "https://api.openai.com/v1"
                - Custom endpoint: "https://your-api.com/v1"
                Must be explicitly provided when using 'llm' method.
            max_workers: Maximum concurrent API workers for LLM batch
                processing. If None (default), automatically set to the
                number of API keys provided (one worker per key for optimal
                parallel processing). You can set this to a smaller value
                to limit concurrency, or larger (though it will be capped
                at the number of API keys by the LLM client).
            timeout: API request timeout in seconds.
            max_retries: Maximum retry attempts for failed API calls.
            temperature: LLM sampling temperature (0.0 = deterministic).
            system_prompt: Custom system prompt for LLM detection.
                If None, uses the default causal detection prompt.
            user_template: Custom user prompt template for LLM detection.
                Must contain {sentence} placeholder. If None, uses default.
            bert_model_name: HuggingFace model name for BERT method.
                Default: "causal-narrative/roberta-causal-narrative-classifier"
                (set automatically if neither model_name nor model_path
                is provided).
            bert_model_path: Local path to a pre-downloaded BERT model
                directory. Takes precedence over bert_model_name if the
                path exists.
            bert_cache_dir: Directory to save the downloaded BERT model
                for future offline reuse. Default: "model" (relative to
                cwd). Set to None to use HuggingFace's default cache only.
            bert_device: Device for BERT inference ('cuda', 'cpu', or None
                for auto-detection).
            bert_threshold: Classification threshold for BERT (default 0.5).

        Raises:
            ValueError: If method is not supported, or required parameters
                are missing for the chosen method.
        """
        if method not in ("heuristic", "llm", "bert"):
            raise ValueError(
                f"Unsupported method: '{method}'. "
                f"Choose from: 'heuristic', 'llm', 'bert'"
            )

        self.method = method
        self.causal_patterns = causal_patterns or DEFAULT_CAUSAL_PATTERNS
        self.llm_client = None
        self.bert_detector = None
        self._max_workers = max_workers

        # Store prompts (now defined in this module)
        self.system_prompt = system_prompt or DETECTION_SYSTEM_PROMPT
        self.user_template = user_template or DETECTION_USER_TEMPLATE

        if method == "llm":
            if not api_keys:
                raise ValueError(
                    "api_keys is required for LLM method. "
                    "Provide at least one API key."
                )
            if not base_url:
                raise ValueError(
                    "base_url is required for LLM method. "
                    "Provide the API endpoint URL. Examples:\n"
                    "  - SiliconFlow: 'https://api.siliconflow.cn/v1'\n"
                    "  - OpenAI: 'https://api.openai.com/v1'\n"
                    "  - Custom: 'https://your-api.com/v1'"
                )
            
            # Set max_workers to number of API keys if not specified
            effective_max_workers = max_workers if max_workers is not None else len(api_keys)
            self._max_workers = effective_max_workers
            
            from causal_narrative.llm import LLMClient

            self.llm_client = LLMClient(
                api_keys=api_keys,
                model_name=model_name,
                base_url=base_url,
                max_workers=effective_max_workers,
                timeout=timeout,
                max_retries=max_retries,
                temperature=temperature,
            )
            logger.info(
                f"CausalDetector initialized with LLM method "
                f"(model={model_name}, base_url={base_url})"
            )
            logger.info(
                f"  API keys: {len(api_keys)}, max_workers: {effective_max_workers}"
                + ("" if max_workers is not None else " (auto-set to #keys)")
                + (" - you can customize max_workers to limit concurrency" if max_workers is None else "")
            )

        elif method == "bert":
            # BERT backend is now built-in
            self.bert_detector = BERTCausalDetector(
                model_name=bert_model_name,  # None -> uses DEFAULT_BERT_DETECTION_MODEL
                model_path=bert_model_path,
                cache_dir=bert_cache_dir,
                device=bert_device,
                threshold=bert_threshold,
            )
            effective_name = bert_model_name or bert_model_path or DEFAULT_BERT_DETECTION_MODEL
            logger.info(
                f"CausalDetector initialized with BERT method "
                f"(model={effective_name}, cache_dir={bert_cache_dir})"
            )

        else:
            logger.info(
                f"CausalDetector initialized with heuristic method "
                f"({len(self.causal_patterns)} patterns)"
            )

    def detect(
        self,
        sentences: Union[str, List[str]],
        show_progress: bool = True,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """Detect causal relationship(s) in sentence(s).

        This method automatically handles both single sentence and batch processing.

        Args:
            sentences: Input sentence (str) or list of sentences (List[str]).
            show_progress: Show progress bar for batch processing (ignored for single sentence).

        Returns:
            - If input is str: Returns a single DetectionResult
            - If input is List[str]: Returns List[DetectionResult] in same order

        Example:
            >>> detector = CausalDetector(method='heuristic')
            >>> 
            >>> # Single sentence
            >>> result = detector.detect("The storm caused flooding.")
            >>> result.has_causality
            True
            >>> 
            >>> # Multiple sentences
            >>> results = detector.detect([
            ...     "The storm caused flooding.",
            ...     "The sun is shining.",
            ...     "High taxes led to protests."
            ... ])
            >>> [r.has_causality for r in results]
            [True, False, True]
        """
        # Handle single sentence
        if isinstance(sentences, str):
            if self.method == "heuristic":
                return self._detect_heuristic(sentences)
            elif self.method == "llm":
                return self._detect_llm(sentences)
            elif self.method == "bert":
                return self._detect_bert(sentences)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        # Handle batch
        elif isinstance(sentences, list):
            if not sentences:
                return []
            
            if self.method == "heuristic":
                return self._detect_heuristic_batch(sentences, show_progress)
            elif self.method == "llm":
                return self._detect_llm_batch(sentences, show_progress)
            elif self.method == "bert":
                return self._detect_bert_batch(sentences, show_progress)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        else:
            raise TypeError(
                f"sentences must be str or List[str], got {type(sentences).__name__}"
            )

    # ---- Heuristic detection ----

    def _detect_heuristic(self, sentence: str) -> DetectionResult:
        """Detect causal relationship using keyword pattern matching."""
        logger.debug(f"Heuristic detection for sentence: {sentence[:50]}...")
        sentence_lower = sentence.lower()
        matches = []

        for pattern in self.causal_patterns:
            if re.search(pattern, sentence_lower):
                matches.append(pattern)

        has_causality = len(matches) > 0
        score = min(len(matches) / 3.0, 1.0) if has_causality else 0.0

        logger.debug(f"Heuristic detection result: has_causality={has_causality}, score={score}, matches={len(matches)}")
        return DetectionResult(
            has_causality=has_causality,
            score=score,
            rationale=f"Matched {len(matches)} pattern(s)" if matches else "No causal patterns found",
            causal_type="heuristic" if has_causality else "none",
            method="heuristic",
        )

    def _detect_heuristic_batch(
        self, sentences: List[str], show_progress: bool
    ) -> List[DetectionResult]:
        """Batch heuristic detection."""
        logger.info(f"Starting batch heuristic detection for {len(sentences)} sentences")
        iterator = sentences
        if show_progress:
            iterator = tqdm(iterator, desc="Heuristic detection", unit="sent")

        results = [self._detect_heuristic(sent) for sent in iterator]
        logger.info(f"Completed batch heuristic detection: {sum(r.has_causality for r in results)} causal sentences found")
        return results

    # ---- LLM detection ----

    def _detect_llm(self, sentence: str) -> DetectionResult:
        """Detect causal relationship using LLM."""
        logger.debug(f"LLM detection for sentence: {sentence[:50]}...")
        prompt = self.user_template.format(sentence=sentence)
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            parsed = parse_detection_response(response.content)
            if parsed:
                logger.debug(f"LLM detection result: has_causality={parsed['has_causality']}, score={parsed['score']}")
                return DetectionResult(
                    has_causality=parsed["has_causality"],
                    score=parsed["score"],
                    rationale=parsed.get("rationale", ""),
                    causal_type=parsed.get("causal_type", "unknown"),
                    method="llm",
                )
        except Exception as e:
            logger.warning(f"LLM detection failed, falling back to heuristic: {e}")

        # Fallback to heuristic
        result = self._detect_heuristic(sentence)
        result.method = "llm_fallback"
        logger.debug(f"Using heuristic fallback result")
        return result

    def _detect_llm_batch(
        self, sentences: List[str], show_progress: bool
    ) -> List[DetectionResult]:
        """Batch LLM detection with parallel API calls."""
        logger.info(f"Starting batch LLM detection for {len(sentences)} sentences")
        prompts = [self.user_template.format(sentence=s) for s in sentences]

        # Use LLMClient's built-in batch_generate which handles parallel execution
        responses = self.llm_client.batch_generate(
            prompts=prompts,
            system_prompt=self.system_prompt,
            show_progress=show_progress,
        )

        results = []
        fallback_count = 0
        for i, response in enumerate(responses):
            try:
                parsed = parse_detection_response(response.content)
                if parsed:
                    results.append(
                        DetectionResult(
                            has_causality=parsed["has_causality"],
                            score=parsed["score"],
                            rationale=parsed.get("rationale", ""),
                            causal_type=parsed.get("causal_type", "unknown"),
                            method="llm",
                        )
                    )
                    continue
            except Exception as e:
                logger.warning(f"Failed to parse LLM response for sentence {i}: {e}")

            # Fallback to heuristic for failed cases
            result = self._detect_heuristic(sentences[i])
            result.method = "llm_fallback"
            results.append(result)
            fallback_count += 1

        logger.info(f"Completed batch LLM detection: {sum(r.has_causality for r in results)} causal sentences found, {fallback_count} fallbacks")
        return results

    # ---- BERT detection ----

    def _detect_bert(self, sentence: str) -> DetectionResult:
        """Detect causal relationship using BERT model."""
        logger.debug(f"BERT detection for sentence: {sentence[:50]}...")
        score = self.bert_detector.predict_score(sentence)
        has_causality = self.bert_detector.has_causality(sentence)

        logger.debug(f"BERT detection result: has_causality={has_causality}, score={score:.4f}")
        return DetectionResult(
            has_causality=has_causality,
            score=score,
            rationale=f"BERT classification score: {score:.4f}",
            causal_type="bert_classification" if has_causality else "none",
            method="bert",
        )

    def _detect_bert_batch(
        self, sentences: List[str], show_progress: bool
    ) -> List[DetectionResult]:
        """Batch BERT detection with per-sentence progress."""
        logger.info(f"Starting batch BERT detection for {len(sentences)} sentences")
        results = []
        iterator = sentences
        if show_progress:
            iterator = tqdm(iterator, desc="BERT detection", unit="sent")

        for sent in iterator:
            score = self.bert_detector.predict_score(sent)
            has_causality = score >= self.bert_detector.threshold
            results.append(
                DetectionResult(
                    has_causality=has_causality,
                    score=score,
                    rationale=f"BERT classification score: {score:.4f}",
                    causal_type="bert_classification" if has_causality else "none",
                    method="bert",
                )
            )

        logger.info(f"Completed batch BERT detection: {sum(r.has_causality for r in results)} causal sentences found")
        return results

    def get_cost_summary(self) -> Optional[Dict[str, Any]]:
        """Get LLM API cost summary (only for LLM method).

        Returns:
            Dictionary with cost stats, or None if not using LLM method.
        """
        if self.llm_client:
            return self.llm_client.get_cost_summary()
        return None

