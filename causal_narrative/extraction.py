"""
Causal Span Extraction Module

This module provides complete utilities for Subtask 2 (causal span extraction):
- Prompt templates for LLM-based span extraction
- JSON response parsers for LLM outputs
- Validators for extracted causal spans
- CausalSpanExtractor class (pattern/LLM/BERT methods)
- BERT span extraction backend

Note: Subtask 1 (causal detection) prompts and parsers are in detection.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from tqdm import tqdm

from causal_narrative.models import CausalSpan

# Try to import transformers for BERT support
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None


# =============================================================================
# Prompt templates for Span Extraction
# =============================================================================

# Cause and effect span extraction
EXTRACTION_SYSTEM_PROMPT = """You are an expert in extracting causal spans from text.

Your task is to identify the specific text spans that represent the CAUSE and the EFFECT in a causal sentence.

Guidelines:
- Extract the minimal span that captures the complete cause/effect
- Include necessary modifiers but avoid unnecessary words
- Use exact character positions (start index is inclusive, end index is exclusive)
- The cause span should represent what leads to the effect
- The effect span should represent what results from the cause

Respond ONLY with valid JSON. Do not include any markdown formatting or code blocks."""

EXTRACTION_USER_TEMPLATE = """Extract cause and effect spans from this sentence:

Sentence: "{sentence}"

Respond in this exact JSON format:
{{
  "cause_text": "extracted cause span",
  "effect_text": "extracted effect span",
  "cause_start": 0,
  "cause_end": 10,
  "effect_start": 15,
  "effect_end": 30
}}

Character positions should be 0-indexed. Start is inclusive, end is exclusive."""


def format_extraction_prompt(sentence: str) -> str:
    """Format extraction prompt for span extraction.

    Args:
        sentence: Input sentence text

    Returns:
        Formatted user prompt
    """
    return EXTRACTION_USER_TEMPLATE.format(sentence=sentence)


# =============================================================================
# JSON response parsers for Span Extraction
# =============================================================================

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


def parse_extraction_response(
    response_text: str,
) -> Optional[Dict[str, Any]]:
    """Parse span extraction response.

    Expected format:
    {
        "cause_text": str,
        "effect_text": str,
        "cause_start": int,
        "cause_end": int,
        "effect_start": int,
        "effect_end": int
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
    required_fields = [
        "cause_text",
        "effect_text",
        "cause_start",
        "cause_end",
        "effect_start",
        "effect_end",
    ]

    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return None

    # Validate types
    try:
        result = {
            "cause_text": str(data["cause_text"]),
            "effect_text": str(data["effect_text"]),
            "cause_start": int(data["cause_start"]),
            "cause_end": int(data["cause_end"]),
            "effect_start": int(data["effect_start"]),
            "effect_end": int(data["effect_end"]),
        }

        # Validate span consistency
        if result["cause_start"] < 0 or result["cause_end"] < 0:
            logger.warning("Negative span indices in cause")
            return None

        if result["effect_start"] < 0 or result["effect_end"] < 0:
            logger.warning("Negative span indices in effect")
            return None

        if result["cause_start"] > result["cause_end"]:
            logger.warning("Invalid cause span: start > end")
            return None

        if result["effect_start"] > result["effect_end"]:
            logger.warning("Invalid effect span: start > end")
            return None

        return result

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid field types in extraction response: {e}")
        return None


def safe_extract_field(
    data: Dict[str, Any],
    field: str,
    default: Any = None,
    field_type: type = str,
) -> Any:
    """Safely extract and convert field from dictionary.

    Args:
        data: Dictionary to extract from
        field: Field name
        default: Default value if field missing or conversion fails
        field_type: Expected type for conversion

    Returns:
        Extracted and converted value
    """
    if field not in data:
        return default

    try:
        return field_type(data[field])
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert field '{field}' to {field_type}")
        return default


# =============================================================================
# Span validators
# =============================================================================

def validate_span(
    text: str,
    span_start: int,
    span_end: int,
    expected_text: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate that a span matches the source text.

    Args:
        text: Source text
        span_start: Start index (inclusive)
        span_end: End index (exclusive)
        expected_text: Expected span text (optional)

    Returns:
        Tuple of (is_valid, extracted_text)
    """
    # Check bounds
    if span_start < 0 or span_end < 0:
        logger.warning(f"Negative span indices: ({span_start}, {span_end})")
        return False, None

    if span_start > span_end:
        logger.warning(f"Invalid span: start > end ({span_start}, {span_end})")
        return False, None

    if span_end > len(text):
        logger.warning(f"Span end {span_end} exceeds text length {len(text)}")
        return False, None

    # Extract span
    extracted = text[span_start:span_end]

    # Compare with expected text if provided
    if expected_text is not None:
        # Normalize whitespace for comparison
        extracted_norm = " ".join(extracted.split())
        expected_norm = " ".join(expected_text.split())

        if extracted_norm != expected_norm:
            logger.warning(
                f"Span mismatch - Expected: '{expected_text}', Got: '{extracted}'"
            )
            return False, extracted

    return True, extracted


def find_best_match_span(
    text: str,
    target_text: str,
    fuzzy_threshold: float = 0.8,
) -> Optional[Tuple[int, int]]:
    """Find best matching span for target text using fuzzy matching.

    Args:
        text: Source text to search
        target_text: Text to find
        fuzzy_threshold: Minimum similarity ratio (0-1)

    Returns:
        Tuple of (start, end) indices or None if no match found
    """
    if not target_text or not text:
        return None

    # Try exact match first
    start_idx = text.find(target_text)
    if start_idx != -1:
        return (start_idx, start_idx + len(target_text))

    # Try case-insensitive match
    lower_text = text.lower()
    lower_target = target_text.lower()
    start_idx = lower_text.find(lower_target)
    if start_idx != -1:
        return (start_idx, start_idx + len(target_text))

    # Try fuzzy matching using difflib
    try:
        from difflib import SequenceMatcher

        target_len = len(target_text)
        best_ratio = 0.0
        best_span = None

        # Sliding window approach
        for i in range(len(text) - target_len + 1):
            window = text[i : i + target_len]
            ratio = SequenceMatcher(None, target_text, window).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (i, i + target_len)

        if best_ratio >= fuzzy_threshold:
            logger.info(
                f"Found fuzzy match with ratio {best_ratio:.2f}: "
                f"'{text[best_span[0]:best_span[1]]}'"
            )
            return best_span

    except Exception as e:
        logger.warning(f"Fuzzy matching failed: {e}")

    # Try partial matches - find longest common substring
    target_tokens = target_text.split()
    if len(target_tokens) > 3:
        # Try with fewer tokens
        reduced_target = " ".join(target_tokens[:len(target_tokens) // 2])
        return find_best_match_span(text, reduced_target, fuzzy_threshold)

    logger.warning(f"No match found for: '{target_text}'")
    return None


def validate_causal_span(
    sentence: str,
    cause_text: str,
    effect_text: str,
    cause_start: int,
    cause_end: int,
    effect_start: int,
    effect_end: int,
    auto_correct: bool = True,
) -> CausalSpan:
    """Validate and optionally correct causal span extraction.

    Args:
        sentence: Original sentence text
        cause_text: Extracted cause text
        effect_text: Extracted effect text
        cause_start: Cause start index
        cause_end: Cause end index
        effect_start: Effect start index
        effect_end: Effect end index
        auto_correct: Attempt to fix invalid spans

    Returns:
        CausalSpan object with validation results
    """
    is_valid = True
    notes = []

    # Validate cause span
    cause_valid, cause_extracted = validate_span(
        sentence, cause_start, cause_end, cause_text
    )

    if not cause_valid:
        is_valid = False
        notes.append("Invalid cause span")

        if auto_correct:
            # Try to find correct span
            corrected_span = find_best_match_span(sentence, cause_text)
            if corrected_span:
                cause_start, cause_end = corrected_span
                cause_valid = True
                notes.append("Auto-corrected cause span")
                logger.info(f"Corrected cause span to: {corrected_span}")

    # Validate effect span
    effect_valid, effect_extracted = validate_span(
        sentence, effect_start, effect_end, effect_text
    )

    if not effect_valid:
        is_valid = False
        notes.append("Invalid effect span")

        if auto_correct:
            # Try to find correct span
            corrected_span = find_best_match_span(sentence, effect_text)
            if corrected_span:
                effect_start, effect_end = corrected_span
                effect_valid = True
                notes.append("Auto-corrected effect span")
                logger.info(f"Corrected effect span to: {corrected_span}")

    # Check for overlap (usually indicates error)
    if cause_valid and effect_valid:
        cause_set = set(range(cause_start, cause_end))
        effect_set = set(range(effect_start, effect_end))
        overlap = cause_set & effect_set

        if overlap:
            notes.append(f"Overlapping spans ({len(overlap)} chars)")
            logger.warning(
                f"Cause and effect spans overlap by {len(overlap)} characters"
            )

    # Use corrected or original values
    final_cause_text = cause_extracted if cause_extracted else cause_text
    final_effect_text = effect_extracted if effect_extracted else effect_text

    return CausalSpan(
        cause_text=final_cause_text,
        effect_text=final_effect_text,
        cause_char_span=(cause_start, cause_end),
        effect_char_span=(effect_start, effect_end),
        is_valid=is_valid and cause_valid and effect_valid,
        notes="; ".join(notes) if notes else None,
    )


def check_span_coverage(
    sentence: str,
    cause_start: int,
    cause_end: int,
    effect_start: int,
    effect_end: int,
) -> float:
    """Calculate what fraction of the sentence is covered by causal spans.

    Args:
        sentence: Original sentence
        cause_start: Cause start index
        cause_end: Cause end index
        effect_start: Effect start index
        effect_end: Effect end index

    Returns:
        Coverage ratio (0-1)
    """
    total_length = len(sentence)
    if total_length == 0:
        return 0.0

    # Calculate covered characters (accounting for overlap)
    cause_set = set(range(max(0, cause_start), min(total_length, cause_end)))
    effect_set = set(range(max(0, effect_start), min(total_length, effect_end)))
    covered = cause_set | effect_set

    return len(covered) / total_length


# =============================================================================
# BERT Model Constants
# =============================================================================

DEFAULT_BERT_SPAN_EXTRACTION_MODEL = "causal-narrative/roberta-causal-span-extractor"


# =============================================================================
# BERT Helper Functions
# =============================================================================

def _resolve_bert_model_source(
    model_name: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    model_class: Any,
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
    if cache_dir and model_name:
        safe_name = model_name.replace("/", "_")
        local_dir = Path(cache_dir) / safe_name
        local_dir.mkdir(parents=True, exist_ok=True)
        
        tokenizer.save_pretrained(str(local_dir))
        model.save_pretrained(str(local_dir))
        logger.info(f"Saved BERT model to local cache: {local_dir}")
        return str(local_dir)
    
    return hub_name


# =============================================================================
# BERT Span Extraction Backend
# =============================================================================

class BERTSpanExtractor:
    """
    BERT-based Causal Span Extractor
    
    Token classification: Extract cause and effect text spans.
    
    Usage:
        extractor = BERTSpanExtractor()
        span = extractor.extract_causal_span("The storm caused flooding.")
        if span:
            print(f"Cause: {span.cause_text}")
            print(f"Effect: {span.effect_text}")
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = "model",
        device: Optional[str] = None,
    ):
        """
        Initialize BERT span extractor.
        
        Args:
            model_name: HuggingFace model name (default: roberta-causal-span-extractor)
            model_path: Local path to pre-downloaded model
            cache_dir: Directory to cache models (default: "model")
            device: Device ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: "
                "pip install transformers torch"
            )
        
        # Default model name
        if model_name is None and model_path is None:
            model_name = DEFAULT_BERT_SPAN_EXTRACTION_MODEL
        
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
            model_class=AutoModelForTokenClassification,
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForTokenClassification.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        
        logger.info(f"BERTSpanExtractor loaded from: {model_source}")
        logger.info(f"Device: {self.device}")
    
    def extract_causal_span(self, text: str) -> Optional['SpanExtractionResult']:
        """
        Extract cause and effect spans from text
        
        Args:
            text: Input text
            
        Returns:
            SpanExtractionResult object or None if no span found
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Convert to labels
        labels = [self.id2label[pred.item()] for pred in predictions]
        
        # Extract cause and effect spans
        cause_tokens = []
        effect_tokens = []
        
        current_cause = []
        current_effect = []
        
        for idx, (label, offset) in enumerate(zip(labels, offset_mapping)):
            char_start, char_end = offset
            
            # Skip special tokens
            if char_start == char_end:
                continue
            
            # Handle CAUSE
            if 'CAUSE' in label:
                if label.startswith('B-') and current_cause:
                    cause_tokens.append(current_cause)
                    current_cause = []
                current_cause.append((char_start, char_end, probs[idx].max().item()))
            elif current_cause:
                cause_tokens.append(current_cause)
                current_cause = []
            
            # Handle EFFECT
            if 'EFFECT' in label:
                if label.startswith('B-') and current_effect:
                    effect_tokens.append(current_effect)
                    current_effect = []
                current_effect.append((char_start, char_end, probs[idx].max().item()))
            elif current_effect:
                effect_tokens.append(current_effect)
                current_effect = []
        
        if current_cause:
            cause_tokens.append(current_cause)
        if current_effect:
            effect_tokens.append(current_effect)
        
        # Extract first cause and effect
        if not cause_tokens or not effect_tokens:
            return None
        
        # Get spans
        cause_span = cause_tokens[0]
        effect_span = effect_tokens[0]
        
        cause_start = cause_span[0][0]
        cause_end = cause_span[-1][1]
        cause_text = text[cause_start:cause_end]
        cause_conf = sum(t[2] for t in cause_span) / len(cause_span)
        
        effect_start = effect_span[0][0]
        effect_end = effect_span[-1][1]
        effect_text = text[effect_start:effect_end]
        effect_conf = sum(t[2] for t in effect_span) / len(effect_span)
        
        return SpanExtractionResult(
            cause_text=cause_text,
            effect_text=effect_text,
            cause_start=cause_start,
            cause_end=cause_end,
            effect_start=effect_start,
            effect_end=effect_end,
            confidence=(cause_conf + effect_conf) / 2,
            method='bert',
        )
    
    def extract_batch(self, texts: List[str]) -> List[Optional['SpanExtractionResult']]:
        """Batch extraction"""
        results = []
        for text in texts:
            span = self.extract_causal_span(text)
            results.append(span)
        return results


# =============================================================================
# Default span extraction patterns
# =============================================================================

DEFAULT_SPAN_PATTERNS = [
    # "X because Y" -> cause=Y, effect=X
    (r'(.+?)\s+because\s+(.+)', 'effect_cause'),
    # "Because X, Y" -> cause=X, effect=Y
    (r'[Bb]ecause\s+(.+?),\s+(.+)', 'cause_effect'),
    # "X caused Y" -> cause=X, effect=Y
    (r'(.+?)\s+caused?\s+(.+)', 'cause_effect'),
    # "X led to Y" -> cause=X, effect=Y
    (r'(.+?)\s+led?\s+to\s+(.+)', 'cause_effect'),
    # "X leads to Y" -> cause=X, effect=Y
    (r'(.+?)\s+leads?\s+to\s+(.+)', 'cause_effect'),
    # "Due to X, Y" -> cause=X, effect=Y
    (r'[Dd]ue\s+to\s+(.+?),\s+(.+)', 'cause_effect'),
    # "X resulted in Y" -> cause=X, effect=Y
    (r'(.+?)\s+resulted?\s+in\s+(.+)', 'cause_effect'),
    # "X therefore Y" -> cause=X, effect=Y
    (r'(.+?)\s+therefore\s+(.+)', 'cause_effect'),
    # "X, thus Y" -> cause=X, effect=Y
    (r'(.+?),?\s+thus\s+(.+)', 'cause_effect'),
    # "X, consequently Y" -> cause=X, effect=Y
    (r'(.+?),?\s+consequently\s+(.+)', 'cause_effect'),
    # "As a result of X, Y" -> cause=X, effect=Y
    (r'[Aa]s\s+a\s+result\s+of\s+(.+?),\s+(.+)', 'cause_effect'),
    # "X so that Y" -> cause=X, effect=Y
    (r'(.+?)\s+so\s+that\s+(.+)', 'cause_effect'),
    # "If X then Y" -> cause=X, effect=Y
    (r'[Ii]f\s+(.+?),?\s+then\s+(.+)', 'cause_effect'),
    # "X forced Y" -> cause=X, effect=Y
    (r'(.+?)\s+forced\s+(.+)', 'cause_effect'),
    # "X enabled Y" -> cause=X, effect=Y
    (r'(.+?)\s+enabled?\s+(.+)', 'cause_effect'),
    # "X prevented Y" -> cause=X, effect=Y
    (r'(.+?)\s+prevented?\s+(.+)', 'cause_effect'),
    # "X is responsible for Y" -> cause=X, effect=Y
    (r'(.+?)\s+(?:is|are|was|were)\s+responsible\s+for\s+(.+)', 'cause_effect'),
    # "X contributed to Y" -> cause=X, effect=Y
    (r'(.+?)\s+contribute[sd]?\s+to\s+(.+)', 'cause_effect'),
]


# =============================================================================
# Data class for span extraction results
# =============================================================================

@dataclass
class SpanExtractionResult:
    """Result of cause/effect span extraction.

    Attributes:
        cause_text: Extracted cause text span
        effect_text: Extracted effect text span
        cause_start: Cause span start index (char-level, inclusive)
        cause_end: Cause span end index (char-level, exclusive)
        effect_start: Effect span start index (char-level, inclusive)
        effect_end: Effect span end index (char-level, exclusive)
        confidence: Extraction confidence [0.0, 1.0]
        method: Extraction method used ('pattern', 'llm', 'bert')
    """
    cause_text: str
    effect_text: str
    cause_start: int = -1
    cause_end: int = -1
    effect_start: int = -1
    effect_end: int = -1
    confidence: float = 0.0
    method: str = "pattern"


# =============================================================================
# CausalSpanExtractor - Main Class
# =============================================================================

class CausalSpanExtractor:
    """Unified cause/effect span extractor supporting multiple backends.

    Extracts the specific text spans representing the cause and effect
    from a causal sentence. Supports three methods:

    - **'pattern'**: Rule-based regex pattern matching (fast, no API needed)
    - **'llm'**: LLM-based extraction via API (accurate, requires API)
    - **'bert'**: Fine-tuned BERT token classification model (accurate, local)

    Attributes:
        method: Extraction method ('pattern', 'llm', 'bert')
        span_patterns: Regex patterns for pattern-based extraction
        llm_client: LLM client (for 'llm' method)
        bert_extractor: BERT extractor (for 'bert' method)

    Example:
        >>> # Pattern-based extraction (single sentence)
        >>> extractor = CausalSpanExtractor(method='pattern')
        >>> result = extractor.extract("Flooding occurred because of heavy rain.")
        >>> print(result.cause_text)  # "heavy rain"
        >>> print(result.effect_text)  # "Flooding occurred"

        >>> # Batch extraction (multiple sentences)
        >>> extractor = CausalSpanExtractor(method='pattern')
        >>> results = extractor.extract([
        ...     "Flooding occurred because of heavy rain.",
        ...     "High taxes led to protests."
        ... ])
        >>> results[0].cause_text
        'heavy rain'

        >>> # LLM-based extraction with parallel processing
        >>> extractor = CausalSpanExtractor(
        ...     method='llm',
        ...     api_keys=['key1', 'key2', 'key3'],
        ...     base_url='https://api.siliconflow.cn/v1',
        ...     model_name='deepseek-ai/DeepSeek-V3'
        ... )
        >>> # Single or batch - same method
        >>> result = extractor.extract("Flooding occurred because of heavy rain.")
        >>> results = extractor.extract(multiple_causal_sentences)
    """

    def __init__(
        self,
        method: str = "pattern",
        # Pattern parameters
        span_patterns: Optional[List[tuple]] = None,
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
    ):
        """Initialize CausalSpanExtractor.

        Args:
            method: Extraction method. One of:
                - 'pattern': Rule-based regex matching (default)
                - 'llm': LLM-based span extraction via API
                - 'bert': Fine-tuned BERT token classification
            span_patterns: Custom (pattern, order) tuples for pattern method.
                Each tuple is (regex_pattern, 'cause_effect' or 'effect_cause').
                If None, uses DEFAULT_SPAN_PATTERNS.
            api_keys: List of API keys for LLM method (required for 'llm').
            model_name: LLM model identifier.
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
            max_retries: Maximum retry attempts.
            temperature: LLM sampling temperature.
            system_prompt: Custom system prompt for LLM extraction.
            user_template: Custom user prompt template. Must contain
                {sentence} placeholder.
            bert_model_name: HuggingFace model name for BERT span extraction.
                Default: "causal-narrative/roberta-causal-span-extractor"
                (set automatically if neither model_name nor model_path
                is provided).
            bert_model_path: Local path to a pre-downloaded BERT span model.
                Takes precedence over bert_model_name if the path exists.
            bert_cache_dir: Directory to save the downloaded BERT model
                for future offline reuse. Default: "model".
            bert_device: Device for BERT inference.

        Raises:
            ValueError: If method is not supported or required parameters
                are missing.
        """
        if method not in ("pattern", "llm", "bert"):
            raise ValueError(
                f"Unsupported method: '{method}'. "
                f"Choose from: 'pattern', 'llm', 'bert'"
            )

        self.method = method
        self.span_patterns = span_patterns or DEFAULT_SPAN_PATTERNS
        self.llm_client = None
        self.bert_extractor = None

        self.system_prompt = system_prompt or EXTRACTION_SYSTEM_PROMPT
        self.user_template = user_template or EXTRACTION_USER_TEMPLATE

        if method == "llm":
            if not api_keys:
                raise ValueError("api_keys is required for LLM method.")
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
                f"CausalSpanExtractor initialized with LLM method (base_url={base_url})"
            )
            logger.info(
                f"  API keys: {len(api_keys)}, max_workers: {effective_max_workers}"
                + ("" if max_workers is not None else " (auto-set to #keys)")
                + (" - you can customize max_workers to limit concurrency" if max_workers is None else "")
            )

        elif method == "bert":
            # BERT backend
            self.bert_extractor = BERTSpanExtractor(
                model_name=bert_model_name,
                model_path=bert_model_path,
                cache_dir=bert_cache_dir,
                device=bert_device,
            )
            effective_name = bert_model_name or bert_model_path or DEFAULT_BERT_SPAN_EXTRACTION_MODEL
            logger.info(
                f"CausalSpanExtractor initialized with BERT method "
                f"(model={effective_name}, cache_dir={bert_cache_dir})"
            )

        else:
            logger.info(
                f"CausalSpanExtractor initialized with pattern method "
                f"({len(self.span_patterns)} patterns)"
            )

    def extract(
        self,
        sentences: Union[str, List[str]],
        show_progress: bool = True,
    ) -> Union[Optional[SpanExtractionResult], List[Optional[SpanExtractionResult]]]:
        """Extract cause and effect span(s) from sentence(s).

        This method automatically handles both single sentence and batch processing.

        Args:
            sentences: Input sentence (str) or list of sentences (List[str]).
            show_progress: Show progress bar for batch processing (ignored for single sentence).

        Returns:
            - If input is str: Returns Optional[SpanExtractionResult] (None if extraction fails)
            - If input is List[str]: Returns List[Optional[SpanExtractionResult]] in same order

        Example:
            >>> extractor = CausalSpanExtractor(method='pattern')
            >>> 
            >>> # Single sentence
            >>> result = extractor.extract("Flooding occurred because of heavy rain.")
            >>> result.cause_text
            'heavy rain.'
            >>> result.effect_text
            'Flooding occurred'
            >>> 
            >>> # Multiple sentences
            >>> results = extractor.extract([
            ...     "Flooding occurred because of heavy rain.",
            ...     "High taxes led to protests."
            ... ])
            >>> results[0].cause_text
            'heavy rain.'
        """
        # Handle single sentence
        if isinstance(sentences, str):
            if self.method == "pattern":
                return self._extract_pattern(sentences)
            elif self.method == "llm":
                return self._extract_llm(sentences)
            elif self.method == "bert":
                return self._extract_bert(sentences)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        # Handle batch
        elif isinstance(sentences, list):
            if not sentences:
                return []
            
            if self.method == "pattern":
                return self._extract_pattern_batch(sentences, show_progress)
            elif self.method == "llm":
                return self._extract_llm_batch(sentences, show_progress)
            elif self.method == "bert":
                return self._extract_bert_batch(sentences, show_progress)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        else:
            raise TypeError(
                f"sentences must be str or List[str], got {type(sentences).__name__}"
            )

    # ---- Pattern extraction ----

    def _extract_pattern(self, sentence: str) -> Optional[SpanExtractionResult]:
        """Extract spans using regex pattern matching."""
        logger.debug(f"Pattern extraction for sentence: {sentence[:50]}...")
        for pattern, order in self.span_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if order == "cause_effect":
                    cause_text = match.group(1).strip()
                    effect_text = match.group(2).strip()
                else:  # effect_cause
                    effect_text = match.group(1).strip()
                    cause_text = match.group(2).strip()

                # Find character positions
                cause_start = sentence.find(cause_text)
                cause_end = cause_start + len(cause_text) if cause_start >= 0 else -1
                effect_start = sentence.find(effect_text)
                effect_end = effect_start + len(effect_text) if effect_start >= 0 else -1

                logger.debug(f"Pattern extraction result: cause='{cause_text[:30]}...', effect='{effect_text[:30]}...'")
                return SpanExtractionResult(
                    cause_text=cause_text,
                    effect_text=effect_text,
                    cause_start=cause_start,
                    cause_end=cause_end,
                    effect_start=effect_start,
                    effect_end=effect_end,
                    confidence=0.7,
                    method="pattern",
                )

        logger.debug(f"No pattern matched for sentence")
        return None

    def _extract_pattern_batch(
        self, sentences: List[str], show_progress: bool
    ) -> List[Optional[SpanExtractionResult]]:
        """Batch pattern extraction."""
        logger.info(f"Starting batch pattern extraction for {len(sentences)} sentences")
        iterator = sentences
        if show_progress:
            iterator = tqdm(iterator, desc="Pattern extraction", unit="sent")
        results = [self._extract_pattern(s) for s in iterator]
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Completed batch pattern extraction: {success_count}/{len(sentences)} successful")
        return results

    # ---- LLM extraction ----

    def _extract_llm(self, sentence: str) -> Optional[SpanExtractionResult]:
        """Extract spans using LLM."""
        logger.debug(f"LLM extraction for sentence: {sentence[:50]}...")
        prompt = self.user_template.format(sentence=sentence)
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            parsed = parse_extraction_response(response.content)
            if parsed:
                logger.debug(f"LLM extraction result: cause='{parsed['cause_text'][:30]}...', effect='{parsed['effect_text'][:30]}...'")
                return SpanExtractionResult(
                    cause_text=parsed["cause_text"],
                    effect_text=parsed["effect_text"],
                    cause_start=parsed.get("cause_start", -1),
                    cause_end=parsed.get("cause_end", -1),
                    effect_start=parsed.get("effect_start", -1),
                    effect_end=parsed.get("effect_end", -1),
                    confidence=0.9,
                    method="llm",
                )
        except Exception as e:
            logger.warning(f"LLM span extraction failed: {e}")

        # Fallback to pattern
        logger.debug(f"Using pattern fallback for extraction")
        result = self._extract_pattern(sentence)
        if result:
            result.method = "llm_fallback"
        return result

    def _extract_llm_batch(
        self, sentences: List[str], show_progress: bool
    ) -> List[Optional[SpanExtractionResult]]:
        """Batch LLM extraction with parallel API calls."""
        logger.info(f"Starting batch LLM extraction for {len(sentences)} sentences")
        prompts = [self.user_template.format(sentence=s) for s in sentences]

        responses = self.llm_client.batch_generate(
            prompts=prompts,
            system_prompt=self.system_prompt,
            show_progress=show_progress,
        )

        results = []
        fallback_count = 0
        for i, response in enumerate(responses):
            try:
                parsed = parse_extraction_response(response.content)
                if parsed:
                    results.append(
                        SpanExtractionResult(
                            cause_text=parsed["cause_text"],
                            effect_text=parsed["effect_text"],
                            cause_start=parsed.get("cause_start", -1),
                            cause_end=parsed.get("cause_end", -1),
                            effect_start=parsed.get("effect_start", -1),
                            effect_end=parsed.get("effect_end", -1),
                            confidence=0.9,
                            method="llm",
                        )
                    )
                    continue
            except Exception as e:
                logger.warning(f"Failed to parse LLM span response {i}: {e}")

            # Fallback to pattern
            result = self._extract_pattern(sentences[i])
            if result:
                result.method = "llm_fallback"
            results.append(result)
            fallback_count += 1

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Completed batch LLM extraction: {success_count}/{len(sentences)} successful, {fallback_count} fallbacks")
        return results

    # ---- BERT extraction ----

    def _extract_bert(self, sentence: str) -> Optional[SpanExtractionResult]:
        """Extract spans using BERT token classification."""
        logger.debug(f"BERT extraction for sentence: {sentence[:50]}...")
        span = self.bert_extractor.extract_causal_span(sentence)
        if span:
            logger.debug(f"BERT extraction result: cause='{span.cause_text[:30]}...', effect='{span.effect_text[:30]}...'")
            return SpanExtractionResult(
                cause_text=span.cause_text,
                effect_text=span.effect_text,
                cause_start=span.cause_start,
                cause_end=span.cause_end,
                effect_start=span.effect_start,
                effect_end=span.effect_end,
                confidence=span.confidence,
                method="bert",
            )
        logger.debug(f"BERT extraction returned no result")
        return None

    def _extract_bert_batch(
        self, sentences: List[str], show_progress: bool
    ) -> List[Optional[SpanExtractionResult]]:
        """Batch BERT extraction with per-sentence progress."""
        logger.info(f"Starting batch BERT extraction for {len(sentences)} sentences")
        results = []
        iterator = sentences
        if show_progress:
            iterator = tqdm(iterator, desc="BERT extraction", unit="sent")

        for sent in iterator:
            span = self.bert_extractor.extract_causal_span(sent)
            if span:
                results.append(
                    SpanExtractionResult(
                        cause_text=span.cause_text,
                        effect_text=span.effect_text,
                        cause_start=span.cause_start,
                        cause_end=span.cause_end,
                        effect_start=span.effect_start,
                        effect_end=span.effect_end,
                        confidence=span.confidence,
                        method="bert",
                    )
                )
            else:
                results.append(None)

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Completed batch BERT extraction: {success_count}/{len(sentences)} successful")
        return results

    def get_cost_summary(self) -> Optional[Dict[str, Any]]:
        """Get LLM API cost summary (only for LLM method).

        Returns:
            Dictionary with cost stats, or None if not using LLM method.
        """
        if self.llm_client:
            return self.llm_client.get_cost_summary()
        return None
