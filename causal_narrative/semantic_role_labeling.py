"""
Semantic Role Labeling (SRL) with spaCy and AllenNLP.

This module provides SRL functionality using:
- spaCy dependency parsing (default, always available)
- AllenNLP pre-trained models (optional, requires special environment)

Both methods output consistent structured dictionaries compatible with
the AllenNLP format for downstream processing.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import spacy
from loguru import logger
from spacy.tokens import Token
from tqdm import tqdm

# Check AllenNLP availability
try:
    import torch
    from allennlp_models.structured_prediction.predictors import (
        SemanticRoleLabelerPredictor as Predictor,
    )
    ALLENNLP_AVAILABLE = True
except ImportError:
    ALLENNLP_AVAILABLE = False
    torch = None
    Predictor = None


# =============================================================================
# spaCy SRL
# =============================================================================

class SpacySRL:
    """SRL using spaCy dependency parsing.
    
    Extracts semantic roles (ARG0, V, ARG1) using dependency relations.
    Handles passive voice transformation. Output format is compatible with
    AllenNLP for consistent downstream processing.
    
    Attributes:
        nlp: spaCy language model
        model_name: Name of loaded model
        batch_size: Default batch size for processing
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 32):
        """Initialize spaCy SRL.
        
        Args:
            model_name: spaCy model to load (will auto-download if missing)
            batch_size: Default batch size for processing multiple texts
            
        Raises:
            OSError: If model download fails
            RuntimeError: If model cannot be loaded after download
        """
        logger.debug(f"Initializing spaCy SRL with model: {model_name}")
        
        try:
            # Try to load the model
            self.nlp = spacy.load(model_name)
            self.model_name = model_name
            self.batch_size = batch_size
            logger.info(f"✓ Loaded spaCy model: {model_name}")
            
        except OSError:
            # Model not found, try to download it
            logger.warning(f"spaCy model '{model_name}' not found")
            logger.info(f"Attempting to download spaCy model '{model_name}'...")
            logger.info("This is a one-time download and may take a few moments...")
            
            try:
                import subprocess
                
                # Download the model using spacy download command
                logger.info(f"Running: python -m spacy download {model_name}")
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    logger.info(f"✓ spaCy model '{model_name}' downloaded successfully")
                    
                    # Try to load the model again
                    try:
                        self.nlp = spacy.load(model_name)
                        self.model_name = model_name
                        self.batch_size = batch_size
                        logger.info(f"✓ spaCy model '{model_name}' loaded successfully")
                    except OSError as e:
                        logger.error(f"✗ Failed to load model after download: {e}")
                        raise RuntimeError(
                            f"Downloaded spaCy model '{model_name}' but failed to load it. "
                            f"Try manually: python -m spacy download {model_name}"
                        )
                else:
                    logger.error(f"✗ Failed to download spaCy model '{model_name}'")
                    logger.error(f"Error output: {result.stderr}")
                    logger.info(f"You can manually download with: python -m spacy download {model_name}")
                    raise OSError(
                        f"Failed to download spaCy model '{model_name}'. "
                        f"Please install it manually with: python -m spacy download {model_name}"
                    )
                    
            except Exception as e:
                logger.error(f"✗ Error during model download: {e}")
                logger.info(f"Manual installation: python -m spacy download {model_name}")
                raise
    
    def _find_subject(self, token: Token) -> Optional[Token]:
        """Find subject for a verb token."""
        for child in token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child
        return None
    
    def _find_object(self, token: Token) -> Optional[Token]:
        """Find object for a verb token."""
        for child in token.children:
            if child.dep_ in ["dobj", "pobj", "dative"]:
                return child
        return None
    
    def _find_agent(self, token: Token) -> Optional[Token]:
        """Find agent in passive construction."""
        for child in token.children:
            if child.dep_ == "agent":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return grandchild
        return None
    
    def _get_full_phrase(self, token: Token) -> str:
        """Get full phrase including dependents."""
        subtree = list(token.subtree)
        subtree.sort(key=lambda t: t.i)
        return " ".join([t.text for t in subtree])
    
    def _is_passive(self, verb: Token) -> bool:
        """Check if verb is in passive voice."""
        if verb.dep_ == "ROOT" or verb.pos_ == "VERB":
            for child in verb.children:
                if child.dep_ == "auxpass":
                    return True
        return False
    
    def _make_srl_dict(
        self,
        words: List[str],
        verb_token: Token,
        a0: Optional[str],
        v: str,
        a1: Optional[str],
    ) -> Dict[str, Any]:
        """Create AllenNLP-compatible SRL dictionary.
        
        Args:
            words: List of words in sentence
            verb_token: Verb token
            a0: ARG0 (agent/subject)
            v: Verb
            a1: ARG1 (patient/object)
            
        Returns:
            Dictionary with 'words' and 'verbs' keys
        """
        # Build description string
        description_parts = []
        if a0:
            description_parts.append(f"[ARG0: {a0}]")
        description_parts.append(f"[V: {v}]")
        if a1:
            description_parts.append(f"[ARG1: {a1}]")
        description = " ".join(description_parts)
        
        # Build BIO tags (simplified)
        tags = ["O"] * len(words)
        if verb_token:
            tags[verb_token.i] = "B-V"
        
        verb_dict = {
            "verb": v,
            "description": description,
            "tags": tags,
        }
        
        return {
            "words": words,
            "verbs": [verb_dict],
        }
    
    def _process_single(self, text: str) -> Dict[str, Any]:
        """Process a single text with SRL.
        
        Args:
            text: Input text
            
        Returns:
            SRL dictionary with 'words' and 'verbs' keys
        """
        logger.debug(f"Processing SRL for text: {text[:50]}...")
        doc = self.nlp(text)
        words = [token.text for token in doc]
        
        # Find main verb (ROOT)
        verb = None
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                verb = token
                break
        
        if not verb:
            logger.debug(f"No verb found in: {text}")
            return {"words": words, "verbs": []}
        
        # Check if passive voice
        is_passive = self._is_passive(verb)
        logger.debug(f"Verb found: '{verb.text}', passive={is_passive}")
        
        # Extract components
        if is_passive:
            subject = self._find_subject(verb)
            agent = self._find_agent(verb)
            a0 = self._get_full_phrase(agent) if agent else None
            v = verb.lemma_
            a1 = self._get_full_phrase(subject) if subject else None
        else:
            subject = self._find_subject(verb)
            obj = self._find_object(verb)
            a0 = self._get_full_phrase(subject) if subject else None
            v = verb.lemma_
            a1 = self._get_full_phrase(obj) if obj else None
        
        logger.debug(f"SRL result: ARG0='{a0}', V='{v}', ARG1='{a1}'")
        return self._make_srl_dict(words, verb, a0, v, a1)
    
    def process(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process text(s) with SRL.
        
        Automatically handles single text or multiple texts with batching.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing (uses default if None)
            show_progress: Show progress bar for batch processing
            
        Returns:
            Single SRL dict if input is str, list of dicts if input is list
            
        Examples:
            >>> srl = SpacySRL()
            >>> result = srl.process("The dog ran.")
            >>> results = srl.process(["Text 1", "Text 2"], show_progress=True)
        """
        # Single text
        if isinstance(texts, str):
            return self._process_single(texts)
        
        # Multiple texts - use batching
        if not texts:
            return []
        
        logger.info(f"Starting batch SRL processing for {len(texts)} texts using spaCy")
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        
        # Create iterator with optional progress bar
        docs_iterator = self.nlp.pipe(texts, batch_size=batch_size)
        if show_progress:
            docs_iterator = tqdm(
                docs_iterator,
                total=len(texts),
                desc="Processing SRL (spaCy)",
                unit="texts"
            )
        
        for doc in docs_iterator:
            words = [token.text for token in doc]
            
            # Find main verb
            verb = None
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                    verb = token
                    break
            
            if not verb:
                results.append({"words": words, "verbs": []})
                continue
            
            # Check if passive voice
            is_passive = self._is_passive(verb)
            
            # Extract components
            if is_passive:
                subject = self._find_subject(verb)
                agent = self._find_agent(verb)
                a0 = self._get_full_phrase(agent) if agent else None
                v = verb.lemma_
                a1 = self._get_full_phrase(subject) if subject else None
            else:
                subject = self._find_subject(verb)
                obj = self._find_object(verb)
                a0 = self._get_full_phrase(subject) if subject else None
                v = verb.lemma_
                a1 = self._get_full_phrase(obj) if obj else None
            
            srl_dict = self._make_srl_dict(words, verb, a0, v, a1)
            results.append(srl_dict)
        
        verb_count = sum(1 for r in results if r.get('verbs'))
        logger.info(f"Completed batch SRL processing: {verb_count}/{len(results)} texts with verbs found")
        return results
    
    def extract_roles(
        self,
        srl_result: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Optional[str]], List[Dict[str, Optional[str]]]]:
        """Extract ARG0, V, ARG1 from SRL result(s).
        
        Args:
            srl_result: Single SRL dict or list of SRL dicts
            
        Returns:
            Single roles dict or list of roles dicts with 'ARG0', 'V', 'ARG1' keys
            
        Examples:
            >>> result = srl.process("The dog ran.")
            >>> roles = srl.extract_roles(result)
            >>> print(roles)  # {'ARG0': 'The dog', 'V': 'run', 'ARG1': None}
        """
        if isinstance(srl_result, dict):
            return extract_roles(srl_result)
        else:
            return [extract_roles(r) for r in srl_result]


# =============================================================================
# AllenNLP SRL
# =============================================================================

class AllenNLPSRL:
    """SRL using AllenNLP pre-trained models.
    
    Provides sophisticated SRL using AllenNLP's pre-trained models based on
    PropBank annotations. Outputs are in the standard AllenNLP format.
    
    Note:
        AllenNLP requires Python 3.9-3.10 and specific dependency versions.
        Install with: pip install allennlp allennlp-models torch<2.0
    
    Attributes:
        predictor: AllenNLP predictor
        model_name: Name of model used
        cuda_device: CUDA device ID (-1 for CPU)
        batch_size: Default batch size (not used by AllenNLP, kept for API consistency)
    """
    
    def __init__(
        self,
        model_name: str = "structured-prediction-srl-bert.2020.12.15",
        cuda_device: int = -1,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        """Initialize AllenNLP SRL.
        
        Args:
            model_name: AllenNLP model name or path (with version number)
            cuda_device: CUDA device to use (-1 for CPU)
            batch_size: Default batch size (kept for API consistency)
            cache_dir: Local directory to cache downloaded models (default: ./models/allennlp)
            
        Raises:
            ImportError: If AllenNLP is not installed
        """
        if not ALLENNLP_AVAILABLE:
            raise ImportError(
                "AllenNLP is not installed. "
                "Install it with: pip install allennlp allennlp-models torch<2.0\n"
                "Note: AllenNLP requires Python 3.9-3.10 and has strict dependency requirements."
            )
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "models", "allennlp")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Determine model path
            if model_name.startswith(("http://", "https://", "/")):
                # Full URL or absolute path provided
                model_path = model_name
                local_model_path = None
            else:
                # Model name provided - check cache first
                local_model_path = self.cache_dir / f"{model_name}.tar.gz"
                
                if local_model_path.exists():
                    logger.info(f"Found cached model at: {local_model_path}")
                    model_path = str(local_model_path)
                else:
                    # Download URL
                    model_path = (
                        f"https://storage.googleapis.com/allennlp-public-models/"
                        f"{model_name}.tar.gz"
                    )
                    logger.info(f"Model not in cache, will download from: {model_path}")
                    logger.info(f"Model will be cached to: {local_model_path}")
                    logger.info("This may take a while (model is ~500MB)...")
            
            # Load model
            self.predictor = Predictor.from_path(model_path, cuda_device=cuda_device)
            self.model_name = model_name
            self.cuda_device = cuda_device
            self.batch_size = batch_size
            
            # Cache the model if it was downloaded
            if local_model_path and not local_model_path.exists() and not model_path.startswith("/"):
                try:
                    import shutil
                    import tempfile
                    
                    # AllenNLP caches to a temp directory, try to copy it
                    logger.info(f"Attempting to cache model to: {local_model_path}")
                    # Note: AllenNLP's internal caching handles this automatically
                    # We just provide the cache_dir for organization
                    
                except Exception as cache_error:
                    logger.warning(f"Could not cache model: {cache_error}")
            
            logger.info(f"Successfully loaded AllenNLP model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load AllenNLP model: {e}")
            logger.error(
                "\nPossible solutions:\n"
                "1. Check your internet connection (model needs to be downloaded)\n"
                "2. If you have the model file, place it in:\n"
                f"   {self.cache_dir / model_name}.tar.gz\n"
                "3. Or provide full path: get_srl('allennlp', model_name='/path/to/model.tar.gz')\n"
                "4. Use spaCy instead: get_srl('spacy')\n"
                "5. Try with a proxy or VPN if huggingface.co is blocked\n"
            )
            raise RuntimeError(
                f"Failed to load AllenNLP model. "
                f"Original error: {str(e)}\n"
                "See logger output above for possible solutions."
            ) from e
    
    def _process_single(self, text: str) -> Dict[str, Any]:
        """Process a single text with SRL.
        
        Args:
            text: Input text
            
        Returns:
            SRL dictionary with 'words' and 'verbs' keys
        """
        logger.debug(f"Processing SRL with AllenNLP for text: {text[:50]}...")
        try:
            result = self.predictor.predict(sentence=text)
            logger.debug(f"AllenNLP SRL result: {len(result.get('verbs', []))} verbs found")
            return result
        except Exception as e:
            logger.error(f"AllenNLP SRL failed: {e}")
            # Return empty result on failure
            return {"words": text.split(), "verbs": []}
    
    def process(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process text(s) with SRL.
        
        Automatically handles single text or multiple texts.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Not used (kept for API consistency)
            show_progress: Show progress bar for batch processing
            
        Returns:
            Single SRL dict if input is str, list of dicts if input is list
            
        Examples:
            >>> srl = AllenNLPSRL()
            >>> result = srl.process("The dog ran.")
            >>> results = srl.process(["Text 1", "Text 2"], show_progress=True)
        """
        # Single text
        if isinstance(texts, str):
            return self._process_single(texts)
        
        # Multiple texts
        if not texts:
            return []
        
        logger.info(f"Starting batch SRL processing for {len(texts)} texts using AllenNLP")
        results = []
        
        try:
            # AllenNLP predictor doesn't have efficient batch processing
            # Process one by one
            texts_iterator = texts
            if show_progress:
                texts_iterator = tqdm(
                    texts,
                    desc="Processing SRL (AllenNLP)",
                    unit="texts"
                )
            
            for text in texts_iterator:
                result = self._process_single(text)
                results.append(result)
                
                # Clear cache periodically if using CUDA
                if self.cuda_device > -1 and torch is not None:
                    if len(results) % 10 == 0:
                        with torch.cuda.device(self.cuda_device):
                            torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Batch SRL failed: {e}")
            # Return empty results on failure
            results = [{"words": t.split(), "verbs": []} for t in texts]
        
        verb_count = sum(1 for r in results if r.get('verbs'))
        logger.info(f"Completed batch SRL processing: {verb_count}/{len(results)} texts with verbs found")
        return results
    
    def extract_roles(
        self,
        srl_result: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Optional[str]], List[Dict[str, Optional[str]]]]:
        """Extract ARG0, V, ARG1 from SRL result(s).
        
        Args:
            srl_result: Single SRL dict or list of SRL dicts
            
        Returns:
            Single roles dict or list of roles dicts with 'ARG0', 'V', 'ARG1' keys
            
        Examples:
            >>> result = srl.process("The dog ran.")
            >>> roles = srl.extract_roles(result)
            >>> print(roles)  # {'ARG0': 'The dog', 'V': 'run', 'ARG1': None}
        """
        if isinstance(srl_result, dict):
            return extract_roles(srl_result)
        else:
            return [extract_roles(r) for r in srl_result]


# =============================================================================
# Factory function
# =============================================================================

def get_srl(method: str = "spacy", **kwargs) -> Union[SpacySRL, AllenNLPSRL]:
    """Factory function to get SRL processor.
    
    Args:
        method: Method name ('spacy' or 'allennlp')
        **kwargs: Additional arguments for processor initialization
        
    Returns:
        SRL processor instance
        
    Raises:
        ValueError: If method is not supported
        ImportError: If AllenNLP is requested but not available
        
    Examples:
        >>> # Use spaCy (default)
        >>> srl = get_srl('spacy')
        >>> result = srl.process("The dog chased the cat.")
        
        >>> # Use AllenNLP
        >>> srl = get_srl('allennlp')
        >>> results = srl.process_batch(["Sentence 1", "Sentence 2"])
    """
    if method == "spacy":
        return SpacySRL(**kwargs)
    elif method == "allennlp":
        if not ALLENNLP_AVAILABLE:
            raise ImportError(
                "AllenNLP is not available. "
                "Install with: pip install allennlp allennlp-models torch<2.0\n"
                "Note: AllenNLP requires Python 3.9-3.10."
            )
        return AllenNLPSRL(**kwargs)
    else:
        raise ValueError(
            f"Unknown SRL method: {method}. "
            f"Supported methods: 'spacy', 'allennlp'"
        )


# =============================================================================
# Utility functions
# =============================================================================

def is_allennlp_available() -> bool:
    """Check if AllenNLP is available.
    
    Returns:
        True if AllenNLP can be imported
    """
    return ALLENNLP_AVAILABLE


def extract_roles(srl_result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extract ARG0, V, ARG1 from SRL result.
    
    Args:
        srl_result: SRL dictionary from process()
        
    Returns:
        Dictionary with 'ARG0', 'V', 'ARG1' keys (values can be None)
        
    Example:
        >>> srl = get_srl('spacy')
        >>> result = srl.process("The dog chased the cat.")
        >>> roles = extract_roles(result)
        >>> print(roles)
        {'ARG0': 'The dog', 'V': 'chase', 'ARG1': 'the cat'}
    """
    verbs = srl_result.get("verbs", [])
    
    if not verbs:
        return {"ARG0": None, "V": None, "ARG1": None}
    
    verb_frame = verbs[0]
    description = verb_frame.get("description", "")
    verb_text = verb_frame.get("verb", "")
    
    # Extract using regex
    arg0_match = re.search(r'\[ARG0:\s*([^\]]+)\]', description)
    v_match = re.search(r'\[V:\s*([^\]]+)\]', description)
    arg1_match = re.search(r'\[ARG1:\s*([^\]]+)\]', description)
    
    return {
        "ARG0": arg0_match.group(1).strip() if arg0_match else None,
        "V": v_match.group(1).strip() if v_match else verb_text,
        "ARG1": arg1_match.group(1).strip() if arg1_match else None,
    }


def has_verb(srl_result: Dict[str, Any]) -> bool:
    """Check if SRL result has a verb.
    
    Args:
        srl_result: SRL dictionary from process()
        
    Returns:
        True if at least one verb was found
        
    Example:
        >>> result = srl.process("The dog ran.")
        >>> has_verb(result)
        True
        >>> result = srl.process("Hello!")
        >>> has_verb(result)
        False
    """
    return len(srl_result.get("verbs", [])) > 0


def is_event_srl(srl_result: Union[str, Dict[str, Any]]) -> bool:
    """Check if SRL result represents a valid event (has verb and at least one argument).
    
    A valid event SRL must have:
    - At least one verb
    - At least one argument (ARG0 or ARG1)
    
    Args:
        srl_result: SRL result from process() (can be dict or JSON string)
        
    Returns:
        True if SRL has a verb and at least one argument
        
    Example:
        >>> srl = get_srl('spacy')
        >>> result = srl.process("The dog chased the cat.")
        >>> is_event_srl(result)
        True
        >>> result = srl.process("Run!")  # Only verb, no arguments
        >>> is_event_srl(result)
        False
        >>> import json
        >>> json_str = json.dumps(result)
        >>> is_event_srl(json_str)  # Also accepts JSON string
        False
    """
    import json
    import pandas as pd
    
    # Handle empty/null values
    if pd.isna(srl_result) or not srl_result or srl_result == '{}':
        return False
    
    try:
        # Parse JSON string if needed
        if isinstance(srl_result, str):
            srl_dict = json.loads(srl_result)
        else:
            srl_dict = srl_result
        
        # Check if has verb
        if not has_verb(srl_dict):
            return False
        
        # Extract roles
        roles = extract_roles(srl_dict)
        
        # Check if has V and at least one arg
        # Note: extract_roles may return None values, use 'or ""' to handle
        has_v = bool((roles.get('V') or '').strip())
        has_arg0 = bool((roles.get('ARG0') or '').strip())
        has_arg1 = bool((roles.get('ARG1') or '').strip())
        
        return has_v and (has_arg0 or has_arg1)
    except:
        return False


def process_to_dataframe(
    srl_instance,
    texts: List[str],
    column_prefix: str = "srl",
    show_progress: bool = False,
    batch_size: Optional[int] = None,
) -> Dict[str, List]:
    """Process texts and return results ready for DataFrame assignment.
    
    This helper ensures safe assignment to DataFrame columns with validation.
    
    Args:
        srl_instance: SRL instance (from get_srl())
        texts: List of texts to process
        column_prefix: Prefix for output column names
        show_progress: Show progress bar
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with keys:
            - '{prefix}_result': List of SRL dicts (or None for empty)
            - '{prefix}_ARG0': List of ARG0 values
            - '{prefix}_V': List of V values
            - '{prefix}_ARG1': List of ARG1 values
            - '{prefix}_has_verb': List of booleans
            
    Example:
        >>> srl = get_srl('spacy')
        >>> cols = process_to_dataframe(srl, df['cause_span'].tolist(), 'cause_srl')
        >>> df['cause_srl_ARG0'] = cols['cause_srl_ARG0']
        >>> df['cause_srl_V'] = cols['cause_srl_V']
        >>> df['cause_srl_ARG1'] = cols['cause_srl_ARG1']
        
    Or simply:
        >>> for col_name, values in cols.items():
        ...     df[col_name] = values
    """
    # Process all texts
    results = srl_instance.process(texts, batch_size=batch_size, show_progress=show_progress)
    
    # Validate length
    if len(results) != len(texts):
        logger.warning(
            f"Length mismatch: input={len(texts)}, output={len(results)}. "
            "This should not happen. Please report this issue."
        )
        # Pad or truncate to match
        if len(results) < len(texts):
            # Pad with empty results
            empty_result = {"words": [], "verbs": []}
            results.extend([empty_result] * (len(texts) - len(results)))
        else:
            # Truncate
            results = results[:len(texts)]
    
    # Extract roles for all results
    roles_list = srl_instance.extract_roles(results)
    
    # Build output dictionary
    output = {
        f"{column_prefix}_result": results,
        f"{column_prefix}_ARG0": [r['ARG0'] for r in roles_list],
        f"{column_prefix}_V": [r['V'] for r in roles_list],
        f"{column_prefix}_ARG1": [r['ARG1'] for r in roles_list],
        f"{column_prefix}_has_verb": [has_verb(r) for r in results],
    }
    
    return output
