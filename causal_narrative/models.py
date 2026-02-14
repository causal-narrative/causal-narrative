"""Core data models for the causal narrative lab."""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Pydantic v1/v2 compatibility
# Use v1 validator syntax which is compatible with both versions
from pydantic import BaseModel, Field, validator


class SentenceRecord(BaseModel):
    """A sentence record with metadata."""

    doc_id: str = Field(..., description="Document identifier")
    sent_id: int = Field(..., description="Sentence index within document")
    text: str = Field(..., description="Sentence text")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = False


class CausalDecision(BaseModel):
    """Result of causal presence detection."""

    has_causality: bool = Field(..., description="Whether causality is present")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    rationale: Optional[str] = Field(None, description="Explanation for the decision")
    causal_type: Optional[str] = Field(None, description="Type of causality if present")
    model: str = Field(..., description="Model used for detection")
    prompt_version: str = Field(..., description="Version of prompt template used")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = False


class CausalSpan(BaseModel):
    """Extracted cause and effect spans from a sentence."""

    cause_text: str = Field(..., description="Cause span text")
    effect_text: str = Field(..., description="Effect span text")
    cause_char_span: Tuple[int, int] = Field(..., description="(start, end) indices for cause")
    effect_char_span: Tuple[int, int] = Field(..., description="(start, end) indices for effect")
    is_valid: bool = Field(default=True, description="Whether spans are validated")
    notes: Optional[str] = Field(None, description="Additional notes or issues")

    @validator('cause_char_span', 'effect_char_span')
    def validate_span(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate that span indices are non-negative and start < end."""
        start, end = v
        if start < 0 or end < 0:
            raise ValueError(f"Span indices must be non-negative, got ({start}, {end})")
        if start > end:
            raise ValueError(f"Span start must be <= end, got ({start}, {end})")
        return v

    class Config:
        frozen = False


class EventTuple(BaseModel):
    """A semantic role labeling event representation."""

    a0: Optional[str] = Field(None, description="Agent/subject (ARG0)")
    v: str = Field(..., description="Verb/predicate")
    a1: Optional[str] = Field(None, description="Patient/object (ARG1)")
    a2: Optional[str] = Field(None, description="Additional argument (ARG2)")
    temporal: Optional[str] = Field(None, description="Temporal information")
    location: Optional[str] = Field(None, description="Location information")

    role_schema: str = Field(default="A0/V/A1", description="Schema used (e.g., 'A0/V/A1')")
    source: str = Field(..., description="Source: 'cause', 'effect', or 'sentence'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    method: str = Field(..., description="Extraction method (e.g., 'spacy', 'allennlp')")

    def to_text(self, include_missing: bool = False) -> str:
        """Convert event tuple to linearized text representation.

        Args:
            include_missing: If True, include None values as 'None'

        Returns:
            Linearized string representation
        """
        parts = []
        if self.a0 or include_missing:
            parts.append(f"A0={self.a0 or 'None'}")
        parts.append(f"V={self.v}")
        if self.a1 or include_missing:
            parts.append(f"A1={self.a1 or 'None'}")
        if self.a2:
            parts.append(f"A2={self.a2}")
        if self.temporal:
            parts.append(f"TEMP={self.temporal}")
        if self.location:
            parts.append(f"LOC={self.location}")
        return " | ".join(parts)

    class Config:
        frozen = False


class ClusteredEvent(BaseModel):
    """An event with cluster assignment."""

    event_id: str = Field(..., description="Unique event identifier")
    event_text: str = Field(..., description="Linearized event text")
    cluster_id: int = Field(..., description="Assigned cluster ID")
    cluster_label: Optional[str] = Field(None, description="Human-readable cluster label")
    exemplar_texts: List[str] = Field(default_factory=list, description="Representative examples")
    centroid_distance: Optional[float] = Field(None, description="Distance to cluster centroid")

    # Original event information
    original_event: Optional[EventTuple] = Field(None, description="Original event tuple")
    source_sent_id: Optional[str] = Field(None, description="Source sentence ID")

    class Config:
        frozen = False


class CausalEdge(BaseModel):
    """An edge in the causal narrative network."""

    cause_cluster_id: int = Field(..., description="Source cluster (cause)")
    effect_cluster_id: int = Field(..., description="Target cluster (effect)")
    weight: float = Field(..., ge=0.0, description="Edge weight (frequency or score)")

    example_pairs: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Example (cause_text, effect_text) pairs"
    )

    # Optional metadata
    cause_label: Optional[str] = Field(None, description="Cause cluster label")
    effect_label: Optional[str] = Field(None, description="Effect cluster label")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "source": self.cause_cluster_id,
            "target": self.effect_cluster_id,
            "weight": self.weight,
            "cause_label": self.cause_label,
            "effect_label": self.effect_label,
            "num_examples": len(self.example_pairs),
        }

    class Config:
        frozen = False


class ClusterSummary(BaseModel):
    """Summary statistics for a cluster."""

    cluster_id: int = Field(..., description="Cluster ID")
    label: Optional[str] = Field(None, description="Cluster label")
    size: int = Field(..., ge=1, description="Number of events in cluster")
    exemplars: List[str] = Field(..., description="Representative event texts")
    keywords: List[str] = Field(default_factory=list, description="Key terms")
    centroid_text: Optional[str] = Field(None, description="Centroid representation")

    class Config:
        frozen = False


class ValidationResult(BaseModel):
    """Result of validation checks."""

    is_valid: bool = Field(..., description="Overall validation status")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = False
