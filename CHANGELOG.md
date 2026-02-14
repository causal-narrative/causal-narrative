# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-14

### Added
- Initial release of causal-narrative package
- Causal relation detection with multiple methods:
  - Heuristic pattern-based detection
  - LLM-based detection (OpenAI)
  - BERT-based detection
- Cause/effect span extraction:
  - Pattern-based extraction
  - LLM-based extraction
  - BERT-based extraction
- Semantic Role Labeling (SRL):
  - spaCy-based SRL (fast, dependency parsing)
  - AllenNLP-based SRL (accurate, transformer-based)
- Event clustering:
  - Role-based embedding (ARG0-V-ARG1)
  - Phrase-based embedding
  - Multiple clustering algorithms (DP-Means, K-Means, HDBSCAN)
  - Automatic cluster naming
- Causal network construction and visualization:
  - NetworkX-based graph construction
  - Interactive visualization with PyVis
  - PNG export support with Playwright
- Built-in datasets for testing and examples
- Comprehensive tutorial notebooks
- Full API documentation in docstrings

### Features
- Unified API for detection and extraction
- Flexible backend selection (heuristic/LLM/BERT)
- Batch processing support
- Export results to CSV/JSON
- Pydantic models for type safety
- Extensive configuration options

### Dependencies
- Python 3.8+ (3.9-3.10 for AllenNLP support)
- Core: pandas, numpy, nltk, spacy, scikit-learn
- Embeddings: sentence-transformers
- Clustering: hdbscan, pdc-dp-means
- Visualization: networkx, matplotlib, seaborn, pyvis
- Optional: allennlp, allennlp-models, playwright

[0.1.0]: https://github.com/causalis-nlp/causal-narrative/releases/tag/v0.1.0
