# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-14

### Added
- **Chinese language support**:
  - HanLP-based SRL for Chinese text analysis
  - Multilingual BERT embedding models with automatic language detection
  - Chinese tutorial notebook (`notebook/tutorial_minimal_zh.ipynb`)
- New SRL backend: `HanLPSRL` class for Chinese semantic role labeling
- Language detection utilities: `detect_language()` and `get_default_model_for_language()`
- `DEFAULT_CHINESE_MODEL_NAME` constant for Chinese embedding model
- `is_hanlp_available()` function to check HanLP installation

### Changed
- Updated `get_srl()` factory function to support 'hanlp' method
- Enhanced `load_embedder()` to accept `language` parameter for automatic model selection
- Updated README with comprehensive Chinese language support documentation
- Improved installation guide with separate section for Chinese language features

### Documentation
- Added Chinese language examples in README
- New tutorial: `tutorial_minimal_zh.ipynb` demonstrating complete Chinese workflow
- Updated API documentation for new Chinese-related functions

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
