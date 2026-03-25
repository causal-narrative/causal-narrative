# `Causal-Narrative`

A Python package for extracting and analyzing causal narratives from text using semantic role labeling and event clustering.

This package accompanies our paper: **Mapping the Causal Narratives in Political Communication Using Large Language Models** *(in submission).*

## What can this package do?

### 1. Causal Relation Detection and Extraction

Identify causal relationships in text and extract cause/effect spans:

- **Pattern-based detection**: Uses linguistic patterns and connectives (e.g., "because", "therefore", "leads to")
- **Classifier-based detection**: Machine learning models for causal relation classification
- **LLM-based detection**: Large language model prompting for complex causal reasoning
- **Span extraction**: Extract cause and effect spans from causal sentences

**Example:**
```
Input: "The pandemic caused widespread unemployment."
Output: {
  "is_causality": True,
  "cause_span": "The pandemic",
  "effect_span": "widespread unemployment"
}
```

### 2. Semantic Role Labeling (SRL)

Extract semantic roles (Agent-Verb-Patient / ARG0-V-ARG1) from causal spans:

- **Dependency parsing SRL** (English): Fast, dependency parsing-based extraction using spaCy
- **AllenNLP SRL** (English): More accurate, transformer-based extraction
- **HanLP SRL** (Chinese): Semantic role labeling for Chinese text

**Example (English):**
```
Input: "The government raised interest rates."
Output: {
  "ARG0": "The government",
  "V": "raised",
  "ARG1": "interest rates"
}
```

**Example (Chinese):**
```
Input: "政府提高了利率。"
Output: {
  "ARG0": "政府",
  "V": "提高",
  "ARG1": "利率"
}
```

### 3. Event Clustering

Group similar causal events into interpretable clusters:

- **Role-based Event Embedding**: Separately embed ARG0, V, ARG1 and concatenate
- **Phrase-based Embedding**: Directly embed raw text spans
- **Multiple clustering algorithms**: DP-Means, K-Means, HDBSCAN
- **Automatic event naming**: Use most frequent SVO or phrase as cluster name

**Example:**
```
Cluster 1: "government raised interest rates"
  - "The Fed increased interest rates"
  - "Central bank raised rates"
  - "Monetary policy tightened"
  
Cluster 2: "pandemic caused unemployment"
  - "COVID-19 led to job losses"
  - "The virus caused layoffs"
```

### 4. Causal Network Construction

Build and visualize causal networks from clustered events:

- **Network graphs**: Directed graphs of cause → effect relationships
- **Community detection**: Identify narrative themes
- **Interactive visualization**: Explore causal narratives

## Installation

### Python Requirements

- **Python 3.8+** for basic features
- **Python 3.9-3.10** for AllenNLP SRL support

### Language Support

- **English**: Full support with spaCy, AllenNLP, and BERT models
- **Chinese (中文)**: Supported with HanLP SRL and multilingual BERT embedding models

### Option 1: Full Installation (includes AllenNLP SRL)

**Use Python 3.9 or 3.10 only**

```bash
# Create environment with Python 3.10
conda create -n causal-narrative python=3.10 -y
conda activate causal-narrative

# Install causal-narrative with AllenNLP support
python -m pip install -U pip wheel setuptools
python -m pip install -U 'causal-narrative[allennlp]'

# Download spaCy model (for English)
python -m spacy download en_core_web_sm
```

**Important Notes for AllenNLP:**
- The correct model URL is: `https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz`
- Models are cached in `~/.allennlp/` after first download
- If you encounter network issues, download the model manually and specify the local path

### Option 2: Without AllenNLP SRL (Dependency Parsing only)

**Can use Python 3.8, 3.9, 3.10, 3.11, or 3.12**

```bash
# Create environment
conda create -n causal-narrative python=3.11 -y
conda activate causal-narrative

# Install causal-narrative without AllenNLP
python -m pip install -U pip wheel setuptools
python -m pip install -U causal-narrative

# Download spaCy model (for English)
python -m spacy download en_core_web_sm
```

**What you get:**
- ✅ Causal relation detection
- ✅ Dependency parsing-based SRL (faster, good for most cases)
- ✅ Event clustering
- ✅ Network construction and visualization
- ❌ AllenNLP-based SRL (more accurate, but requires Python 3.9-3.10)

### Option 3: Chinese Language Support

**For Chinese text analysis, install jieba and optionally HanLP:**

```bash
# Basic Chinese support (recommended - stable)
pip install 'causal-narrative[chinese]'

# Test Chinese support
python -c "import jieba; print('Jieba available:', True)"
```

**Chinese Features:**
- ✅ Jieba-based SRL for Chinese text (lightweight, stable)
- ✅ Multilingual BERT embedding models (automatic language detection)
- ✅ Same clustering and visualization as English

**Note on HanLP:**
HanLP provides more sophisticated Chinese SRL but may have compatibility issues with newer transformers versions. If you encounter `AttributeError: BertTokenizer has no attribute encode_plus`, the jieba-based fallback will be used automatically.

To resolve HanLP compatibility issues:
```bash
pip install 'transformers<4.31'
```

**Example Usage (Chinese):**
```python
from causal_narrative import get_srl, SentenceEmbedder

# Initialize Chinese SRL
srl = get_srl('hanlp')
result = srl.process("政府提高了利率。")

# Initialize Chinese embedding model
from causal_narrative.embedding import DEFAULT_CHINESE_MODEL_NAME
embedder = SentenceEmbedder(model_name=DEFAULT_CHINESE_MODEL_NAME)
```

**See Tutorial:** Check `notebook/tutorial_minimal_zh.ipynb` for a complete Chinese example.

### Important: DP-Means Clustering with Cosine Similarity

The **DP-Means clustering** feature uses a specialized implementation based on cosine similarity for clustering sentence embeddings. This requires a custom installation.

#### Standard Installation

The package uses `pdc-dp-means` by default, which can be installed via pip:

```bash
pip install pdc-dp-means
```

#### Advanced: Custom DP-Means with Cosine Similarity

For users who need the specialized **MiniBatch PDC-DP-Means via Cosine Similarity** implementation (removes random initialization, optimized for sentence embeddings), follow these steps:

**Important**: This approach requires building scikit-learn from source and has specific version requirements.

**Version Requirements:**
```
scikit-learn>=1.2,<1.3
numpy>=1.23.0,<2.0
```

**Installation Steps:**

1. **Clone the specialized DP-Means implementation:**
   ```bash
   git clone https://github.com/hanshanley/narrative-influence.git
   cd narrative-influence/dpmeans_clustering
   ```

2. **Clone scikit-learn:**
   ```bash
   git clone https://github.com/scikit-learn/scikit-learn.git
   cd scikit-learn
   git checkout 1.2.2  # Use version 1.2.x
   ```

3. **Replace scikit-learn files:**
   ```bash
   # Copy the modified files from narrative-influence/dpmeans_clustering
   # to sklearn/cluster/ in your scikit-learn clone:
   # - __init__.py
   # - _k_means_lloyd.pyx
   # - _kmeans.py
   ```

4. **Build and install scikit-learn from source:**
   
   Follow the official guide: https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge
   
   ```bash
   pip install --editable . --no-build-isolation
   ```

5. **Verify installation:**
   ```python
   from sklearn.cluster import MiniBatchDPMeans, DPMeans
   print("DP-Means with cosine similarity installed successfully!")
   ```

**Usage:**

Once installed, you can use DP-Means just like K-Means:

```python
from sklearn.cluster import MiniBatchDPMeans

clusterer = MiniBatchDPMeans(
    delta=0.1,           # Distance threshold parameter
    batch_size=50,       # Batch size for MiniBatch variant
    random_state=42
)
labels = clusterer.fit_predict(embeddings)
```

**Reference:**
- Original implementation: [BGU-CS-VIL/pdc-dp-means](https://github.com/BGU-CS-VIL/pdc-dp-means/tree/main/paper_code)
- Cosine similarity version: [hanshanley/narrative-influence/dpmeans_clustering](https://github.com/hanshanley/narrative-influence/tree/main/dpmeans_clustering)

**When to use this custom version:**
- You need cosine similarity metric (standard DP-Means uses Euclidean distance)
- You're clustering sentence embeddings with no random initialization
- You have specific performance requirements for large-scale clustering


## Tutorials

Please see our hands-on tutorials in the `notebook/` directory:

- **`tutorial_minimal.ipynb`**: A minimal runnable tutorial (~2 mins). Designed for quick execution and understanding of the core pipeline.
- **`tutorial_minimal_zh.ipynb`**: Chinese version of the minimal tutorial (中文版最小实例教程)
- **`tutorial_trump.ipynb`**: Complete pipeline for the Trump Tweet Archive




## Citation

If you use this package in your research, please cite:

```bibtex
@software{causal_narrative,
  title = {Mapping the Causal Narratives in Political Communication Using Large Language Models},
  year = {2026},
  url = {https://github.com/causal-narrative/causal-narrative}
}
```

## License

MIT License - see LICENSE file for details

## Changelog

### Version 0.1.0 (2026-02-14)

- Initial release
- Causal detection with pattern, classifier, and LLM approaches
- Semantic role labeling with spaCy and AllenNLP
- Event clustering with Role-based and Phrase-based strategies
- Support for DP-Means, K-Means, and HDBSCAN
- Causal network construction and visualization
- Complete tutorial notebooks

## Disclaimer

This is a research tool designed for academic and experimental purposes. Results should be validated for production use.
