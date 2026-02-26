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

- **spaCy-based SRL**: Fast, dependency parsing-based extraction
- **AllenNLP-based SRL**: More accurate, transformer-based extraction

**Example:**
```
Input: "The government raised interest rates."
Output: {
  "ARG0": "The government",
  "V": "raised",
  "ARG1": "interest rates"
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

### Option 1: Full Installation (includes AllenNLP SRL)

**Use Python 3.9 or 3.10 only**

```bash
# Create environment with Python 3.10
conda create -n causal-narrative python=3.10 -y
conda activate causal-narrative

# Install causal-narrative with AllenNLP support
python -m pip install -U pip wheel setuptools
python -m pip install -U 'causal-narrative[allennlp]'

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Option 2: Without AllenNLP SRL (Dependency Parsing only)

**Can use Python 3.8, 3.9, 3.10, 3.11, or 3.12**

```bash
# Create environment
conda create -n causal-narrative python=3.11 -y
conda activate causal-narrative

# Install causal-narrative without AllenNLP
python -m pip install -U pip wheel setuptools
python -m pip install -U causal-narrative

# Download spaCy model
python -m spacy download en_core_web_sm
```

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

Please see our hands-on tutorials in the `notebooks/` directory:

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
