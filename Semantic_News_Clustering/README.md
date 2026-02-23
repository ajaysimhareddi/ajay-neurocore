<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f0c29,50:302b63,100:24243e&height=160&text=🗞️%20Semantic%20News%20Clustering&fontSize=40&fontColor=ffffff&fontAlignY=45&desc=Sentence-BERT%20·%20Louvain%20Graph%20Communities%20·%20UMAP%20·%20BERT%20Sentiment&descSize=14&descAlignY=72&animation=blinking"/>

<br/>

```
┌─────────────────────────────────────────────────────────────────┐
│  SBERT Embeddings  →  Cosine Similarity Graph  →  Louvain       │
│  K-Means Baseline  →  UMAP 2D Projection  →  ARI / NMI / Sil.  │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

<img src="https://img.shields.io/badge/Model-all--MiniLM--L6--v2-302b63?style=for-the-badge&logo=huggingface&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-HuffPost%20News%2050K-0f0c29?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/Clustering-Louvain%20%2B%20K--Means-24243e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Visualization-UMAP%20%2B%20Plotly-5c6bc0?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>

</div>

---

## Overview

This project implements a **hybrid semantic news clustering system** that combines dense contextual embeddings from Sentence-BERT with graph-theoretic community detection to group semantically similar news articles without any labeled training data. It compares this graph-based approach against a traditional K-Means baseline across multiple internal and external evaluation metrics.

The pipeline operates on the **HuffPost News Category Dataset v3**, processing 5,000 articles from a larger 200K+ corpus. Each article's headline and short description are combined, preprocessed, and encoded into a 384-dimensional semantic embedding using the `all-MiniLM-L6-v2` Sentence-BERT model. A cosine similarity graph is constructed over these embeddings, and Louvain community detection partitions the graph into semantically coherent clusters without requiring a pre-specified cluster count — a key advantage over K-Means.

A separate BERT sentiment analysis module (`bert-base-uncased` fine-tuned for binary classification) is also included, demonstrating how the same BERT family of models can be adapted for both unsupervised clustering and supervised sentiment tasks.

---

## Research Objective

> **Can graph-based community detection on a Sentence-BERT similarity graph discover semantically coherent news topic clusters that outperform traditional K-Means clustering — and how do both methods align with ground-truth editorial category labels?**

---

## Dataset

| Property | Value |
|---|---|
| Name | HuffPost News Category Dataset v3 |
| File | `News_Category_Dataset_v3.json` (JSON Lines format) |
| Full size | ~200,000 articles |
| Subset used | 5,000 articles (`data.head(5000)`) |
| Key columns | `headline`, `short_description`, `category` |
| Input field | `headline + " " + short_description` (concatenated, then cleaned) |
| Ground truth | `category` column — factorized to integer labels for external evaluation |

The dataset spans multiple editorial categories including Politics, Entertainment, Travel, Tech, Sports, and others. Category labels are used only during external evaluation (ARI, NMI) — not during clustering — making this a genuinely unsupervised pipeline.

---

## System Architecture

```
News_Category_Dataset_v3.json  (5,000 articles)
              │
              ▼
   preprocess_text()
   Lowercase · strip special chars · whitespace normalize
              │
              ▼
   ┌──────────────────────────────────────────┐
   │      CLASSICAL VECTORIZATION             │
   │  BOW Binary      (5000 × 5000)           │
   │  BOW Frequency   (5000 × 5000)           │
   │  One-Hot Encoding (word level)           │
   │  TF-IDF          (5000 × 5000)           │
   └──────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────┐
   │   SENTENCE-BERT  (all-MiniLM-L6-v2)      │
   │   Input: cleaned article strings         │
   │   Output: embeddings  [5000 × 384]       │
   └──────────────────────────────────────────┘
              │
              ▼
   Cosine Similarity Matrix  [5000 × 5000]
   Threshold = 0.70
   Edge added if sim(i,j) > 0.70
              │
              ▼
   NetworkX Graph  G
   Nodes = 5000 articles
   Edges = pairs above threshold
              │
        ┌─────┴──────┐
        ▼            ▼
   Louvain        K-Means
   community_louvain   KMeans(n_clusters=
   .best_partition(G)  num_clusters_louvain)
        │            │
        ▼            ▼
   graph_labels   kmeans_labels
        │            │
        └─────┬──────┘
              ▼
   UMAP  (n_neighbors=15, min_dist=0.1)
   embeddings [5000×384] → embedding_2d [5000×2]
              │
              ▼
   ┌────────────────────────────────────────────┐
   │         EVALUATION                         │
   │  Silhouette Score  (internal)              │
   │  Adjusted Rand Index  (external vs. true)  │
   │  Normalized Mutual Information  (external) │
   └────────────────────────────────────────────┘
              │
              ▼
   Interactive Plotly Visualizations
   UMAP scatter · Network graph · Bar comparison
```

---

## Text Preprocessing

All articles pass through `preprocess_text()` before vectorization or embedding:

```python
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()
```

This removes HTML entities, punctuation, and special characters while preserving alphanumeric tokens and whitespace. The preprocessed string is then used both for classical vectorization and as input to the SBERT encoder. Tokenized word lists (`article.split()`) are additionally used for BOW and one-hot encoding steps.

---

## Feature Representations

### Classical Vectorization

Four classical text representations are computed to contextualize the baseline before moving to semantic embeddings:

| Method | Representation | Shape |
|---|---|---|
| BOW Binary | 1 if word present, 0 otherwise | `[5000 × 5000]` |
| BOW Frequency | Raw word count per document | `[5000 × 5000]` |
| One-Hot Encoding | Word-level, full vocabulary | `[5000 × vocab_size]` |
| TF-IDF | Term frequency × inverse document frequency | `[5000 × 5000]` |

All four use `max_features=5000` as the vocabulary limit via `CountVectorizer` / `TfidfVectorizer`.

### Sentence-BERT Embeddings

The `all-MiniLM-L6-v2` model from the `sentence-transformers` library generates dense contextual embeddings:

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Encoding | `model.encode(articles, show_progress_bar=True)` |
| Output | `embeddings` array of shape `[5000, 384]` |

Unlike TF-IDF, SBERT embeddings capture semantic meaning — two articles discussing the same event in different words will produce geometrically proximate vectors, while topically unrelated articles will be distant regardless of vocabulary overlap.

---

## Graph Construction

A cosine similarity matrix is computed over all 5,000 SBERT embedding pairs, producing a dense `[5000 × 5000]` matrix. A NetworkX undirected graph is constructed by adding an edge between articles `i` and `j` if and only if their cosine similarity exceeds the threshold of **0.70**:

```python
sim_matrix = cosine_similarity(embeddings)
threshold = 0.7
G = nx.Graph()
for i in range(len(articles)):
    G.add_node(i, label=articles[i])
    for j in range(i+1, len(articles)):
        if sim_matrix[i][j] > threshold:
            G.add_edge(i, j, weight=sim_matrix[i][j])
```

The threshold of 0.70 was selected to balance graph density — too low a threshold produces a near-complete graph where community detection is trivial; too high produces an overly sparse graph with many isolated nodes. The resulting graph structure encodes semantic proximity as graph connectivity, transforming the clustering problem into a community detection problem.

---

## Clustering

### Louvain Community Detection

Louvain (`community_louvain.best_partition(G)`) maximizes the **modularity** of the graph partition — a measure of how densely connected nodes within communities are relative to a random graph baseline. The algorithm iteratively reassigns nodes to communities that maximally increase modularity, then collapses communities into super-nodes and repeats.

**Key advantage over K-Means:** Louvain automatically determines the number of communities from the graph structure. No `k` needs to be specified in advance.

```python
partition = community_louvain.best_partition(G)
graph_labels = [partition[i] for i in range(len(articles))]
num_clusters_louvain = len(set(graph_labels))
```

### K-Means Clustering

K-Means is applied directly on the 384-dimensional SBERT embeddings using the same cluster count as Louvain, enabling a fair comparison:

```python
num_clusters = num_clusters_louvain
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(embeddings)
```

K-Means minimizes within-cluster sum of squared distances in embedding space. Unlike Louvain, it cannot adapt to non-spherical or irregularly shaped cluster geometries.

---

## Dimensionality Reduction and Visualization

**UMAP** (Uniform Manifold Approximation and Projection) reduces the 384-dimensional SBERT embeddings to 2D for visualization, preserving both local neighborhood structure and global topology better than PCA or t-SNE:

```python
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)
```

Three interactive **Plotly** visualizations are generated:

| Visualization | Description |
|---|---|
| UMAP Scatter — Louvain | 2D scatter of all articles, coloured by Louvain community label (Viridis colormap, hover shows article text) |
| UMAP Scatter — K-Means | Same projection, coloured by K-Means cluster assignment for side-by-side comparison |
| Network Graph | Force-directed layout (`nx.spring_layout`, `k=0.15`, `iterations=20`, `seed=42`) with edge traces and node traces coloured by Louvain community |
| Silhouette Bar Chart | Grouped bar chart comparing Louvain vs. K-Means silhouette scores |

The network graph visualization renders graph edges as `go.Scatter` line traces (grey, `width=0.5`) and nodes as scatter markers coloured by `partition[node]`, with article text as hover labels.

---

## Evaluation

### Internal Evaluation — Silhouette Score

Silhouette score measures how similar each article is to its own cluster versus the nearest alternative cluster. Range: `[-1, 1]`, higher is better.

```python
silhouette_graph  = silhouette_score(embeddings, graph_labels)
silhouette_kmeans = silhouette_score(embeddings, kmeans_labels)
```

Example reference values used in visualization: Louvain `≈ 0.250`, K-Means `≈ 0.285`.

### External Evaluation — ARI and NMI

The original `category` column is factorized into integer labels and used as ground truth. Two external metrics compare clustering assignments against these editorial labels:

| Metric | Formula basis | Interpretation |
|---|---|---|
| **Adjusted Rand Index (ARI)** | Pair-counting, chance-adjusted | 1.0 = perfect agreement · 0.0 = random · negative = worse than random |
| **Normalized Mutual Information (NMI)** | Information-theoretic, normalized | 1.0 = perfect mutual information · 0.0 = independent |

```python
ari_graph   = adjusted_rand_score(true_labels, graph_labels)
ari_kmeans  = adjusted_rand_score(true_labels, kmeans_labels)
nmi_graph   = normalized_mutual_info_score(true_labels, graph_labels)
nmi_kmeans  = normalized_mutual_info_score(true_labels, kmeans_labels)
```

ARI and NMI are computed for both Louvain and K-Means, providing a direct measure of how well each unsupervised method recovers the original editorial topic structure.

---

## BERT Sentiment Analysis Module

A second module in the notebook documents a fine-tuned BERT architecture for binary sentiment classification, demonstrating how the BERT encoder backbone can be adapted from unsupervised embedding generation to supervised classification:

```
Input Text
     │
     ▼
BertTokenizer (bert-base-uncased)
     ├─ Tokenization + [CLS] / [SEP] insertion
     ├─ input_ids + attention_mask
     ▼
BERT Encoder (bert-base-uncased)
     ├─ 12 Transformer Encoder Layers
     ├─ Multi-Head Self-Attention
     ├─ Hidden Size = 768
     ▼
[CLS] Token Representation
     ▼
Classification Head
     ├─ Linear Layer  768 → 2
     ├─ Logits: [negative_score, positive_score]
     ▼
Softmax → Predicted Label (0 = Negative · 1 = Positive)
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Loss function | `CrossEntropyLoss` |
| Optimizer | `AdamW` |
| Batch size | 2 |
| Epochs | 10 |
| Max sequence length | 128 tokens |
| Padding | Enabled |
| Device | CPU / GPU auto-detect |

---

## Quick Start

> **Prerequisites:** Google account · `News_Category_Dataset_v3.json` uploaded to Colab session.

**Step 1 — Open in Colab**

Upload `Semantic_News_Clustering.ipynb` to [Google Colab](https://colab.research.google.com). GPU runtime is recommended for SBERT encoding speed.

**Step 2 — Upload dataset**

Run Cell 1 to trigger `google.colab.files.upload()` and upload `News_Category_Dataset_v3.json`.

**Step 3 — Install dependencies**

```bash
pip install sentence-transformers networkx python-louvain umap-learn plotly pandas numpy scikit-learn
```

**Step 4 — Run all cells in order**

Cells execute the full pipeline: preprocessing → classical vectorization → SBERT encoding → graph construction → Louvain clustering → K-Means → UMAP → Plotly visualizations → evaluation metrics.

> **Note:** The notebook includes graceful `try/except` fallback stubs for all optional libraries (`sentence_transformers`, `networkx`, `umap`, `plotly`), allowing the script to be inspected and partially run even in environments where these packages are unavailable.

---

## Dependencies

| Library | Role |
|---|---|
| `sentence-transformers` | SBERT `all-MiniLM-L6-v2` model loading and article encoding |
| `networkx` | Graph construction, node/edge management, spring layout |
| `python-louvain` | Louvain community detection via `community.best_partition()` |
| `umap-learn` | UMAP dimensionality reduction (384D → 2D) |
| `plotly` | Interactive scatter plots and network graph visualization |
| `scikit-learn` | TF-IDF, BOW, K-Means, cosine similarity, silhouette score, ARI, NMI |
| `pandas` | Dataset loading (JSON Lines), DataFrame operations, category factorization |
| `numpy` | Array operations, embedding manipulation |
| `re` | Regex-based text cleaning in `preprocess_text()` |

---

## Project Structure

```
Semantic_News_Clustering.ipynb
│
├── Cell 0   Dataset upload  (google.colab.files.upload)
├── Cell 1   Dependency installation
├── Cell 2   Library imports + fallback stubs + preprocess_text()
│             Step 1: Load dataset · subset 5000 · combine + clean text
│             Step 2: Classical vectorization  (BOW binary/freq · One-Hot · TF-IDF)
├── Cell 3   Full production script  (with FileNotFoundError dummy data fallback)
│             Step 3: SBERT embeddings  (all-MiniLM-L6-v2)
│             Step 4: Cosine similarity graph  (threshold=0.70)
│             Step 5: Louvain community detection
│             Step 6: K-Means clustering  (k = num_clusters_louvain)
│             Step 7: UMAP 2D reduction
│             Step 8: Plotly UMAP scatter  (Louvain + K-Means)
│             Step 9: Silhouette score evaluation
├── Cell 4   Step 7–8 standalone  (UMAP + Plotly scatter, importable independently)
├── Cell 5   Step 10: Interactive network graph  (NetworkX spring layout + Plotly)
├── Cell 6   Step 9 standalone  (Silhouette score with trivial-check guard)
├── Cell 7   Silhouette bar chart  (Plotly go.Bar comparison figure)
├── Cell 8   External evaluation  (ARI + NMI vs. true category labels)
└── Cell 9   BERT sentiment analysis architecture diagram + training configuration
```

---

## Key Design Decisions

**Graph-based clustering over direct K-Means on embeddings.** Transforming the similarity matrix into a graph and applying Louvain community detection allows the algorithm to discover clusters of arbitrary shape. K-Means assumes spherical, equal-variance clusters in high-dimensional space — an assumption that rarely holds for real-world news topics with irregular semantic boundaries.

**Threshold selection at 0.70.** The cosine similarity threshold of 0.70 determines graph density. A lower threshold creates a near-complete graph where every node connects to nearly every other, making community structure trivial. A higher threshold creates excessive sparsity and many isolated nodes. 0.70 represents a semantically meaningful boundary: articles must share substantial topical and lexical semantic content to be connected.

**K-Means cluster count derived from Louvain.** Using `num_clusters = num_clusters_louvain` for K-Means ensures both methods operate on the same number of groups, making silhouette score, ARI, and NMI comparisons directly meaningful rather than confounded by different `k` values.

**UMAP over t-SNE for visualization.** UMAP preserves both local neighbourhood structure and global topology, runs faster than t-SNE on large datasets, and is deterministic with `random_state=42`. `n_neighbors=15` controls the local neighbourhood size used during manifold learning; `min_dist=0.1` controls how tightly points are packed in the 2D projection.

**Graceful fallback stubs.** Every optional library import is wrapped in a `try/except` block that defines a minimal stub class if the import fails. This allows the notebook to be fully inspected, executed partially, and used pedagogically even on machines without GPU support or in restricted environments.

**Ground-truth evaluation without label leakage.** Category labels are never used during preprocessing, embedding, graph construction, or clustering. They enter only at the external evaluation stage, ensuring the pipeline is genuinely unsupervised and that ARI/NMI scores reflect cluster quality rather than supervision.

---

## Limitations and Future Work

**Threshold sensitivity.** The 0.70 cosine similarity threshold significantly influences graph structure and therefore Louvain output. A rigorous grid search or an adaptive thresholding strategy (e.g., using the k-nearest-neighbour graph instead of a fixed threshold) would make the pipeline more robust.

**Subset size.** Processing only 5,000 of 200,000+ available articles limits representativeness. Scaling to the full dataset would require approximate nearest-neighbour search (e.g., FAISS or Annoy) instead of the full pairwise cosine similarity matrix, which grows as O(n²) in memory.

**Single embedding model.** The pipeline uses only `all-MiniLM-L6-v2`. Comparing against larger SBERT variants (`all-mpnet-base-v2`, `paraphrase-multilingual-mpnet-base-v2`) or domain-adapted news encoders would provide stronger baselines.

**Dynamic news streams.** The current system processes a static snapshot. Extending to an online or streaming architecture — incrementally updating the similarity graph as new articles arrive — would enable real-time topic monitoring.

**Topic labeling.** Clusters are currently unlabeled. Applying TF-IDF top-k term extraction or LDA within each cluster to generate human-readable topic labels would make the output directly usable for journalism or content recommendation applications.

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:24243e,50:302b63,100:0f0c29&height=80&text=Built%20with%20Sentence-BERT%20·%20NetworkX%20·%20Louvain%20·%20UMAP%20·%20Plotly&fontSize=13&fontColor=aaaacc&fontAlignY=50"/>

<br/>

*If this project was useful to you, consider giving it a ⭐*

</div>
