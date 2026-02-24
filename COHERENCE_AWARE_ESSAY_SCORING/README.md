# Enhancing Automated Essay Scoring with Coherence-Aware Transformer Modeling

> A reproducible, end-to-end experimental pipeline for Scopus-indexed publication.  
> **Research Question:** Does fusing a semantic coherence feature into BERT improve Automated Essay Scoring (AES) on the ASAP dataset?

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40.0-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/transformers)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-ASAP%20(Kaggle)-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/asap-aes)
[![Seed](https://img.shields.io/badge/Seed-42-8A2BE2?style=flat-square)]()

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Hypothesis & Contribution](#-hypothesis--contribution)
- [Architecture](#-architecture)
- [Pipeline Overview](#-pipeline-overview)
- [Quick Start](#-quick-start)
- [Experimental Configuration](#-experimental-configuration)
- [Results](#-results)
- [Outputs & Artifacts](#-outputs--artifacts)
- [Project Structure](#-project-structure)
- [Reproducibility](#-reproducibility)
- [Dataset](#-dataset)
- [Citation](#-citation)

---

## 🧠 Overview

This project presents a **coherence-aware BERT regression model** for Automated Essay Scoring (AES). The system evaluates essay quality by combining two complementary signals:

1. **Contextual semantics** — captured by the `[CLS]` token embedding from `bert-base-uncased`
2. **Discourse coherence** — computed as the mean cosine similarity between consecutive sentence embeddings, measuring how logically connected an essay's sentences are

The complete pipeline covers data ingestion, two classical baselines, a BERT-only model, the proposed fusion model, ablation study, statistical significance testing via bootstrap resampling, error analysis, and paper-ready outputs (LaTeX tables, result paragraphs, publication figures).

---

## 🎯 Hypothesis & Contribution

| | Statement |
|---|---|
| **H₀** | Coherence fusion provides no statistically significant improvement over BERT-only |
| **H₁** | BERT + coherence fusion significantly improves Quadratic Weighted Kappa (QWK) over the BERT-only baseline |
| **Primary metric** | Quadratic Weighted Kappa (QWK) — the standard metric in AES literature |
| **Secondary metrics** | Pearson r, Spearman ρ, RMSE, MAE |

**Key insight:** Raw token-level representations capture local semantics but miss global discourse structure. A student can write grammatically fluent sentences that are thematically incoherent. This work quantifies how much that structural signal is worth in a scoring context.

---

## 🏗️ Architecture

### BERT-Only Baseline

```
Essay Text
    │
    ▼
bert-base-uncased
    │
    ▼
[CLS] embedding  (768-d)
    │
    ▼
Dropout → Linear(768 → 1)
    │
    ▼
Predicted Score ∈ [0, 1]
```

### Proposed: BERT + Coherence Fusion

```
Essay Text                         Essay Text (sentence-split)
    │                                        │
    ▼                                        ▼
bert-base-uncased              bert-base-uncased (shared encoder)
    │                                        │
    ▼                             per-sentence [CLS] embeddings
[CLS] embedding (768-d)                      │
    │                             mean cosine similarity between
    │                             consecutive sentences → scalar
    │                                        │
    └──────────── concat ────────────────────┘
                     │
              fused vector (769-d)
                     │
              Dropout → Linear(769 → 256) → GELU
                     │
              Dropout → Linear(256 → 1)
                     │
              Predicted Score ∈ [0, 1]
```

The coherence encoder **reuses the weights** of the fine-tuned BERT encoder — no additional pretrained model is loaded, keeping memory overhead near zero.

---

## 🔬 Pipeline Overview

The notebook is organized into 14 sequential cells:

| Cell | Description |
|------|-------------|
| **1** | Environment setup, reproducibility seeding, hyperparameter registry |
| **2** | Dataset download (Kaggle), column normalization, per-prompt score normalization, stratified 70/15/15 split |
| **3** | Metric utilities — `quadratic_weighted_kappa()`, `compute_metrics()`, `prompt_wise_qwk()` |
| **4** | Classical baselines — TF-IDF (unigram+bigram, 10k features) + Linear Regression and SVR |
| **5** | `EssayDataset` class and BERT-compatible `DataLoader` construction |
| **6** | `BertRegressor` architecture, differential learning rates, AdamW + linear warmup scheduler |
| **7** | BERT-only training with early stopping; training curve plots saved |
| **8** | Sentence-level coherence pre-computation; `BertCoherenceRegressor` architecture; `EssayDatasetCoh` |
| **9** | BERT + Coherence fusion training; comparative QWK progression plot |
| **10** | Full ablation table across all 4 models; prompt-wise QWK breakdown |
| **11** | Bootstrap resampling (N=1000) confidence intervals; permutation test p-value |
| **12** | Error analysis — residual distribution, scatter, prompt-wise MAE |
| **13** | Publication-quality 2×2 results panel (bar chart, correlation scatter, confusion heatmap, prompt-wise QWK) |
| **14** | Paper-ready outputs — LaTeX tables, auto-generated experimental setup and results paragraphs, CSV |

---

## 🚀 Quick Start

### Requirements

All dependencies are pre-installed in Google Colab. For local runs:

```bash
pip install transformers==4.40.0 scikit-learn scipy numpy pandas \
            matplotlib seaborn tqdm torch kagglehub
```

### Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `AES_Coherence_Transformer.ipynb`
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells — upload the ASAP dataset when prompted in Cell 2

> **Tip:** Replace the manual upload in Cell 2 with the `kagglehub` snippet for fully automated ingestion:

```python
import kagglehub
path = kagglehub.dataset_download("kaggle/asap-aes")
```

### Runtime Estimates (T4 GPU)

| Stage | Estimated Time |
|-------|----------------|
| Data preprocessing | < 1 min |
| Baseline training (LR + SVR) | 2–5 min |
| BERT-only training (8 epochs) | 20–35 min |
| Coherence pre-computation | 15–25 min |
| BERT + Coherence training (8 epochs) | 25–40 min |
| Bootstrap significance testing | 3–5 min |
| **Total** | **~1.5–2 hours** |

---

## ⚙️ Experimental Configuration

All hyperparameters are centralized in the `HP` dictionary in Cell 1 — change once, propagates everywhere.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `bert_model` | `bert-base-uncased` | Standard AES benchmark choice |
| `max_seq_len` | 512 | BERT maximum context window |
| `batch_size` | 8 | Fits comfortably on 16GB T4/V100 |
| `bert_lr` | 2e-5 | Standard BERT fine-tuning rate |
| `head_lr` | 1e-4 | Higher LR for randomly initialized projection head |
| `weight_decay` | 0.01 | AdamW regularization |
| `epochs` | 8 | Upper bound; early stopping (patience=3) applies |
| `warmup_ratio` | 0.1 | 10% of total steps for LR warmup |
| `grad_clip` | 1.0 | Prevents gradient explosion |
| `train / val / test` | 70 / 15 / 15% | Stratified by essay prompt |
| `tfidf_max_features` | 10,000 | Unigram + bigram features |
| `bootstrap_iters` | 1,000 | CI and permutation test iterations |
| `seed` | 42 | Full reproducibility |

---

## 📊 Results

> *Actual values depend on your hardware and dataset. Run all cells to populate. The table below is auto-saved to `aes_results.csv` and `table_results.tex`.*

### Overall Performance (Test Set)

| Model | QWK ↑ | Pearson ↑ | Spearman ↑ | RMSE ↓ | MAE ↓ |
|---|---|---|---|---|---|
| TF-IDF + Linear Regression | — | — | — | — | — |
| TF-IDF + SVR | — | — | — | — | — |
| BERT-only | — | — | — | — | — |
| **BERT + Coherence (proposed)** ⭐ | **—** | **—** | **—** | **—** | **—** |

### Statistical Significance

The improvement of BERT + Coherence over BERT-only is validated via:

- **95% Bootstrap Confidence Intervals** (N=1,000) on QWK for each model
- **One-tailed permutation test** — H₀: BERT+Coh QWK ≤ BERT-only QWK

Results are printed to console and visualized in `bootstrap_ci.png`.

---

## 📁 Outputs & Artifacts

After a full notebook run, the following files are saved to the working directory:

| File | Description |
|------|-------------|
| `aes_results.csv` | All model metrics in tabular CSV format |
| `table_results.tex` | LaTeX-ready results table (drop directly into paper) |
| `table_hyperparams.tex` | LaTeX hyperparameter table |
| `setup_paragraph.txt` | Auto-generated experimental setup paragraph |
| `results_paragraph.txt` | Auto-generated results discussion paragraph |
| `bert_only_training_curves.png` | Loss + QWK curves for BERT-only |
| `bert_coh_training_curves.png` | Loss + comparative QWK for BERT+Coherence |
| `bootstrap_ci.png` | Bootstrap QWK distributions with 95% CI bands |
| `error_analysis.png` | Scatter plot, residual histogram, prompt-wise MAE |
| `results_panel.png` | 2×2 publication-quality summary figure (150 DPI) |

All figures are saved at ≥130 DPI and are submission-ready for IEEE / Springer / Elsevier templates.

---

## 📂 Project Structure

```
AES_Coherence_Transformer/
│
├── AES_Coherence_Transformer.ipynb   # Main notebook (14 cells)
├── README.md                          # This file
│
└── outputs/  (generated at runtime)
    ├── aes_results.csv
    ├── table_results.tex
    ├── table_hyperparams.tex
    ├── setup_paragraph.txt
    ├── results_paragraph.txt
    ├── bert_only_training_curves.png
    ├── bert_coh_training_curves.png
    ├── bootstrap_ci.png
    ├── error_analysis.png
    └── results_panel.png
```

---

## 🔁 Reproducibility

This pipeline is designed for **exact reproducibility** across runs:

- Global seed (`42`) applied to Python `random`, NumPy, PyTorch CPU/CUDA, and `PYTHONHASHSEED`
- `torch.backends.cudnn.deterministic = True` and `benchmark = False`
- Stratified splits preserve essay prompt distribution across train/val/test
- Bootstrap and permutation tests use `np.random.default_rng(SEED)` for deterministic sampling
- All hyperparameters live in one `HP` dict — no magic numbers scattered through the code

> **Note:** CUDA operations on some GPU models may introduce minor non-determinism even with fixed seeds. Results are reproducible within a consistent hardware environment.

---

## 🗃️ Dataset

**ASAP (Automated Student Assessment Prize)** — 8 essay prompts, ~12,976 student essays scored by trained human raters.

| Prompt | Score Range | Topic |
|--------|-------------|-------|
| 1 | 2–12 | Computers / libraries |
| 2 | 1–6 | Censorship |
| 3 | 0–3 | Laughter |
| 4 | 0–3 | Patience |
| 5 | 0–4 | Rough road |
| 6 | 0–4 | Gentleness |
| 7 | 0–30 | Evidence |
| 8 | 0–60 | Banning cell phones |

Scores are **normalized per-prompt** to [0, 1] using the known min/max ranges before training. Access the dataset at the [Kaggle ASAP AES Competition](https://www.kaggle.com/c/asap-aes) (free account required).

---

## 📖 Citation

If you use this code or pipeline in your research, please cite:

```bibtex
@inproceedings{yourname2025aes,
  title     = {Enhancing Automated Essay Scoring with Coherence-Aware Transformer Modeling},
  author    = {Your Name and Co-Author Name},
  booktitle = {Proceedings of [Conference Name]},
  year      = {2025},
  publisher = {[Publisher]},
  note      = {Scopus-indexed}
}
```

---

## 📄 License

This project is released for academic and research use. Please cite appropriately if building upon this work.

---

<p align="center">
  Built for reproducible NLP research · BERT-base-uncased · ASAP Dataset · PyTorch · Google Colab
</p>
