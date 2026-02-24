# 🎮 Steam Game Recommender System

> **Content-based filtering** powered by TF-IDF vectorization and cosine similarity — discover games you'll actually love, not just games that are popular.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Example Output](#-example-output)
- [API Reference](#-api-reference)
- [Limitations & Future Work](#-limitations--future-work)
- [Dataset](#-dataset)
- [License](#-license)

---

## 🧠 Overview

This project builds a **content-based game recommendation engine** trained on the Steam games library. Given any game title, it returns the top N most similar games — ranked by how closely their descriptions match.

Unlike collaborative filtering (which needs user interaction data), this system works purely from **game descriptions**, making it useful even for newly released titles with no review history.

| | Details |
|---|---|
| **Approach** | Content-Based Filtering |
| **Similarity Metric** | Cosine Similarity |
| **Text Representation** | TF-IDF (Term Frequency–Inverse Document Frequency) |
| **Dataset** | Steam Games Dataset (Kaggle) |
| **Runtime** | Google Colab (CPU, no GPU needed) |

---

## ⚙️ How It Works

```
Game Descriptions (text)
        │
        ▼
  TF-IDF Vectorizer          ← Converts text into weighted numerical vectors
        │                       Stop words removed, rare terms boosted
        ▼
  TF-IDF Matrix (sparse)     ← Shape: [n_games × n_vocab_terms]
        │
        ▼
  Cosine Similarity Matrix   ← Shape: [n_games × n_games]
        │                       Each cell = similarity score between two games (0–1)
        ▼
  Top-N Lookup               ← Sort scores for a given game, return top N matches
        │
        ▼
  Recommendations ✅
```

**Why TF-IDF?**
Raw word counts favor common words like "the" or "and." TF-IDF penalizes words that appear in every game description and rewards words that are distinctive to a specific game — like "post-apocalyptic," "roguelite," or "turn-based."

**Why Cosine Similarity?**
It measures the *angle* between two vectors, not their magnitude. This means a short description and a long description of the same genre will still be considered similar — the length of the text doesn't skew the result.

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/) and upload `steam_game_recommender.py`
2. Run all cells top to bottom — the dataset downloads automatically via `kagglehub`

> **First run only:** `kagglehub` will prompt you to authenticate with your Kaggle account. After that, the dataset is cached and subsequent runs are instant.

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/steam-game-recommender.git
cd steam-game-recommender

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas scikit-learn kagglehub

# 4. Run the script
python steam_game_recommender.py
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF vectorization & cosine similarity |
| `kagglehub` | Automatic dataset download from Kaggle |

---

## 📁 Project Structure

```
steam-game-recommender/
│
├── steam_game_recommender.py   # Main script (all 8 steps)
└── README.md                   # You are here
```

The script is organized into 8 self-contained steps — each can be run as its own Colab cell:

| Step | Description |
|------|-------------|
| 1 | Install & import libraries |
| 2 | Download dataset via `kagglehub` |
| 3 | Preprocess & clean data |
| 4 | Build TF-IDF matrix |
| 5 | Compute cosine similarity matrix |
| 6 | Define the `steam_game_recommender()` function |
| 7 | Run example recommendations |
| 8 | Interactive query (optional) |

---

## 📊 Example Output

```
=======================================================
  🎮 Top 10 games similar to: DayZ
=======================================================
                          name         genre  original_price  similarity_score
0              The Walking Dead         Action           $4.99            0.4821
1                  Dying Light         Action          $29.99            0.4703
2     Project Zomboid          Indie, RPG      $19.99            0.4612
3                    SCUM       Action, Indie    $29.99            0.4508
...

=======================================================
  🎮 Top 10 games similar to: DOOM
=======================================================
                          name         genre  original_price  similarity_score
0                  DOOM Eternal        Action          $39.99            0.7103
1                  Quake Champions       Action           $0.00            0.5892
2             Wolfenstein II    Action          $39.99            0.5601
...
```

---

## 📖 API Reference

### `steam_game_recommender(title, top_n=10)`

Returns a ranked DataFrame of the most similar games to the one provided.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | required | Exact game name as it appears in the Steam dataset |
| `top_n` | `int` | `10` | Number of recommendations to return |

**Returns**

A `pd.DataFrame` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Game title |
| `genre` | str | Steam genre tags |
| `original_price` | str | Price on Steam |
| `similarity_score` | float | Cosine similarity to input game (0–1) |

**Raises**

`ValueError` — if the game title is not found in the dataset. A "did you mean?" suggestion is included when partial matches exist.

**Example**

```python
recs = steam_game_recommender("DOOM", top_n=5)
print(recs)
```

---

## ⚠️ Limitations & Future Work

**Current limitations:**

- Similarity is based **only on text descriptions** — gameplay mechanics, graphics style, and player ratings are not considered
- Requires an **exact title match** (case-sensitive) — fuzzy search support is partial
- The TF-IDF matrix is recomputed **in memory on every run** — not persisted to disk

**Ideas for improvement:**

- Add **tag-based and genre-based** similarity alongside description similarity (multi-feature fusion)
- Integrate **user review sentiment** as an additional signal
- Build a **Streamlit or Gradio web UI** for non-technical users
- Persist the similarity matrix using `joblib` to avoid recomputation
- Add **fuzzy title matching** with `rapidfuzz` for more forgiving lookups
- Experiment with **sentence transformers** (e.g., `all-MiniLM-L6-v2`) for richer semantic embeddings

---

## 📦 Dataset

**Steam Games Dataset** by [fronkongames](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) on Kaggle.

Contains metadata for tens of thousands of Steam titles including descriptions, genres, tags, prices, and more. Downloaded automatically at runtime via `kagglehub` — no manual download required.

> A free Kaggle account is required. Sign up at [kaggle.com](https://www.kaggle.com).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

<p align="center">Built with 🎮 and Python · Runs entirely in Google Colab · No GPU required</p>
