<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=800&size=28&pause=1000&color=6366F1&center=true&vCenter=true&width=700&lines=Research+Novelty+Evaluation+System;AI-Powered+Academic+Innovation+Advisor;Scholar+Search+%C2%B7+Semantic+NLP+%C2%B7+Innovation+Scoring" alt="Typing SVG" />

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Scholar%20Search-Automated-6366F1?style=flat-square"/>
  <img src="https://img.shields.io/badge/NLP%20Similarity-Semantic-8B5CF6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Summarization-DistilBART-38BDF8?style=flat-square"/>
  <img src="https://img.shields.io/badge/Innovation%20Score-0--100-4ADE80?style=flat-square"/>
  <img src="https://img.shields.io/badge/Gap%20Analysis-Domain%20Aware-FB923C?style=flat-square"/>
</p>

<br/>

> **A production-grade AI system that evaluates the novelty of any research idea in minutes.**
> Acts as your personal research supervisor — searching literature, scoring originality, and prescribing exactly how to make your idea publishable.

<br/>

---

</div>

## 📌 Table of Contents

- [✨ Overview](#-overview)
- [🎬 Demo](#-demo)
- [🔬 How It Works](#-how-it-works)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Features](#️-features)
- [📊 Output Structure](#-output-structure)
- [🚀 Quick Start — Google Colab](#-quick-start--google-colab)
- [🖥️ Running Locally](#️-running-locally)
- [📦 Dependencies](#-dependencies)
- [📐 System Design](#-system-design)
- [🔧 Configuration & Customization](#-configuration--customization)
- [🧪 Example Evaluation](#-example-evaluation)
- [🛣️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## ✨ Overview

**Research Novelty Evaluation System (RNES)** is an end-to-end AI pipeline designed for researchers, students, and academics who want an instant, data-driven assessment of whether their research idea is truly novel — before committing months of work.

Most researchers rely on manual literature reviews that are slow, biased, and incomplete. RNES automates the entire process:

```
Your Idea  →  Scholar Search  →  NLP Analysis  →  Novelty Score  →  Actionable Report
```

Think of it as a **GPT-powered research supervisor** available 24/7, with no office hours.

### Why RNES?

| Traditional Literature Review | RNES |
|---|---|
| 2–4 weeks of reading | ⚡ Under 3 minutes |
| Manual keyword searches | 🤖 Auto-generated search queries |
| Subjective novelty assessment | 📊 Quantified semantic similarity score |
| No suggestions provided | 💡 Concrete innovation pivots & gap analysis |
| Requires deep domain expertise | 🌐 Works on any CS/AI research domain |

---

## 🎬 Demo

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│  🔬  RESEARCH NOVELTY EVALUATION SYSTEM                         │
│  AI-POWERED ACADEMIC INNOVATION ADVISOR                         │
│                                                                 │
│  Title:  Federated Learning for Medical Image Segmentation      │
│  Desc:   Using FL with differential privacy to segment CT...    │
│                                                                 │
│  [  🚀  EVALUATE NOVELTY  ]        [ Fast Mode ☐ ]             │
└─────────────────────────────────────────────────────────────────┘

  ✅ Decomposing idea into components...
  ✅ Searching Google Scholar...
  ✅ Summarizing 9 related papers...
  ✅ Computing semantic similarity...
  ✅ Assessing novelty & generating report...

  ┌──────────────────────────────────────────────────────────┐
  │  NOVELTY SCORE           Innovation Level: HIGH          │
  │  ●●●●●●●●●○  78/100      Avg Similarity: 28%            │
  │                          Papers Analyzed: 9              │
  └──────────────────────────────────────────────────────────┘
```

</div>

**🔗 [Open in Google Colab](https://colab.research.google.com)** — upload the `.ipynb` and run all cells.

---

## 🔬 How It Works

The system executes a **7-stage AI pipeline** the moment you click *Evaluate*:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RNES EVALUATION PIPELINE                            │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  Stage 1 │  Stage 2 │  Stage 3 │  Stage 4 │  Stage 5 │  Stage 6 │  Stage 7 │
│  Idea    │  Scholar │  Trans-  │  Semantic│  Novelty │  Gap     │  Final   │
│  Decom-  │  Search  │  former  │  Similar-│  Assess- │  & Sug-  │  Verdict │
│  position│          │  Summar. │  ity     │  ment    │  gestions│          │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### Stage 1 — Idea Decomposition
Parses your title and description using regex pattern matching against 15+ tech domains (NLP, Computer Vision, Federated Learning, Generative AI, etc.), extracts key noun phrases, and auto-generates 3–4 targeted search queries.

### Stage 2 — Google Scholar Search
Queries Google Scholar via the `scholarly` library (no API key needed), with a BeautifulSoup scraper as fallback. Retrieves paper titles, authors, publication year, citation counts, abstract, and direct URLs. Deduplicates results across all queries.

### Stage 3 — Transformer Summarization
Passes each abstract through `sshleifer/distilbart-cnn-12-6` (a distilled BART model) to generate a concise, readable 2–3 sentence summary per paper. Skippable via **Fast Mode** for quick previews.

### Stage 4 — Semantic Similarity
Encodes your idea and all paper abstracts using `all-MiniLM-L6-v2` (SentenceTransformers). Computes cosine similarity scores — giving you a quantified measure of how closely existing work overlaps with your idea.

### Stage 5 — Novelty Assessment
Combines average and maximum similarity into a weighted novelty formula:

```
raw_novelty  =  1  −  (avg_similarity × 0.6  +  max_similarity × 0.4)
novelty_score = clamp(raw_novelty × 100,  min=10,  max=98)
```

Produces a **0–100 innovation score** and a **Low / Medium / High** label.

### Stage 6 — Gap Identification & Suggestions
Cross-references your idea's keywords against what's covered in the literature, surfaces unexplored dimensions as research gaps, and generates domain-specific suggestions for advanced techniques, datasets, and multimodal integrations.

### Stage 7 — Final Verdict
Synthesizes all signals into a single expert verdict — written the way a research supervisor would phrase it — with tailored advice on publication venue suitability.

---

## 🏗️ Architecture

```
research-novelty-system/
│
├── 📓 Research_Novelty_Evaluation_System.ipynb   # Main Colab notebook
│
│   ├── Cell 1 ── Package Installation
│   │             (flask, transformers, scholarly, sentence-transformers, ...)
│   │
│   ├── Cell 2 ── Core Backend Pipeline
│   │             ├── decompose_idea()          # Stage 1: NLP decomposition
│   │             ├── search_scholar()          # Stage 2: Scholar API
│   │             ├── search_scholar_scrape()   # Stage 2: Fallback scraper
│   │             ├── summarize_abstract()      # Stage 3: DistilBART
│   │             ├── compute_similarity()      # Stage 4: MiniLM embeddings
│   │             ├── assess_novelty()          # Stage 5: Scoring formula
│   │             ├── identify_gaps()           # Stage 6: Gap analysis
│   │             └── evaluate_research_idea()  # Master pipeline
│   │
│   ├── Cell 3 ── Flask REST API  (port 8766)
│   │             ├── POST /evaluate
│   │             └── GET  /health
│   │
│   ├── Cell 4 ── HTML/CSS/JS Frontend Generation
│   │             (dark theme, animated UI, score ring, paper cards)
│   │
│   └── Cell 5 ── Colab HTTPS Proxy Launch
│                 (auto-patches API URL, displays launch button)
│
└── 📄 README.md
```

**Technology Stack:**

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML5 / CSS3 / JavaScript (ES6+) |
| API Server | Flask 2.x + Flask-CORS |
| NLP Summarization | HuggingFace `sshleifer/distilbart-cnn-12-6` |
| Semantic Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Scholar Search | `scholarly` + BeautifulSoup4 (fallback) |
| Similarity Metric | Cosine Similarity (scikit-learn) |
| Hosting | Google Colab HTTPS Proxy |

---

## ⚙️ Features

### 🎨 Frontend
- **Dark professional UI** — inspired by modern research tooling
- **Animated progress tracker** — 5-step loader with live status updates
- **Circular novelty gauge** — animated SVG ring with color-coded scoring
- **Semantic similarity bars** — per-paper color-coded overlap visualization
- **Clickable paper titles** — direct links to source URLs
- **Fast Mode toggle** — skip transformer summarization for instant previews
- **Fully responsive** — works on desktop, tablet, and mobile

### 🤖 AI/ML
- **15+ domain detectors** — NLP, CV, RL, Generative AI, Healthcare, Robotics, and more
- **Multi-query Scholar search** — up to 4 auto-generated queries per idea
- **Deduplication** — intelligent cross-query paper deduplication
- **Transformer summarization** — academic-quality abstract compression
- **Semantic similarity** — state-of-the-art sentence embedding comparison
- **Domain-specific recommendations** — curated techniques per AI subdomain
- **Dataset suggestions** — 30+ curated academic datasets mapped to domains

### 🏗️ Engineering
- **Modular pipeline** — each stage is an independent, testable function
- **Lazy model loading** — transformers loaded on first use to minimize cold start
- **Dual Scholar fallback** — `scholarly` library + direct scraper
- **Error boundaries** — graceful degradation on Scholar rate-limiting
- **Thread-safe Flask server** — runs in background daemon thread
- **Auto URL patching** — Colab proxy URL injected at runtime

---

## 📊 Output Structure

Every evaluation produces a **6-section structured report**:

```
📋 SECTION 1 — IDEA BREAKDOWN
    ├── Detected Domains        (e.g., Federated Learning, Healthcare AI)
    ├── Key Concepts            (extracted noun phrases)
    └── Search Queries          (auto-generated for Scholar)

📚 SECTION 2 — EXISTING SIMILAR WORK
    └── Per paper:
        ├── Title + Clickable URL
        ├── Authors · Year · Citation Count
        ├── AI-Generated Summary (DistilBART)
        └── Semantic Similarity Bar (color-coded %)

⚖️  SECTION 3 — COMPARATIVE ANALYSIS
    └── Narrative comparison of your idea vs. the literature field

🎯 SECTION 4 — NOVELTY SCORE
    ├── Innovation Score        (0–100 with animated ring)
    ├── Novelty Level           (Low / Medium / High)
    ├── Average Similarity      (across all papers)
    ├── Maximum Similarity      (closest existing work)
    └── Papers Analyzed         (total corpus size)

💡 SECTION 5 — INNOVATION SUGGESTIONS
    ├── Research Gaps           (unexplored dimensions)
    ├── Improvement Pivots      (concrete differentiators)
    ├── Advanced Techniques     (domain-specific recommendations)
    └── Recommended Datasets    (curated for your domain)

📝 SECTION 6 — FINAL VERDICT
    └── Expert narrative verdict with publication guidance
```

---

## 🚀 Quick Start — Google Colab

The fastest way to run RNES. No local setup required.

### Step 1 — Open the Notebook

Upload `Research_Novelty_Evaluation_System.ipynb` to [Google Colab](https://colab.google.com):

```
File → Upload notebook → Select the .ipynb file
```

Or open directly from GitHub:

```
File → Open notebook → GitHub tab → paste repo URL
```

### Step 2 — Run All Cells in Order

```
Runtime → Run all   (or press Ctrl+F9)
```

> ⏱️ First run takes ~3–5 minutes as models are downloaded. Subsequent runs are fast.

### Step 3 — Click the Launch Button

Cell 5 generates a styled launch panel (identical to the example in the screenshot). Click **🚀 OPEN FULL SCREEN** — the app opens in a new browser tab via Colab's HTTPS proxy, with camera and clipboard access fully functional.

```
✅ Server running on port 8767
🔗 App URL: https://8767-m-s-XXXXX.us-east1-1.prod.colab.dev/research_novelty_ui.html
⚠️  Keep this cell running — the server stops when the kernel restarts
```

### Step 4 — Evaluate Your Idea

1. Enter your **Research Title**
2. Paste your **Description / Abstract** (3–5 sentences minimum for best results)
3. Toggle **Fast Mode** if you want results in under 30 seconds (skips summarization)
4. Click **🚀 EVALUATE NOVELTY**
5. View your full structured report

---

## 🖥️ Running Locally

For development or private use without Colab:

### Prerequisites

```bash
Python 3.8+
pip 21+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/research-novelty-system.git
cd research-novelty-system

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Start the Flask API (in one terminal)
python app.py

# Serve the frontend (in another terminal)
cd frontend
python -m http.server 8767

# Open in browser
open http://localhost:8767/research_novelty_ui.html
```

> The API will be available at `http://localhost:8766`

---

## 📦 Dependencies

```txt
flask>=2.3.0
flask-cors>=4.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
scholarly>=1.7.11
transformers>=4.35.0
torch>=2.0.0
sentence-transformers>=2.2.2
scikit-learn>=1.3.0
numpy>=1.24.0
```

Install all at once:

```bash
pip install flask flask-cors requests beautifulsoup4 lxml scholarly \
            transformers torch sentence-transformers scikit-learn numpy
```

**Model Downloads (first run only):**

| Model | Size | Purpose |
|---|---|---|
| `sshleifer/distilbart-cnn-12-6` | ~1.2 GB | Abstract summarization |
| `all-MiniLM-L6-v2` | ~90 MB | Semantic embeddings |

---

## 📐 System Design

### API Endpoints

```http
POST /evaluate
Content-Type: application/json

{
  "title":       "Your Research Title",
  "description": "Your research description...",
  "fast_mode":   false
}
```

**Response Schema:**

```json
{
  "status": "ok",
  "title": "...",
  "decomposition": {
    "domains": ["NLP", "Healthcare AI"],
    "keywords": ["segmentation", "privacy", ...],
    "search_queries": ["...", "..."]
  },
  "papers": [
    {
      "title": "...", "authors": "...", "year": "2023",
      "abstract": "...", "summary": "...",
      "citations": 142, "url": "...",
      "similarity": 0.34
    }
  ],
  "novelty": {
    "level": "High", "score": 78,
    "avg_similarity": 0.28, "max_similarity": 0.41,
    "confidence": "High"
  },
  "gaps_info": {
    "gaps": ["..."], "suggestions": ["..."],
    "advanced_techniques": ["..."], "datasets": ["..."]
  },
  "verdict": "🟢 HIGH NOVELTY (78/100): ..."
}
```

```http
GET /health
→ { "status": "ok", "message": "Research Novelty Evaluation System is running!" }
```

### Novelty Scoring Formula

```
novelty_raw   =  1  −  ( avg_similarity × 0.6  +  max_similarity × 0.4 )
novelty_score =  clamp( round(novelty_raw × 100),  min=10,  max=98 )

Level:
  score ≥ 70  →  HIGH    (publication-ready territory)
  score ≥ 45  →  MEDIUM  (promising, needs differentiation)
  score  < 45 →  LOW     (significant overlap detected)
```

### Scholar Search Strategy

```python
# Multi-query approach for maximum coverage
queries = [
    original_title,                          # Exact title search
    f"{domain_1} {top_3_keywords}",          # Domain-targeted search
    f"{domain_2} {top_3_keywords}",          # Second domain
    " ".join(top_5_keywords)                 # Keyword-only search
]

# Deduplication by exact title match
# Limit: 12 unique papers across all queries
```

---

## 🔧 Configuration & Customization

### Change Summarization Model

In Cell 2, replace the model name:

```python
# Faster (smaller):
model='sshleifer/distilbart-cnn-6-6'

# Higher quality (larger):
model='facebook/bart-large-cnn'

# Domain-specific (biomedical):
model='allenai/led-base-16384'
```

### Add New Domain Detectors

```python
tech_keywords['Quantum Computing'] = r'quantum|qubit|superposition|qml|variational'
tech_keywords['Graph Neural Networks'] = r'gnn|gcn|graph attention|node embedding'
```

### Adjust Novelty Thresholds

```python
# In assess_novelty():
if score >= 70:   level = 'High'    # Change 70 to adjust High threshold
elif score >= 45: level = 'Medium'  # Change 45 to adjust Medium threshold
else:             level = 'Low'
```

### Add SerpAPI for More Reliable Search

Replace `search_scholar()` with:

```python
from serpapi import GoogleSearch

def search_via_serpapi(query, api_key, num=5):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": num
    }
    results = GoogleSearch(params).get_dict()
    return results.get("organic_results", [])
```

---

## 🧪 Example Evaluation

**Input:**
```
Title:       Multimodal Emotion Recognition Using Federated Learning on Edge Devices

Description: We propose a privacy-preserving multimodal emotion recognition system
             that fuses facial expressions, voice tone, and physiological signals
             (EEG, GSR) using a cross-modal transformer trained via federated
             learning on distributed edge devices. The system targets real-time
             mental health monitoring in clinical settings without centralizing
             sensitive patient data.
```

**Output (abridged):**
```
🧩 IDEA BREAKDOWN
  Domains:  Federated Learning · Multimodal AI · Healthcare AI · Edge Computing
  Keywords: emotion, recognition, federated, edge, physiological, clinical, privacy
  Queries:  ["Multimodal Emotion Recognition...", "Federated Learning emotion edge", ...]

📚 EXISTING SIMILAR WORK  (8 papers found)
  ├── "Multimodal Sentiment Analysis: A Survey" — 2022, cited 891
  │    Similarity: 41%  ▓▓▓▓░░░░░░
  ├── "Federated Learning for Healthcare" — 2023, cited 312
  │    Similarity: 29%  ▓▓░░░░░░░░
  └── ... (6 more)

🎯 NOVELTY SCORE
  Innovation Score: 74 / 100
  Novelty Level:    HIGH  🟢
  Avg Similarity:   26%
  Papers Analyzed:  8

💡 INNOVATION SUGGESTIONS
  Gaps:     Physiological signal fusion · EEG+GSR integration · Edge deployment evaluation
  Advanced: Use CLIP-style cross-modal alignment for EEG-video fusion
            Apply FLWR framework for production federated orchestration
  Datasets: DEAP · SEED-IV · MAHNOB-HCI · AMIGOS

📝 FINAL VERDICT
  🟢 HIGH NOVELTY (74/100): Your idea presents a strong research opportunity.
  The specific combination of EEG/GSR physiological signals with federated
  edge deployment is underexplored in the surveyed literature. Targeting
  IEEE JBHI or EMBC 2025 is recommended...
```

---

## 🛣️ Roadmap

- [x] Core 7-stage evaluation pipeline
- [x] Google Scholar integration with fallback scraper
- [x] DistilBART transformer summarization
- [x] MiniLM semantic similarity scoring
- [x] Domain-specific gap analysis & dataset suggestions
- [x] Dark-mode interactive web UI
- [x] Colab HTTPS proxy deployment
- [ ] **SerpAPI integration** for higher reliability Scholar search
- [ ] **PDF upload** — evaluate ideas from existing draft papers
- [ ] **Citation network visualization** — D3.js graph of related work
- [ ] **Multi-user history** — save and compare past evaluations
- [ ] **Export to PDF/LaTeX** — generate a formatted literature review document
- [ ] **Streamlit version** — alternative frontend for easier local deployment
- [ ] **GPT-4 / Claude API mode** — optional LLM-powered verdict enhancement
- [ ] **ArXiv + Semantic Scholar integration** — extend beyond Google Scholar
- [ ] **Fine-tuned domain classifiers** — replace regex with BERT-based detection

---

## 🤝 Contributing

Contributions are warmly welcome! Here's how to get started:

```bash
# Fork and clone
git clone https://github.com/yourusername/research-novelty-system.git
cd research-novelty-system

# Create a feature branch
git checkout -b feature/arxiv-integration

# Make your changes, then commit
git commit -m "feat: add ArXiv API as secondary search source"

# Push and open a Pull Request
git push origin feature/arxiv-integration
```

### Contribution Areas

- 🔍 **New search sources** — ArXiv, Semantic Scholar, PubMed, IEEE Xplore
- 🧠 **Better models** — domain-specialized summarizers or embedders
- 🌍 **Internationalization** — UI translations, multilingual search
- 🐛 **Bug reports** — open an issue with reproduction steps
- 📖 **Documentation** — tutorials, example notebooks, video walkthroughs

---

## 📜 License

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

<div align="center">

**Built with 🧠 by researchers, for researchers.**

*If this tool helped you evaluate or improve your research idea, please consider giving it a ⭐ — it helps others find the project.*

<br/>

[![Star History](https://img.shields.io/github/stars/yourusername/research-novelty-system?style=social)](https://github.com/yourusername/research-novelty-system)
[![Follow](https://img.shields.io/github/followers/yourusername?style=social)](https://github.com/yourusername)

<br/>

```
"The best research ideas are not just creative —
 they are precisely placed at the frontier of what is known."
```

</div>
