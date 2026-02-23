<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6a0dad,50:a855f7,100:06b6d4&height=200&section=header&text=NLP%20Prompt%20Strategy%20Comparator&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Zero-Shot%20%C2%B7%20One-Shot%20%C2%B7%20Few-Shot%20%C2%B7%20Chain-of-Thought%20%E2%80%94%20IMDB%20Sentiment%20Analysis&descAlignY=60&descSize=15"/>

<br/>

<img src="https://img.shields.io/badge/Model-FLAN--T5--Base-6a0dad?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-IMDB%2050K-a855f7?style=for-the-badge&logo=imdb&logoColor=white"/>
<img src="https://img.shields.io/badge/Task-Sentiment%20Classification-06b6d4?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-ffffff?style=for-the-badge"/>

</div>

---

## Overview

This project is a systematic comparison of four prompting strategies applied to binary sentiment classification on the IMDB Movie Reviews dataset. Using **Google's FLAN-T5-Base** (250M parameters) as the backbone language model, the notebook evaluates how prompt design alone — without any fine-tuning — affects classification accuracy on a balanced test set of 20 reviews.

The four strategies evaluated are Zero-Shot, One-Shot, Few-Shot, and Chain-of-Thought (CoT). Each strategy is implemented as a structured prompt template fed to the same frozen model, with results collected, compared in a summary table, and visualized across four charts. The final cell allows the user to input any custom movie review and receive predictions from all four strategies simultaneously, with a majority-vote confidence score.

This project was developed as a final project for an NLP course, exploring how prompt engineering can substitute for expensive model fine-tuning in classification tasks.

---

## Research Question

> **To what extent does prompt strategy design affect the classification accuracy of a frozen instruction-tuned language model on a binary sentiment task — and what are the trade-offs between accuracy, speed, and explainability across prompting approaches?**

---

## System Architecture

The notebook is structured as a linear pipeline across 12 cells:

1. **Installation:** `transformers`, `torch`, `sentencepiece`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` are installed via pip.
2. **Data Loading:** The user uploads an IMDB CSV file via `google.colab.files.upload()`. Label distribution and average review length are reported.
3. **Model Loading:** `google/flan-t5-base` is loaded using `T5ForConditionalGeneration` and `T5Tokenizer`. The model is moved to CUDA if a GPU is available, otherwise falls back to CPU.
4. **Helper Functions:** Four core utilities — `clean_review()`, `generate_response()`, `extract_label()`, and `evaluate()` — are defined once and reused across all strategy cells.
5. **Test Set Sampling:** 10 positive and 10 negative reviews are sampled with `random_state=99`, forming a reproducible, balanced 20-review test set.
6. **Strategy Execution:** Cells 6–9 implement and evaluate each of the four prompt strategies in sequence.
7. **Comparison & Visualization:** Results are aggregated into a summary table and rendered across four matplotlib charts on a dark theme.
8. **Safety & Bias Reflection:** Five identified failure modes are documented with observed cause and recommended fix.
9. **Interactive Test:** The final cell accepts a user-typed review and runs all four strategies, displaying a majority-vote verdict.

```
IMDB CSV Upload
      │
      ▼
clean_review()              ← strips HTML tags, truncates to 400 chars
      │
      ▼
Prompt Template             ← strategy-specific wrapper
      │
      ▼
FLAN-T5-Base (frozen)       ← num_beams=4, max_new_tokens=10–50
      │
      ▼
extract_label()             ← parses "positive" / "negative" from output
      │
      ├──► Zero-Shot results
      ├──► One-Shot results
      ├──► Few-Shot results
      └──► Chain-of-Thought results
                │
                ▼
      Comparison Table + 4 Matplotlib Charts
                │
                ▼
      Interactive Custom Review Test (majority-vote verdict)
```

---

## Model

| Property | Value |
|---|---|
| Model ID | `google/flan-t5-base` |
| Architecture | T5 — Text-to-Text Transfer Transformer |
| Parameters | ~250 Million |
| Training | Instruction fine-tuned on 1,800+ NLP task variants (FLAN collection) |
| Inference mode | Frozen — no fine-tuning applied in this project |
| Decoding | Beam search · `num_beams=4` · `early_stopping=True` |
| Max input tokens | 512 |
| Max output tokens | 10 (label-only strategies) · 50 (CoT reasoning) |
| Device | CUDA T4 GPU on Colab / CPU fallback |

---

## Dataset

| Property | Value |
|---|---|
| Dataset | IMDB Movie Reviews |
| Total size | 50,000 reviews |
| Labels | Binary — `positive` / `negative` |
| Test set | 20 reviews — 10 positive + 10 negative (balanced) |
| Sampling seed | `random_state=99` (fully reproducible) |
| Preprocessing | HTML tag removal via regex · Truncation to 400 characters |

The dataset is provided as a user-uploaded CSV with columns `review` and `sentiment`. No external API or automatic download is used — the file is loaded via Colab's file upload widget.

---

## Prompt Strategies

### Strategy 1 — Zero-Shot

No examples are provided. The model receives only a task instruction and the review text.

```
Classify the sentiment of this movie review as positive or negative.
Review: "<review_text>"
Sentiment:
```

**Characteristics:** Fastest execution (~10 output tokens per query). Relies entirely on the model's pre-trained instruction-following capability. Zero context overhead.

---

### Strategy 2 — One-Shot

One labelled example is prepended to establish the expected output format.

```
You are a movie review sentiment classifier.
Classify the sentiment as exactly one word: positive or negative.

Example:
Review: "A breathtaking masterpiece. The performances were sublime and the direction was inspired."
Sentiment: positive

Now classify this review with one word only:
Review: "<review_text>"
Sentiment:
```

**Characteristics:** Adds format anchoring through a single positive example. Improves output consistency over Zero-Shot for edge cases and ambiguous phrasing.

---

### Strategy 3 — Few-Shot

Three labelled examples are provided, covering a strong positive, a strong negative, and a tricky negative (designed to counter surface-word overweighting).

```
You are an expert movie review sentiment classifier.
Classify each review as exactly one word: positive or negative.

Examples:
Review: "An absolute masterpiece. One of the greatest films ever made. Unforgettable."
Sentiment: positive

Review: "Terrible acting, lazy writing, a complete waste of two hours of my life."
Sentiment: negative

Review: "Started with promise but descended into a confusing, painfully dull mess."
Sentiment: negative

Now classify this review with one word only:
Review: "<review_text>"
Sentiment:
```

**Characteristics:** The third example — "started with promise but..." — specifically targets the model's tendency to overweight early positive phrases in mixed reviews.

---

### Strategy 4 — Chain-of-Thought (CoT)

A four-step reasoning chain is constructed programmatically per review using lexicon-based keyword detection before the model produces its final label.

**Step 1 — Keyword detection:** Positive and negative sentiment words are identified against curated `POSITIVE_WORDS` and `NEGATIVE_WORDS` sets.

**Step 2 — Negation detection:** `NEGATION_WORDS` (e.g. *not*, *never*, *barely*) are scanned with a 3-word lookahead window. If a negation–negative co-occurrence is detected, that signal's polarity is flipped from negative to positive.

**Step 3 — Signal weighting:** Adjusted positive and negative signal counts are compared. The majority determines the reasoning conclusion.

**Step 4 — Label:** The model outputs a single-word label. If the model returns `unknown`, `smart_fallback()` resolves the label using the lexicon counts directly — ensuring no invalid entries in the results table.

**Characteristics:** Highest explainability of all four strategies. Negation handling adds robustness over surface-word matching alone. Full reasoning trace is displayed per review in the output.

---

## Evaluation

All four strategies are evaluated on the same 20-review balanced test set. A unified comparison table is generated at runtime:

| # | Strategy | Explainability | Speed |
|---|---|---|---|
| ⚡ | Zero-Shot | Low | Fastest (~10 tokens/query) |
| 1️⃣ | One-Shot | Medium | Fast |
| 📚 | Few-Shot | Medium-High | Moderate |
| 🧠 | Chain-of-Thought | Very High | Slowest (reasoning construction) |

Accuracy values, correct counts, and error counts are computed dynamically at runtime and printed alongside a `█░` progress bar per strategy.

The comparison cell also highlights the best-performing strategy (🏆), the fastest (Zero-Shot), and the most explainable (Chain-of-Thought).

---

## Visualizations

Cell 10 renders a 2×2 figure on a dark `#1a1a2e` background with four charts:

| Chart | Type | Content |
|---|---|---|
| 1 | Grouped bar chart | Accuracy (%) per strategy — bars coloured green ≥90%, orange ≥80%, red <80% |
| 2 | Stacked bar chart | Correct vs. error count per strategy |
| 3 | Radar / polar chart | Multi-axis comparison across accuracy, speed, and explainability |
| 4 | Per-review heatmap | Correct/incorrect result per strategy per individual review |

---

## Safety & Bias Analysis

Five failure modes are identified and documented in Cell 11:

**1. Positivity Bias** — Reviews with nostalgic positive words (*loved*, *hero*) are mislabelled as positive even when the overall verdict is negative. Mitigation: include nostalgic-but-negative examples in Few-Shot templates.

**2. HTML / Noise Bias** — Raw IMDB reviews contain `<br />` and other HTML markup that the tokenizer may misinterpret as content. Mitigation: `clean_review()` is applied to all inputs throughout the notebook.

**3. Sarcasm & Irony Blindness** — Mixed or ironic reviews (*"amazing start, awful ending"*) mislead Zero-Shot by surface-level word association. Mitigation: include ironic examples in Few-Shot and expand CoT lexicons.

**4. Verbosity Truncation Bias** — Long reviews truncated to 400 characters may lose the final sentiment signal. Mitigation: apply extractive summarisation as a preprocessing step, or upgrade to `flan-t5-large`.

**5. Harmful Output Risk** — Incorrectly labelled few-shot examples corrupt model outputs systematically because the model treats provided labels as authoritative ground truth. Mitigation: audit all few-shot examples before production deployment.

---

## Interactive Review Test

Cell 12 accepts any user-typed movie review and runs all four strategies in sequence. Output includes:

- Raw model output and extracted label per strategy
- Full CoT step trace (keywords → negation → weighting → label)
- Positive / negative vote count across all four strategies
- Majority-vote final verdict with confidence bar `[████░] 3/4`
- Confidence level string: `HIGH CONFIDENCE` (4/4), `LIKELY` (3/4), `MIXED / AMBIGUOUS` (2/4)

If no input is provided, an ambiguous default review is used to demonstrate split-verdict behaviour.

---

## Quick Start

> **Prerequisites:** A Google account, a GPU-enabled Colab runtime (T4 recommended), and an IMDB CSV file with `review` and `sentiment` columns.

**Step 1 — Open in Colab**

Upload `NLP_FINAL_PROJECT.ipynb` to [Google Colab](https://colab.research.google.com). Set runtime to GPU via `Runtime → Change runtime type → T4 GPU`.

**Step 2 — Run Cell 1 (Installation)**

```python
!pip install transformers torch sentencepiece pandas matplotlib seaborn scikit-learn -q
```

**Step 3 — Run Cell 2 (Data Upload)**

Upload your IMDB CSV when prompted. Label distribution and average review length are printed automatically.

**Step 4 — Run Cell 3 (Model Loading)**

`google/flan-t5-base` is downloaded from HuggingFace Hub (~990 MB). GPU availability and parameter count are confirmed.

**Step 5 — Run Cells 4–12 in sequence**

Each cell is self-contained and prints its own results. Cell 12 pauses for user keyboard input.

---

## Dependencies

| Library | Purpose |
|---|---|
| `transformers` | FLAN-T5 model and tokenizer loading via HuggingFace Hub |
| `torch` | Inference engine and GPU/CPU device management |
| `sentencepiece` | T5 SentencePiece tokenizer subword encoding |
| `pandas` | Dataset loading, filtering, and test set sampling |
| `matplotlib` | 4-chart result visualization on dark theme |
| `seaborn` | Chart styling and heatmap rendering |
| `scikit-learn` | Accuracy and classification metric utilities |
| `re` | HTML tag stripping and label extraction via regex |

---

## Project Structure

```
NLP_FINAL_PROJECT.ipynb
│
├── Cell 01  — Library installation
├── Cell 02  — Dataset upload and inspection
├── Cell 03  — FLAN-T5-Base model loading
├── Cell 04  — Helper functions: clean_review · generate_response · extract_label · evaluate
├── Cell 05  — Balanced test set sampling (20 reviews, random_state=99)
├── Cell 06  — Strategy 1: Zero-Shot evaluation
├── Cell 07  — Strategy 2: One-Shot evaluation
├── Cell 08  — Strategy 3: Few-Shot evaluation
├── Cell 09  — Strategy 4: Chain-of-Thought evaluation
├── Cell 10  — Comparison table + 4-chart visualization
├── Cell 11  — Safety & bias reflection (5 failure modes)
└── Cell 12  — Interactive custom review test (majority-vote verdict)
```

---

## Key Design Decisions

**Frozen model, no fine-tuning.** The experiment deliberately avoids any parameter updates to isolate the effect of prompt design on model behaviour. This reflects a realistic production constraint where retraining is computationally expensive.

**Balanced test set.** Equal positive and negative samples prevent majority-class bias from inflating reported accuracy figures.

**Deterministic sampling.** `random_state=99` ensures that the 20-review test set is identical across all runs and across all users sharing the notebook.

**Negation handling in CoT.** A sliding 3-word lookahead window after each negation token detects polarity flips that surface-word classifiers miss, which is the primary accuracy advantage of CoT over Zero-Shot on ambiguous and mixed-sentiment reviews.

**Smart fallback.** When the model returns `unknown`, `smart_fallback()` resolves the prediction using lexicon signal counts directly, ensuring the results table remains fully populated with valid labels.

---

## Roadmap

- [ ] Extend to five-class sentiment (very positive / positive / neutral / negative / very negative)
- [ ] Add `flan-t5-large` (780M) and `flan-t5-xl` (3B) for model-size ablation study
- [ ] Evaluate on domain-shifted corpora — product reviews, restaurant reviews, social media
- [ ] Add self-consistency prompting — sample multiple outputs and apply majority vote
- [ ] Build a Gradio interface for interactive non-technical demonstration
- [ ] Automated few-shot example selection using embedding similarity retrieval

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6a0dad,50:a855f7,100:06b6d4&height=100&section=footer"/>

**Built with FLAN-T5 · HuggingFace Transformers · Python · Google Colab**

*If this project was useful to you, consider giving it a ⭐*

</div>
