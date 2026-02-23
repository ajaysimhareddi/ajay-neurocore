<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:1a3a5c,100:3fb950&height=220&section=header&text=Voice-Based%20Stress%20Detection&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=MFCC%20%2B%20Time-Series%20Features%20%C2%B7%20RAVDESS%20Dataset%20%C2%B7%20ML%20Classification%20Pipeline&descAlignY=60&descSize=15"/>

<br/>

<img src="https://img.shields.io/badge/Dataset-RAVDESS-3fb950?style=for-the-badge&logo=soundcloud&logoColor=white"/>
<img src="https://img.shields.io/badge/Classifiers-SVM%20%7C%20RF%20%7C%20KNN%20%7C%20GBM-58a6ff?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Features-187%20Acoustic-a371f7?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Task-Binary%20Stress%20Classification-f85149?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>

</div>

---

## Overview

This project presents a complete end-to-end machine learning pipeline for detecting psychological stress from speech signals, using the **RAVDESS Emotional Speech Audio** dataset. The system classifies audio recordings as **normal** or **stressed** by extracting a 187-dimensional acoustic feature vector per file and training five classical machine learning classifiers.

The pipeline covers every stage from raw `.wav` ingestion to final model evaluation: silence trimming, amplitude normalization, Hanning window framing, STFT and Mel spectrogram analysis, 40-coefficient MFCC extraction, MFCC delta dynamics, MFCC time-series trajectory and autocorrelation analysis, pitch contour estimation via `pyin`, zero crossing rate, spectral centroid, spectral contrast, RMS energy, variance and jitter analysis, correlation heatmapping, and multi-classifier benchmarking complete with confusion matrices, ROC curves, learning curves, and Random Forest feature importances.

Fifteen publication-quality plots are generated throughout, all rendered on a consistent dark GitHub-style theme matching `#0d1117`. This project was developed as an Applied Time-Series and Signal Processing case study combining classical DSP analysis with supervised ML classification.

---

## Research Objective

> **Can acoustic and time-series features extracted from short speech segments — without any deep learning — reliably distinguish stressed from non-stressed speech, and which signal features and classifiers contribute most to that separation?**

---

## Dataset

| Property | Value |
|---|---|
| Name | RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song |
| Source | Kaggle — `uwrfkaggler/ravdess-emotional-speech-audio` |
| Format | `.wav` · 22,050 Hz sample rate · 16-bit PCM |
| Actors | 24 professional actors (12 male, 12 female) |
| Original emotions | 8 classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Binary mapping | `normal` → neutral, calm, happy, sad &nbsp;·&nbsp; `stressed` → angry, fearful, disgust, surprised |
| Label parsing | Third hyphen-delimited field in filename → `EMOTION_MAP` → `STRESS_MAP` |
| Task | Binary stress classification |

**Filename format:** `03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`

---

## System Architecture

The notebook is structured as a 24-cell linear pipeline:

```
RAVDESS .wav files  (KaggleHub download)
           │
           ▼
   parse_ravdess_label()
   Emotion code → emotion string → normal / stressed
           │
           ▼
  ┌──────────────────────────────────────────────┐
  │           SIGNAL PREPROCESSING               │
  │  librosa.effects.trim()   top_db = 20        │
  │  librosa.util.normalize() amplitude → [-1,1] │
  │  Hanning window framing   n_fft = 2048       │
  │  hop_length = 512   SR = 22,050 Hz           │
  └──────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────┐
  │       FEATURE EXTRACTION  (187-dim)          │
  │                                              │
  │  MFCC 40 coeff  mean + std      →  80        │
  │  MFCC Delta     mean + std      →  80        │
  │  Zero Crossing Rate             →   2        │
  │  Spectral Centroid              →   2        │
  │  Spectral Rolloff               →   2        │
  │  Spectral Bandwidth             →   2        │
  │  Spectral Contrast (7 bands)    →  14        │
  │  RMS Energy                     →   2        │
  │  Pitch F0  mean + std + var     →   3        │
  │                          TOTAL  → 187        │
  └──────────────────────────────────────────────┘
           │
           ▼
  Train / Test Split  80 / 20  stratified  random_state=42
  StandardScaler  fit on train only  →  no data leakage
           │
           ▼
  ┌────────────────────────────────────────────────────────┐
  │                    CLASSIFIERS                         │
  │  SVM (RBF)        C=5,  gamma='scale'                  │
  │  SVM (Linear)     C=1,  max_iter=5000                  │
  │  KNN              k=7,  euclidean, distance weights     │
  │  Random Forest    100 estimators, max_depth=12          │
  │  Gradient Boosting 100 estimators, lr=0.1, depth=4     │
  └────────────────────────────────────────────────────────┘
           │
           ▼
  Accuracy · Precision · Recall · F1
  3-Fold StratifiedKFold CV · ROC-AUC · Learning Curves
           │
           ▼
  15 Plots  +  Final Dashboard  +  Feature Importances
```

---

## Signal Processing Pipeline

### Preprocessing

Each file passes through three sequential steps before any feature is computed. `librosa.effects.trim()` with `top_db=20` removes silence that falls more than 20 dB below the signal peak. `librosa.util.normalize()` scales amplitude to `[-1, 1]`. Files shorter than 100 ms after trimming are discarded (`return None`).

### Framing and Windowing

The normalized waveform is divided into overlapping frames with `frame_length=2048` samples and `hop_length=512` samples, giving 75% frame overlap. A Hanning window is multiplied against each raw frame before spectral transformation to suppress spectral leakage at frame boundaries.

```
SR         = 22,050 Hz
N_FFT      = 2,048 samples   →  93 ms per frame
HOP_LENGTH =   512 samples   →  23 ms per hop
Overlap    = 75%
```

### STFT and Spectrogram Analysis

STFT is computed via `librosa.stft()` and converted to decibel scale with `librosa.amplitude_to_db()`. Mel spectrograms are computed with 128 Mel filter banks via `librosa.feature.melspectrogram()`, mapped to the Mel scale to model human auditory frequency resolution. Both STFT (magma colormap) and Mel (viridis colormap) spectrograms are plotted side-by-side for a normal vs. stressed pair.

---

## Feature Engineering

### MFCC — 80 Features

40 Mel-Frequency Cepstral Coefficients are extracted per file. Mean and standard deviation across all time frames are computed per coefficient, yielding an 80-dimensional sub-vector. MFCCs compactly represent short-term spectral envelope shape in a perceptually motivated cepstral domain.

### MFCC Delta — 80 Features

First-order delta coefficients are computed via `librosa.feature.delta()`, capturing the velocity of MFCC coefficient change over time. Mean and standard deviation of deltas are appended for a further 80 features. Delta features encode temporal dynamics — how rapidly the vocal tract shape changes — a known acoustic correlate of emotional arousal and stress.

### MFCC Time-Series Analysis

Individual MFCC coefficient trajectories over time are plotted with rolling mean overlays (window = 10 frames). MFCC-1 undergoes dedicated time-series analysis including:

- **Autocorrelation** — normalized zero-lag ACF to detect phoneme periodicity (regular oscillations indicate rhythmic speech patterns)
- **Power Spectral Density** — Welch's method (`scipy.signal.welch`, `nperseg=32`) to identify dominant rhythmic frequencies
- **Rolling variance** — window = 20 frames, to track local spectral instability as a stress indicator
- **Peak detection** — `scipy.signal.find_peaks()` on the ACF to locate phoneme repetition intervals

### Pitch Contour (F0) — 3 Features

Fundamental frequency is estimated using `librosa.pyin()` (probabilistic YIN algorithm), `fmin=50 Hz`, `fmax=400 Hz`. Mean, standard deviation, and variance of F0 across voiced frames are the three extracted features. Pitch jitter is additionally computed for visualization:

```
Jitter (%) = mean(|F0[t] - F0[t-1]|) / mean(F0) × 100
```

Stressed speech consistently shows elevated mean F0 and higher jitter, reflecting increased vocal cord tension and irregular voicing patterns.

### Spectral Features — 25 Features

| Feature | Description | Features |
|---|---|---|
| Zero Crossing Rate | Frame-level sign-change rate of the waveform | 2 — mean, std |
| Spectral Centroid | Weighted mean frequency of the power spectrum | 2 — mean, std |
| Spectral Rolloff | Frequency below which 85% of spectral energy lies | 2 — mean, std |
| Spectral Bandwidth | Spread of power distribution around the centroid | 2 — mean, std |
| Spectral Contrast | Peak-to-valley ratio across 7 sub-band frequency bands | 14 — mean+std per band |
| RMS Energy | Root mean square amplitude per frame | 2 — mean, std |

### Variance and Statistical Analysis

Per-frame variance across 40 MFCC coefficients and per-coefficient variance across time frames are computed for both classes. Global kurtosis (`scipy.stats.kurtosis`) and skewness (`scipy.stats.skew`) of the flattened MFCC matrix are reported — higher-order statistics that capture distributional shape differences between calm and stressed vocal production.

---

## Feature Correlation

Eight key features are assembled into a Pearson correlation matrix including the binary stress label (`stress_binary = 1` for stressed, `0` for normal):

`MFCC_1_mean` · `MFCC_2_mean` · `MFCC_3_mean` · `ZCR_mean` · `SpCentroid_mean` · `RMS_mean` · `Pitch_mean` · `Pitch_var`

A Seaborn heatmap with `RdYlGn` colormap and annotated `r` values identifies which features most strongly discriminate the two classes. The top 8 features by absolute correlation with `stress_binary` are printed with their direction indicator (UP / DOWN relative to stress).

---

## Machine Learning

### Data Preparation

The full feature matrix `X` (shape: `[n_samples, 187]`) is cleaned with `np.nan_to_num()` — replacing NaN, `+inf`, and `-inf` with `0`, `1×10⁶`, and `-1×10⁶` — before splitting to prevent classifier failures from degenerate pitch estimates on very short voiced segments.

An 80/20 stratified split is applied with `random_state=42`. `StandardScaler` is fit exclusively on the training partition and applied identically to the test set, eliminating any distribution leakage.

### Classifiers and Configuration

| Classifier | Key Hyperparameters |
|---|---|
| SVM (RBF Kernel) | `C=5`, `gamma='scale'`, `random_state=42` |
| SVM (Linear) | `C=1`, `max_iter=5000`, `random_state=42` |
| KNN | `n_neighbors=7`, `metric='euclidean'`, `weights='distance'` |
| Random Forest | `n_estimators=100`, `max_depth=12`, `n_jobs=-1`, `random_state=42` |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`, `random_state=42` |

### Evaluation Protocol

Each classifier is evaluated on six metrics: test accuracy, training accuracy (overfitting gap = train − test), precision (weighted), recall (weighted), F1 score (weighted), and 3-fold stratified CV accuracy with standard deviation.

The best classifier is dynamically identified and re-evaluated with a full sklearn classification report, 5-fold CV accuracy, 5-fold F1, and ROC-AUC per class (one-vs-rest). A learning curve is generated for the best model by training on 8 progressively larger fractions of the dataset (10% to 100%) using a `Pipeline(StandardScaler → classifier)` to avoid scaling leakage within cross-validation folds.

---

## Visualizations

Fifteen plots are generated and saved as PNG files during execution:

| File | Description |
|---|---|
| `dataset_distribution.png` | Emotion class bar chart + Normal vs. Stressed pie chart |
| `waveforms.png` | Amplitude waveforms for both classes with RMS energy annotations |
| `framing_windowing.png` | Raw vs. Hanning-windowed frames at three frame indices |
| `spectrograms.png` | STFT spectrograms (dB, magma) — normal vs. stressed |
| `mel_spectrograms.png` | Mel spectrograms (viridis) — normal vs. stressed |
| `mfcc_heatmap.png` | 40-coefficient MFCC heatmap over time (coolwarm) |
| `mfcc_trajectories.png` | MFCC-1 through MFCC-6 time-series with rolling mean overlays and variance annotations |
| `pitch_contours.png` | F0 contours with voiced frame markers, mean F0 and jitter variance |
| `feature_comparison.png` | Overlapping histograms for 8 key features with class mean lines |
| `correlation_heatmap.png` | Pearson correlation matrix including `stress_binary` label |
| `model_comparison.png` | Grouped bar chart — 5 metrics × 5 classifiers |
| `confusion_matrices.png` | 2×3 grid of confusion matrices with accuracy and F1 in titles |
| `feature_importances.png` | Top 20 Random Forest Gini importances (horizontal bar, Blues colormap) |
| `timeseries_analysis.png` | MFCC-1 time series, rolling variance, per-coefficient variance, violin plot |
| `final_dashboard.png` | 2×2 composite — model bars, best CM (%), top-10 importances, full results table |

All visualizations share a unified dark theme (`figure.facecolor='#0d1117'`, `axes.facecolor='#161b22'`) configured once via `plt.rcParams` at import time.

---

## Results

Model performance is computed at runtime from the full RAVDESS dataset. The table structure below reflects the reported metrics — actual values are printed in the notebook output:

| Model | Train Acc | Test Acc | Precision | Recall | F1 | CV (3-fold) |
|---|---|---|---|---|---|---|
| SVM (RBF) | — | — | — | — | — | — |
| SVM (Linear) | — | — | — | — | — | — |
| KNN (k=7) | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — | — |

The final summary cell prints the best model name, full classification report, 5-fold CV score ± std, training accuracy, and overfitting gap. The per-class ROC-AUC and learning curve for the winning model follow.

---

## Feature Importances

Random Forest Gini impurity reduction importances are extracted from the trained ensemble and sorted descending. The top 20 features are plotted as a horizontal bar chart with a Blues gradient colormap. The top 10 are printed with rank and importance score. Lower-index MFCC mean and delta coefficients — encoding broad spectral envelope shape and its rate of change — typically dominate the ranking, consistent with acoustic literature on speech emotion recognition.

---

## Quick Start

> **Prerequisites:** Google account · Kaggle API token configured · GPU runtime recommended.

**Step 1 — Open in Colab**

Upload `ATS_CASE_STUDY.ipynb` to [Google Colab](https://colab.research.google.com) and select `Runtime → Change runtime type → T4 GPU`.

**Step 2 — Run Cell 1 (Installation)**

```python
for pkg in ["kagglehub","librosa","soundfile","scikit-learn",
            "seaborn","matplotlib","numpy","pandas","tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
```

**Step 3 — Run Cell 2 (Dataset Download)**

```python
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
```

A valid Kaggle API token must be present before this cell executes. Configure it via `google.colab.files.upload()` and `os.makedirs`.

**Step 4 — Run all remaining cells in order**

Feature extraction (Cell 13) is the most compute-intensive step. `pyin` pitch estimation runs on every audio file — expect 3–8 minutes on a T4 GPU depending on dataset size.

---

## Dependencies

| Library | Role |
|---|---|
| `librosa` | Audio loading, MFCC, STFT, Mel spectrogram, `pyin`, spectral features |
| `numpy` | Feature vector construction, statistics, array operations |
| `pandas` | Metadata DataFrame, rolling windows, feature comparison |
| `scikit-learn` | Classifiers, StandardScaler, cross-validation, metrics, learning curves |
| `matplotlib` | All visualizations — waveforms, spectrograms, dashboards |
| `seaborn` | Correlation heatmaps, violin plots |
| `scipy` | Autocorrelation, Welch PSD, `find_peaks`, kurtosis, skewness |
| `tqdm` | Progress bar for batch feature extraction loop |
| `soundfile` | Auxiliary WAV I/O backend for librosa |
| `kagglehub` | RAVDESS dataset download from Kaggle |

---

## Project Structure

```
ATS_CASE_STUDY.ipynb
│
├── Cell 01   Install & import all dependencies
├── Cell 02   Download RAVDESS dataset via KaggleHub
├── Cell 03   Emotion label parsing  (EMOTION_MAP + STRESS_MAP)
├── Cell 04   Dataset distribution visualization  (bar + pie)
├── Cell 05   Load sample audio — normal vs. stressed playback
├── Cell 06   Waveform comparison with RMS energy annotation
├── Cell 07   Signal framing & Hanning window visualization
├── Cell 08   STFT + Mel spectrogram comparison
├── Cell 09   MFCC extraction & 40-coefficient heatmap
├── Cell 10   MFCC time-series trajectories  (MFCC-1 to MFCC-6)
├── Cell 11   Pitch contour (pyin F0 + voiced frame markers)
├── Cell 11b  ZCR · Spectral Centroid · RMS Energy time-series
├── Cell 11c  MFCC per-coefficient plots + mean±std across all 40
├── Cell 11d  Autocorrelation · PSD · rolling mean of MFCC-1
├── Cell 11e  Variance & frequency jitter analysis
├── Cell 12   extract_all_features()  — full 187-feature function
├── Cell 13   Batch feature extraction across entire dataset
├── Cell 14   Feature comparison histograms + group statistics
├── Cell 15   Feature correlation heatmap (incl. stress binary)
├── Cell 16   Train/test split (80/20 stratified) + StandardScaler
├── Cell 17   Model training — SVM · KNN · RF · Gradient Boosting
├── Cell 18   Model comparison bar chart  (5 metrics × 5 classifiers)
├── Cell 19   Confusion matrices — all classifiers  (2×3 grid)
├── Cell 19b  ROC curves — best model  (one-vs-rest AUC)
├── Cell 19c  Evaluation metrics bar chart + learning curves + 5-fold CV
├── Cell 20   Detailed classification report  (best model)
├── Cell 21   Random Forest feature importances  (top 20)
├── Cell 22   MFCC time-series variance & seasonality analysis
├── Cell 23   Final summary dashboard  (2×2 composite figure)
└── Cell 24   Full project summary report  (printed to stdout)
```

---

## Key Design Decisions

**187-dimensional feature vector.** The feature set is deliberately broad, combining time-domain, frequency-domain, cepstral, and temporal-dynamic representations. Including MFCC delta coefficients captures speech dynamics — how rapidly the vocal tract shape is changing — which is a known acoustic correlate of arousal and stress that static MFCC means alone cannot encode.

**pyin over yin for pitch estimation.** `pyin` (probabilistic YIN) returns explicit voiced frame confidence flags, allowing silent and unvoiced frames to be excluded from F0 statistics via `np.isnan()` filtering. Simple `yin` applies no such frame validity detection, inflating pitch variance with silent segment estimates.

**Stratified splitting at every stage.** The 80/20 train-test split and all 3-fold and 5-fold cross-validation folds use `stratify=y` to preserve the balanced class distribution in every data partition, preventing accuracy inflation from class imbalance artefacts.

**NaN and infinity guard before splitting.** `np.nan_to_num()` is applied to the full feature matrix before any train-test splitting. This prevents degenerate pyin estimates on very short voiced segments from propagating into classifier training or evaluation, which would cause `ValueError` or silently distort model parameters.

**Global dark theme via `rcParams`.** All `matplotlib.rcParams` are configured once at import time (`#0d1117`, `#161b22`, white axes). This eliminates per-plot style boilerplate and guarantees visual cohesion across all 15 saved figures without redundant configuration code in each cell.

**Learning curve with full pipeline.** The learning curve is generated using a `Pipeline(StandardScaler → classifier)` object, ensuring that `StandardScaler` is fit independently on each cross-validation training fold. Without this, scaling the full dataset before CV would constitute data leakage and inflate reported validation accuracy.

---

## Limitations and Future Work

This project establishes a strong classical ML baseline on a controlled studio dataset. Several directions extend this work:

- **Deep learning:** A CNN applied directly to Mel spectrograms, or an LSTM/GRU on MFCC time-series, would capture spatial and temporal dependencies that frame-level aggregated statistics cannot represent.
- **Speaker normalization:** RAVDESS uses professional actors in a controlled acoustic environment. Real-world stress detection requires removing inter-speaker pitch range and vocal tract size differences through speaker-level normalization or speaker-adaptive features.
- **Fine-grained labels:** The binary mapping merges angry, fearful, disgusted, and surprised into a single "stressed" class. A multi-class model separating these emotions would be more clinically meaningful and would reveal which emotions the feature set best discriminates.
- **Real-time streaming:** A live implementation processing overlapping frames from a microphone feed via `sounddevice` would enable deployment in call centre monitoring, driver alertness systems, or clinical mental health screening applications.
- **Dataset diversity:** Combining RAVDESS with CREMA-D, TESS, or EmoDB would substantially increase training set size, speaker demographics, and acoustic environment variety, improving generalization to real-world speech.

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3fb950,50:1a3a5c,100:0d1117&height=120&section=footer"/>

**Built with librosa · scikit-learn · NumPy · Matplotlib · Google Colab**

*If this project was useful to you, consider giving it a ⭐*

</div>
