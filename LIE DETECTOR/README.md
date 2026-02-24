<div align="center">

```
██╗   ██╗███████╗██████╗ ██╗████████╗ █████╗ ███████╗
██║   ██║██╔════╝██╔══██╗██║╚══██╔══╝██╔══██╗██╔════╝
██║   ██║█████╗  ██████╔╝██║   ██║   ███████║███████╗
╚██╗ ██╔╝██╔══╝  ██╔══██╗██║   ██║   ██╔══██║╚════██║
 ╚████╔╝ ███████╗██║  ██║██║   ██║   ██║  ██║███████║
  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
```

### **AI-Powered Multimodal Deception Analysis System**
*Computer Vision · Voice Stress · Micro-Expressions · NLP Fusion*

<br/>

[![Status](https://img.shields.io/badge/Status-Active-39ff14?style=for-the-badge&labelColor=0a0f1e)](#)
[![Version](https://img.shields.io/badge/Version-2.4.0-00d4ff?style=for-the-badge&labelColor=0a0f1e)](#)
[![License](https://img.shields.io/badge/License-MIT-ff3c6e?style=for-the-badge&labelColor=0a0f1e)](#)
[![Browser](https://img.shields.io/badge/Platform-Browser%20Native-ffb800?style=for-the-badge&labelColor=0a0f1e)](#)
[![No Backend](https://img.shields.io/badge/Backend-None%20Required-39ff14?style=for-the-badge&labelColor=0a0f1e)](#)

<br/>

> *"The face tells what the voice conceals, and the voice reveals what words deny."*

<br/>

---

</div>

<br/>

## 🧠 What is VERITAS?

**VERITAS** is a real-time, browser-native deception analysis engine that fuses **6 simultaneous behavioral data streams** into a single **Truthfulness Confidence Score**. Unlike single-channel systems (e.g., polygraph which only measures physiological signals), VERITAS cross-validates multiple modalities — making it significantly harder to fool and far more probabilistically accurate.

It runs **entirely in your browser** using the Web Audio API, Canvas 2D API, and getUserMedia — zero server calls, zero data leakage, zero setup.

```
Subject → Camera + Microphone → 6-Channel Analysis Engine → Truth Score (0–100%)
```

<br/>

---

<br/>

## ⚡ Quick Start

```bash
# No installation needed. Just open the file.
open lie-detector.html
```

**Or serve it locally:**
```bash
# Python
python -m http.server 8080

# Node.js
npx serve .

# Then visit → http://localhost:8080/lie-detector.html
```

> **Requirements:** Any modern browser (Chrome 90+, Edge 90+, Firefox 88+, Safari 14+) · Webcam · Microphone
>
> 🎭 **No camera?** The system auto-activates **Demo Mode** with full simulated biometric signals — every feature works.

<br/>

---

<br/>

## 🔬 Analysis Channels

VERITAS operates **6 behavioral channels in parallel**, each contributing a weighted score to the final truth probability.

<br/>

### `[CH-01]` 😶 Micro-Expression Detection

Involuntary facial movements lasting **40–500 milliseconds** that leak suppressed emotional states. These are nearly impossible to consciously control and are one of the strongest known deception indicators.

| Expression | Deception Correlation | Description |
|---|:---:|---|
| `NEUTRAL` | Low | Baseline resting state |
| `MICRO-FEAR` | **High** | Involuntary fear flash — strong deception marker |
| `SUPPRESSED SMILE` | Medium | Duping delight — joy at deceiving successfully |
| `BROW FURROW` | Medium | Cognitive effort, discomfort |
| `NOSTRIL FLARE` | **High** | Fight-or-flight activation, stress response |
| `LIP COMPRESSION` | Medium | Suppressed speech, withheld information |
| `EYE WIDENING` | Low | Surprise, genuine emotional response |

**Weight in final score:** `25%`

---

### `[CH-02]` 👁 Eye Movement & Blink Analysis

Involuntary oculomotor signals that reveal cognitive processing patterns and autonomic arousal states.

```
Blink Rate Baseline:   15–20 blinks/min  →  Normal
Deception Threshold:   >25 blinks/min    →  Elevated arousal flag
Gaze Direction:        Rightward shift   →  Confabulation indicator
                       Downward shift    →  Shame / concealment
Gaze Stability:        <70% stable       →  Evasive behavior flag
```

**Weight in final score:** `20%`

---

### `[CH-03]` 🎙 Voice Stress Analysis

Real microphone frequency data processed through a **32-band spectral analyzer**. Stress manifests as pitch variance, tremor, and abnormal energy distribution across frequency bands.

```
█  Green  bars  →  Normal vocal energy      (amplitude < 120)
█  Yellow bars  →  Elevated stress signal   (amplitude 120–180)
█  Red    bars  →  High deception marker    (amplitude > 180)
```

**Signals measured:**
- Fundamental frequency (F0) variance
- High-frequency tremor index
- Speech energy distribution across 32 bands
- Micro-pause frequency between words

**Weight in final score:** `25%`

---

### `[CH-04]` 🧠 NLP Linguistic Analysis

Rule-based linguistic pattern engine identifying **deceptive speech markers** embedded in word choice, sentence structure, and narrative specificity.

<details>
<summary><b>📋 Click to expand: Full Signal Dictionary</b></summary>

<br/>

**🔴 Deception Indicators** *(lower truth score)*

| Pattern | Examples | Risk Weight |
|---|---|:---:|
| Hedge Words | `maybe`, `perhaps`, `I think`, `sort of`, `roughly` | `+4 pts` |
| Protest Phrases | `trust me`, `I swear`, `believe me`, `honestly` | `+8 pts` |
| Filler Words | `um`, `uh`, `like`, `you know` | `+1.5 pts` |
| Excessive Negation | `I would never`, `not`, `nobody`, `never` | `+3 pts` |
| Evasive Brevity | Responses under 10 words | `+8 pts` |
| Over-Elaboration | Avg sentence >25 words (over-explaining) | `+5 pts` |

<br/>

**🟢 Truth Indicators** *(raise truth score)*

| Pattern | Examples | Trust Weight |
|---|---|:---:|
| Specific Dates | Day names, month names, years | `-3 pts` |
| Specific Numbers | Quantified facts and figures | `-2 pts` |
| Contextual Detail | Concrete, verifiable specifics | Score boost |

</details>

**Weight in final score:** `15%`

---

### `[CH-05]` ⏱ Response Delay / Cognitive Load

Truthful responses are typically **spontaneous**. Deceptive answers require real-time story construction — measurable as elevated response latency.

```
 < 400ms   →  Spontaneous response    ✅  High trust
 400–700ms →  Normal processing       🟡  Neutral
 700ms–1s  →  Elevated latency        🟠  Mild flag
 > 1000ms  →  High cognitive load     🔴  Deception marker
```

**Weight in final score:** `10%`

---

### `[CH-06]` 💓 Physiological Estimation

CV-derived estimation of physiological arousal from facial appearance micro-changes caused by autonomic nervous system activation.

- **Facial flush** — peripheral blood flow increase under stress or embarrassment
- **Perspiration estimation** — skin texture micro-changes from sympathetic activation

**Weight in final score:** `5%`

<br/>

---

<br/>

## 📊 Scoring System

The final **Truth Probability** is a temporally-smoothed weighted fusion across all 6 channels:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   T = (μ × 0.25) + (ε × 0.20) + (ν × 0.25)                   │
│       + (λ × 0.15) + (δ × 0.10) + (φ × 0.05)                  │
│                                                                 │
│   μ = Micro-expressions     ε = Eye movement                   │
│   ν = Voice stress          λ = NLP linguistic score           │
│   δ = Response delay        φ = Physiological signals          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Smoothing:** Score transitions use exponential moving average (`α = 0.1`) to prevent noise spikes from individual frames corrupting the score.

<br/>

### Verdict Thresholds

```
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ████████████████████████   ✅  TRUTHFUL
  75% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ████████████████████       🔵  LIKELY TRUTHFUL
  55% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ██████████████             🟡  INCONCLUSIVE
  40% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ████████                   🟠  LIKELY DECEPTIVE
  25% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ████                       🔴  DECEPTIVE
   0% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<br/>

---

<br/>

## 🖥 Interface Guide

```
┌──────────────────────────────────────────────────────────────────────┐
│  VERITAS ◈ DECEPTION ANALYSIS SYSTEM v2.4          ● CAMERA LIVE    │
├───────────────────────────────────┬──────────────────────────────────┤
│                                   │   TRUTH PROBABILITY              │
│  ┌─────────────────────────────┐  │                                  │
│  │    [LIVE VIDEO FEED]        │  │          ╭────────╮              │
│  │                             │  │         ╱   87%    ╲             │
│  │   ┌──────────────┐          │  │        │  TRUTHFUL  │            │
│  │   │  FACE BOX    │          │  │         ╲          ╱             │
│  │   │  👁      👁  │          │  │          ╰────────╯              │
│  │   └──────────────┘          │  ├──────────────────────────────────┤
│  │   [MICRO-EXPR TAGS]         │  │ 😶 MICRO-EXPR   ████████  82%   │
│  └─────────────────────────────┘  │ 👁 EYE MOVE     ███████   78%   │
│                                   │ 🎙 VOICE STRESS ██████    71%   │
│  [~~~~~ WAVEFORM ~~~~~~~~~~~~~~~~]│ 🧠 NLP SCORE    █████████ 91%   │
│                                   │ ⏱ RESPONSE DLY  ███████   76%   │
│  [TEXT INPUT → NLP ANALYSIS]      │ 💓 PHYSIO       ████████  80%   │
│  [▶ START] [■ STOP] [↺] [⬇]      ├──────────────────────────────────┤
│                                   │  BEHAVIORAL ANALYSIS FEED        │
│                                   │  00:12 ● Blink rate: 18 bpm ✓   │
│                                   │  00:09 ⚠ Micro-fear detected    │
│                                   │  00:06 ● Voice stress: LOW       │
└───────────────────────────────────┴──────────────────────────────────┘
```

### Button Reference

| Control | Action |
|---|---|
| `▶ START ANALYSIS` | Activates camera, mic, and all 6 analysis channels simultaneously |
| `■ STOP` | Ends session, freezes final scores |
| `↺ RESET` | Clears all data, returns to standby |
| `⬇ EXPORT` | Downloads complete `.txt` session report |
| `⟳ RUN NLP ANALYSIS` | Runs linguistic analysis on typed statement |

<br/>

---

<br/>

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VERITAS ENGINE v2.4                          │
├─────────────────┬───────────────────────┬───────────────────────────┤
│  INPUT LAYER    │   PROCESSING LAYER    │     OUTPUT LAYER          │
│                 │                       │                           │
│ 📷 getUserMedia │ CV Analysis           │ Truth Score Ring          │
│ 🎙 Web Audio   │ ├─ Face bounding box  │ Animated verdict badge    │
│ ⌨  Text Input  │ ├─ Eye tracking canvas│ 6-channel metric bars     │
│                 │ └─ Micro-expressions  │ Behavioral event feed     │
│                 │                       │ 40-block timeline         │
│                 │ Audio Analysis        │ Deception flag alerts     │
│                 │ ├─ Waveform render    │ Exportable report         │
│                 │ ├─ FFT 32-band        │                           │
│                 │ └─ Stress visualizer  │                           │
│                 │                       │                           │
│                 │ NLP Engine            │                           │
│                 │ ├─ Hedge detection    │                           │
│                 │ ├─ Protest phrases    │                           │
│                 │ └─ Specificity score  │                           │
│                 │                       │                           │
│                 │ Score Fusion          │                           │
│                 │ ├─ Weighted average   │                           │
│                 │ ├─ EMA smoothing      │                           │
│                 │ └─ Verdict classifier │                           │
└─────────────────┴───────────────────────┴───────────────────────────┘
```

<br/>

---

<br/>

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Video Capture** | `getUserMedia API` | Real-time webcam stream acquisition |
| **Audio Processing** | `Web Audio API` · `AnalyserNode` | FFT frequency analysis, waveform data |
| **Visual Rendering** | `Canvas 2D API` | Waveform draw, eye-tracking overlay |
| **Face Overlay** | `CSS Absolute Positioning` | Bounding box, corner brackets, expression tags |
| **NLP Engine** | `Vanilla JavaScript` | Rule-based linguistic deception classifier |
| **Animations** | `CSS Keyframes` · `CSS Transitions` | Scan lines, pulsing ring, metric transitions |
| **Typography** | `Orbitron` · `Share Tech Mono` · `Exo 2` | High-contrast interface aesthetics |
| **Deployment** | Single `.html` file | Zero-dependency, zero-server, portable |

<br/>

---

<br/>

## 📁 File Structure

```
veritas/
│
├── 📄 lie-detector.html     ←  Complete self-contained application
└── 📄 README.md             ←  This file
```

> The entire application — HTML, CSS, and JavaScript — is bundled in a **single portable file**. No `node_modules`, no build step, no runtime dependencies beyond Google Fonts.

<br/>

---

<br/>

## 🔭 Roadmap

### Phase 1 — Real Computer Vision
```
[ ] TensorFlow.js FaceMesh — 468-point facial landmark tracking
[ ] MediaPipe Face Detection — production-grade face localization
[ ] True blink detection via Eye Aspect Ratio (EAR) calculation
[ ] FACS Action Unit (AU) coding for clinical micro-expression grading
```

### Phase 2 — Clinical Voice Analysis
```
[ ] Proper PSE (Psychological Stress Evaluator) algorithms
[ ] Jitter and shimmer measurement (clinical voice tremor markers)
[ ] Fundamental frequency (F0) extraction and trend tracking
[ ] Harmonics-to-Noise Ratio (HNR) for vocal quality scoring
```

### Phase 3 — ML-Powered NLP
```
[ ] Fine-tune BERT on Columbia Statement Deception corpus
[ ] Train on Real-Life Trial dataset (1,000+ labeled statements)
[ ] Add multilingual NLP (Spanish, Hindi, Arabic, Mandarin)
[ ] Semantic coherence scoring beyond keyword pattern matching
```

### Phase 4 — Production Grade
```
[ ] 5-minute baseline calibration phase with neutral questions
[ ] WebRTC session recording with frame-accurate analysis overlay
[ ] Multi-subject side-by-side comparison mode
[ ] REST API endpoint for third-party platform integration
[ ] Electron desktop wrapper for offline deployment
```

<br/>

---

<br/>

## ⚠️ Disclaimer & Ethical Notice

```
╔══════════════════════════════════════════════════════════════════════╗
║  IMPORTANT — PLEASE READ BEFORE USE                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. NO deception detection technology — including clinical           ║
║     polygraph — achieves 100% accuracy. False positives and          ║
║     false negatives are inherent to all such systems.                ║
║                                                                      ║
║  2. VERITAS produces probabilistic indicators, not verdicts.         ║
║     Results must NEVER be used as evidence in any legal,             ║
║     employment, security, or disciplinary context.                   ║
║                                                                      ║
║  3. Micro-expression and physiological signals in v2.4 are           ║
║     SIMULATED via behavioral models. Real clinical inference         ║
║     requires dedicated, trained ML models.                           ║
║                                                                      ║
║  4. All processing is local and ephemeral. No biometric data         ║
║     is transmitted, stored, or logged anywhere.                      ║
║                                                                      ║
║  5. Deploying deception analysis tools without consent may           ║
║     violate privacy laws in your jurisdiction. Use responsibly.      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

<br/>

---

<br/>

## 📜 License

```
MIT License — Copyright (c) 2025 VERITAS Project

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software to use, copy, modify, merge, publish, and
distribute, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

<br/>

---

<br/>

<div align="center">

```
VERITAS v2.4  ·  Web Audio API  ·  Canvas API  ·  getUserMedia
Zero dependencies  ·  Zero backend  ·  Zero data collection
```

**[⬆ Back to Top](#)**

</div>
