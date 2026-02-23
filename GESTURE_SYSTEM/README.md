<div align="center">

<br/>

```
╔══════════════════════════════════════════════════════════╗
║          ✋  CV GESTURE SYSTEM ULTRA  v3                  ║
║     Air Writing · 13 Shapes · Custom Gestures · Zero-Lag ║
╚══════════════════════════════════════════════════════════╝
```

<img src="https://img.shields.io/badge/MediaPipe-Hands-00c8ff?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/JavaScript-Vanilla-7c3aed?style=for-the-badge&logo=javascript&logoColor=white"/>
<img src="https://img.shields.io/badge/No%20Install-Zero%20Dependencies-00ff88?style=for-the-badge"/>

<br/><br/>

**A real-time, browser-based computer vision canvas controlled entirely by hand gestures.**
Draw in the air, stamp emojis, lock geometric shapes, and undo strokes — no mouse, no touch, no setup required.

<br/>

</div>

---

## ✨ Feature Highlights

| Feature | Details |
|---|---|
| 🤚 **9 Built-in Gestures** | Pointing, Fist, Peace, Thumbs Up/Down, Rock On, OK, Open Hand, Both Fists |
| 🔷 **13 Geometric Shapes** | Circle, Ellipse, Rectangle, Square, Triangle, Diamond, Star (5/6-pt), Pentagon, Hexagon, Cross, Arrow, Double Arrow |
| 🖌️ **4 Brush Styles** | Round, Square, Glow, Neon — with adjustable size & 8-color palette |
| 🎨 **Stamp Mode** | 12 emoji stamps (⭐❤️🔥💎🎯⚡🌈🚀💀👑🌸✅) with adjustable size |
| ↩️ **Undo / Redo** | 20-level history stack via Rock On gesture or `Ctrl+Z` |
| ✊✊ **Fist-to-Lock** | Hold both fists 1.5s to permanently lock any shape onto the canvas |
| ➕ **Custom Gestures** | Capture your own hand poses and bind them to actions |
| 💾 **Canvas Export** | Preview + download composite drawing as PNG |
| ⚡ **Zero-Lag Pipeline** | MediaPipe Hands runs in-browser via CDN — no Python inference bottleneck |
| 📱 **Responsive UI** | Adapts to desktop and mobile layouts automatically |

---

## 🎬 How It Works

```
webcam feed
    │
    ▼
MediaPipe Hands (browser CDN)
    │  21 landmarks per hand @ 30fps
    ▼
Gesture Classifier (JS)
    │  Pointing / Fist / Peace / OK / ThumbsUp ...
    ▼
Action Dispatcher
    │
    ├──► Draw Mode  → brush strokes on <canvas>
    ├──► Erase Mode → circular eraser at fingertip
    ├──► Stamp Mode → emoji placed at index tip
    └──► Shape Mode → ghost preview + Fist×2 to commit
```

The app runs entirely inside a single HTML page served through a lightweight Python HTTP server embedded in the Colab notebook — no external backend, no WebSocket, no cloud inference.

---

## 🚀 Quick Start

> **Requirements:** A Google account and a webcam. That's it.

**1. Open the notebook in Google Colab**

```
File → Open Notebook → Upload → CV_GESTURE_SYSTEM.ipynb
```

Or click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**2. Run Cell 1**

```python
# CELL 1 — No pip installs needed! MediaPipe runs in browser via CDN
print('✅ Ready! Run Cell 2 to launch the zero-lag interface.')
```

**3. Run Cell 2**

A local HTTP server starts on port `8765`. A Colab tunnel URL is printed and opened automatically.

**4. Allow camera access** in the new browser tab, then click **▶ START**.

---

## 🤚 Gesture Reference

| Gesture | Action |
|---|---|
| ☝️ **Index Finger (Pointing)** | Draw / Stamp / Erase depending on active mode tab |
| ✊ **Fist** | Lift pen — pauses drawing without switching mode |
| ✊✊ **Both Fists** | Hold 1.5 seconds → lock current shape permanently onto canvas |
| 👍 **Thumbs Up** | Instant eraser at thumb tip |
| 👎 **Thumbs Down** | Cycle draw color through the palette |
| 👌 **OK Sign** | Stamp a glowing circle at the hand position |
| ✌️ **Peace** | Shape resize (use two hands to scale) |
| 🤘 **Rock On** | Undo last stroke |
| 🖐 **Open Hand** | Pause / freeze current tool |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl + Z` | Undo |
| `Space` | Clear canvas |
| `1` | Draw mode |
| `2` | Shape mode |
| `3` | Stamp mode |
| `4` | Erase mode |

---

## 🔷 Shape Mode — Step by Step

1. Click the **🔷 Shapes** tab and select a shape (e.g. Star 5pt)
2. Configure **Fill color**, **Stroke color**, and **Opacity**
3. Show **two open hands** to the camera — the shape preview tracks your hand position
4. When you're happy with the placement, close **both hands into fists** simultaneously
5. Hold for **1.5 seconds** — the animated ring fills up → shape locks to canvas 🔒

---

## ➕ Adding Custom Gestures

1. Hold any hand pose you want to teach
2. Click **➕ GESTURE** in the toolbar
3. Name your gesture, assign an emoji, and choose its action
4. Your gesture is saved in `localStorage` and appears in the **Gesture Guide** panel

---

## 🏗️ Architecture

```
CV_GESTURE_SYSTEM.ipynb
│
├── Cell 1  ─── Sanity check (no installs needed)
│
└── Cell 2  ─── Main app
    │
    ├── Python layer
    │   ├── http.server.HTTPServer  (serves APP_HTML on port 8765)
    │   ├── threading.Thread        (non-blocking background server)
    │   └── google.colab.output.eval_js  (tunnel URL retrieval)
    │
    └── Browser layer (single HTML string)
        ├── <video>   — raw webcam feed (hidden)
        ├── #camCanvas  — mirrored video frame render
        ├── #drawCanvas — persistent freehand strokes
        ├── #shapeCanvas — live shape ghost preview
        ├── #uiCanvas — landmarks, cursor, HUD overlays
        │
        ├── MediaPipe Hands  (CDN, runs on device GPU/CPU)
        │   └── 21 3D landmarks → classifyGesture()
        │
        ├── Gesture Engine
        │   ├── isFist()         — finger-curl detection
        │   ├── classifyGesture() — rule-based pose matching
        │   └── customGestures   — user-defined landmark snapshots
        │
        ├── Drawing Engine
        │   ├── Catmull-Rom smoothing (SMOOTH_N = 3)
        │   ├── Brush styles: round / square / glow / neon
        │   ├── Undo stack (MAX_UNDO = 20, ImageData snapshots)
        │   └── Stamp renderer (emoji on 2D canvas)
        │
        └── Shape Engine
            ├── 13 shape renderers (arc, path, polygon helpers)
            ├── Two-hand scale detection (Peace gesture)
            ├── Both-fists lock countdown (SVG ring animation)
            └── lockedShapes[] → merged into drawCanvas on commit
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Hand tracking | [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) via CDN |
| Rendering | HTML5 Canvas API (4 stacked layers) |
| Server | Python `http.server` + `threading` |
| Colab bridge | `google.colab.output.eval_js` |
| Fonts | Rajdhani + Share Tech Mono (Google Fonts) |
| Dependencies | **None** (everything loaded from CDN at runtime) |

---

## 💡 Tips for Best Results

- **Lighting:** Use a well-lit environment — natural or front-facing light works best
- **Background:** Plain, uncluttered backgrounds improve landmark detection accuracy
- **Distance:** Keep your hand 40–70 cm from the camera
- **Speed:** Slow, deliberate gestures reduce false positives
- **Both-Fist Lock:** Make sure both hands are fully visible before clenching

---

## 🗺️ Roadmap

- [ ] Voice command integration alongside gesture control
- [ ] Multi-layer canvas with individual layer management
- [ ] Text tool — spell words letter-by-letter in the air
- [ ] WebSocket mode for collaborative multi-user canvas
- [ ] Gesture macro recording and playback
- [ ] Export as SVG (vector paths from landmark trajectories)
- [ ] Standalone web app (no Colab dependency)

---

## 📄 License

MIT © — free to use, fork, and build upon.

---

<div align="center">

**Built with MediaPipe · Python · Vanilla JS · ☕**

*If this project helped you, consider giving it a ⭐*

</div>
