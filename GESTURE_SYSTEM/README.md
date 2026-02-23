<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c8ff,50:7c3aed,100:00ff88&height=200&section=header&text=CV%20Gesture%20System%20&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Real-Time%20Hand%20Gesture%20Canvas%20%E2%80%94%20Air%20Writing%20%C2%B7%2013%20Shapes%20%C2%B7%20Zero-Lag&descAlignY=60&descSize=16"/>

<br/>

<img src="https://img.shields.io/badge/Version-3.0%20Ultra-00c8ff?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/MediaPipe-Hands-7c3aed?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/Zero%20Install-CDN%20Only-00ff88?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-ffffff?style=for-the-badge"/>

</div>

---

## Overview

CV Gesture System Ultra is a real-time, browser-based drawing canvas controlled entirely through hand gestures. Using **MediaPipe Hands** running fully in-browser via CDN, the system tracks 21 hand landmarks per frame and maps them to a rich set of drawing actions — no installation, no Python ML dependencies, no external servers.

The application runs as a single self-contained HTML page served from inside a Google Colab notebook via a lightweight Python HTTP server. Opening it launches a full creative suite: freehand air drawing, 13 geometric shapes with gesture-driven placement and locking, emoji stamping, multi-brush styles, undo/redo history, and an extensible custom gesture engine — all rendered across four stacked HTML5 canvas layers at real-time frame rates.

This project was developed to explore the intersection of computer vision, human-computer interaction, and zero-dependency browser applications.

---

## System Architecture

The system is built on a two-layer model: a minimal Python backend embedded in Colab, and a fully self-contained browser frontend.

**The workflow is as follows:**

1. **Server:** A Python `http.server.HTTPServer` is launched on port `8765` inside a background `threading.Thread`. It serves the entire application as a single HTML string — no files on disk.

2. **Tunnel:** `google.colab.output.eval_js` retrieves the Colab-generated public tunnel URL, which is printed to the notebook output for the user to open.

3. **Camera:** The browser tab requests webcam access and feeds raw frames into a hidden `<video>` element.

4. **Landmark Detection:** MediaPipe Hands (loaded from jsDelivr CDN) processes each video frame and returns an array of 21 normalized 3D landmarks per detected hand at up to 30 fps.

5. **Gesture Classification:** A rule-based JavaScript classifier (`classifyGesture`) evaluates finger extension states from the landmark positions to identify one of 9 built-in gestures. Unrecognized poses are passed to a custom gesture matcher that uses normalized Euclidean distance against stored landmark snapshots.

6. **Action Dispatch:** The resolved gesture key is mapped to a drawing action — stroke, erase, stamp, shape preview, shape lock, undo, color cycle — which is applied to the appropriate canvas layer.

7. **Rendering:** Four stacked `<canvas>` elements handle separation of concerns: camera feed, persistent drawing strokes, live shape ghost preview, and real-time UI overlays (skeleton, cursor, HUD badges).

```
Webcam Feed
     │
     ▼
MediaPipe Hands (Browser CDN)
     │  21 landmarks × up to 2 hands @ ~30 fps
     ▼
classifyGesture()
     │  Finger extension rules (tip.y vs pip.y)
     │  + matchCustomGesture() for user-trained poses
     ▼
Action Dispatcher
     │
     ├──► Draw / Erase    →  #drawCanvas   (Catmull-Rom smoothed strokes)
     ├──► Shape Engine    →  #shapeCanvas  (Ghost preview → Fist×2 to lock)
     └──► Stamp / Undo   →  #drawCanvas   (ImageData snapshot stack)
```

---

## Canvas Layer Model

The visual output is composed of four independently managed `<canvas>` elements stacked via CSS `position: absolute`:

| Layer | Element | Z-Index | Purpose |
|---|---|---|---|
| Camera | `#camCanvas` | Base | Mirrored video frame + hand skeleton overlay |
| Drawing | `#drawCanvas` | 2 | Persistent freehand strokes, stamps, locked shapes |
| Shape Preview | `#shapeCanvas` | 5 | Live ghost preview of active shape (cleared each frame) |
| UI Overlay | `#uiCanvas` | 15 | HUD badges, cursor indicators, FPS counter |

This separation ensures that expensive re-renders on one layer — such as the shape ghost preview at 30 fps — never corrupt persistent drawing data on another.

---

## Gesture Reference

The gesture classifier evaluates five binary finger states (thumb, index, middle, ring, pinky) derived from landmark Y-position comparisons between fingertip and proximal joint. For the OK gesture, thumb-to-index Euclidean distance is used instead.

| Gesture | Hand Pose | Triggered Action |
|---|---|---|
| ☝️ **Pointing** | Index extended only | Draw stroke / Stamp / Erase at fingertip |
| ✊ **Fist** | All fingers curled | Lift pen — pauses stroke without mode change |
| ✊✊ **Both Fists** | Two hands, all fingers curled | Hold 1.5 s → lock active shape to canvas |
| 👍 **Thumbs Up** | Thumb extended upward | Instant eraser activated at thumb tip |
| 👎 **Thumbs Down** | Thumb extended downward | Cycle draw color through 8-color palette |
| 👌 **OK Sign** | Thumb–index pinch, others extended | Stamp a glowing circle at hand center |
| ✌️ **Peace** | Index + middle extended | Shape resize via two-hand spread |
| 🤘 **Rock On** | Index + pinky extended | Undo last stroke |
| 🖐 **Open Hand** | All four fingers extended | Pause / freeze active tool |

Custom gestures extend this table via normalized landmark snapshot matching with a configurable Euclidean distance threshold of `0.10`.

---

## Drawing Engine

**Freehand strokes** use a Catmull-Rom smoothing buffer (`SMOOTH_N = 3`) to interpolate between raw landmark positions, eliminating the jitter inherent in landmark detection. Each stroke is drawn with `lineCap: round` and `lineJoin: round` for natural-looking paths.

**Four brush styles are available:**

| Style | Rendering Technique |
|---|---|
| ● Round | Standard `lineCap: round` stroke |
| ■ Square | `lineCap: butt` with square joins |
| ✨ Glow | Multiple strokes at decreasing opacity with blur |
| ⚡ Neon | Additive composite mode with color bloom |

**Eraser** is implemented as a `clearRect` circle at the index fingertip, with adjustable radius (10–80 px) and a dashed circle cursor indicator rendered on the camera canvas.

**Undo / Redo** stores up to 20 `ImageData` snapshots of the drawing canvas. Rock On gesture and `Ctrl+Z` trigger undo. A redo stack is maintained until a new stroke invalidates it.

---

## Shape Engine

Thirteen shape types are supported. Each is drawn via a shared `renderShapePath()` function that takes two anchor points — derived from the two detected hand positions — to define center and radius dynamically.

**Available shapes:** `circle` · `ellipse` · `rectangle` · `square` · `triangle` · `diamond` · `star (5-pt)` · `star (6-pt)` · `pentagon` · `hexagon` · `cross` · `arrow` · `double arrow`

**Shape placement workflow:**

1. Select a shape from the **🔷 Shapes** tab and configure fill color, stroke color, and opacity
2. Show two open hands — the shape ghost preview tracks hand positions in real time on `#shapeCanvas`
3. Close both hands into fists simultaneously
4. An SVG ring countdown animation fills over **1,500 ms** (`LOCK_HOLD_MS`)
5. On completion, the shape is committed to `#drawCanvas` and added to `lockedShapes[]`

The Peace gesture enables dynamic scaling of the shape preview by measuring inter-hand distance in real time.

---

## Custom Gesture Engine

Users can train the system to recognize any hand pose:

1. Hold the desired hand shape in front of the camera
2. Click **➕ GESTURE** to capture the current 21-landmark snapshot
3. Assign a name, emoji, and associated drawing action

At runtime, `matchCustomGesture()` normalizes incoming landmarks relative to the wrist position and wrist-to-middle-knuckle scale, then computes mean Euclidean distance across 20 landmarks against each stored snapshot. The closest match below threshold `0.10` wins. Custom gestures are persisted in `localStorage`, appear in the Gesture Guide panel with a purple `CUSTOM` badge, and can be deleted individually.

---

## Tool Modes

| Mode | Tab | Activation Gesture | Configurable Options |
|---|---|---|---|
| **Draw** | ✍️ Draw | Pointing | Color palette · Brush size 2–30 px · Brush style |
| **Shape** | 🔷 Shapes | Both Fists to lock | Shape type · Fill color · Stroke color · Opacity |
| **Stamp** | 🎨 Stamps | Pointing | 12 emoji stamps · Size 20–120 px |
| **Erase** | ⬜ Erase | Pointing or Thumbs Up | Eraser radius 10–80 px |

Modes can be switched via toolbar tabs or keyboard shortcuts `1` through `4`.

---

## Canvas Export

Clicking **💾 SAVE** composites all visible canvas layers into a single merged canvas displayed in a preview modal. The modal shows stroke count, point count, and canvas dimensions. The result can be downloaded as a PNG file via a generated `<a download>` link.

---

## Quick Start

> **Prerequisites:** A Google account and a webcam. No local installation required.

**Step 1 — Open the notebook**

Upload `CV_GESTURE_SYSTEM.ipynb` to [Google Colab](https://colab.research.google.com) or open it directly from your Drive.

**Step 2 — Run Cell 1**

Confirms that no pip installs are needed. MediaPipe is loaded entirely from CDN at runtime.

```python
# CELL 1
print('✅ Ready! Run Cell 2 to launch the zero-lag interface.')
```

**Step 3 — Run Cell 2**

Starts the Python HTTP server on port `8765` in a background thread and prints the Colab tunnel URL.

```python
import os, threading, http.server, socketserver
from google.colab.output import eval_js
# Server starts → tunnel URL printed → open in new browser tab
```

**Step 4 — Allow camera access** in the new browser tab, then click **▶ START**.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` | Switch to Draw mode |
| `2` | Switch to Shape mode |
| `3` | Switch to Stamp mode |
| `4` | Switch to Erase mode |
| `Ctrl + Z` | Undo last stroke |
| `Space` | Clear canvas |

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand tracking | MediaPipe Hands via jsDelivr CDN |
| Rendering | HTML5 Canvas API — 4-layer composite |
| Gesture classification | Vanilla JavaScript — rule-based + Euclidean landmark matching |
| HTTP server | Python `http.server` + `threading.Thread` |
| Colab integration | `google.colab.output.eval_js` |
| Fonts | Rajdhani + Share Tech Mono — Google Fonts |
| Gesture persistence | Browser `localStorage` |
| External dependencies | None |

---

## Tips for Best Results

- **Lighting:** Use a well-lit environment. Natural or front-facing light produces the most stable landmark detection.
- **Background:** A plain, low-contrast background behind your hand significantly improves tracking accuracy.
- **Distance:** Keep your hand 40–70 cm from the camera for optimal landmark resolution.
- **Both-Fist Lock:** Ensure both hands are fully visible in frame before clenching. Partial occlusion resets the countdown.
- **Custom Gestures:** Capture poses in the same lighting and at the same distance you plan to use them. Choose poses that are visually distinct from built-in gestures to avoid false positives.

---

## Roadmap

- [ ] Voice command layer alongside gesture control
- [ ] Multi-layer canvas with per-layer visibility and blending modes
- [ ] Air text mode — spell words letter by letter using hand positions
- [ ] WebSocket relay for collaborative multi-user canvas sessions
- [ ] Gesture macro recording and sequenced playback
- [ ] SVG export using landmark trajectory vectorization
- [ ] Standalone web app deployment — no Colab dependency

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c8ff,50:7c3aed,100:00ff88&height=100&section=footer"/>

**Built with MediaPipe · Python · Vanilla JavaScript**

*If this project was useful to you, consider giving it a ⭐*

</div>
