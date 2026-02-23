<div align="center">

<!--  HEADER — split-panel terminal aesthetic, distinct from wave/cylinder styles  -->
<table width="100%" cellspacing="0" cellpadding="0" border="0">
<tr>
<td width="50%" align="center" bgcolor="#0a0f1e" style="padding:30px 20px;">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&pause=1000&color=00FF9F&center=true&vCenter=true&width=380&lines=AUDITABLE+AI+DIAGNOSTIC;BLOCKCHAIN+FAULT+CHAIN;GRAD-CAM+%2B+SHA-256+DLT;WHO+IS+ACCOUNTABLE%3F" alt="Typing SVG"/>
</td>
<td width="50%" align="center" bgcolor="#0d1b2a" style="padding:30px 20px;">
<img src="https://capsule-render.vercel.app/api?type=slice&color=0:00ff9f,100:0d1b2a&height=120&text=BCF+SYSTEM&fontSize=36&fontColor=ffffff&fontAlignY=70&rotate=0"/>
</td>
</tr>
</table>

<br/>

<img src="https://img.shields.io/badge/Model-ResNet50_%2B_Grad--CAM-00ff9f?style=for-the-badge&logo=pytorch&logoColor=black"/>
<img src="https://img.shields.io/badge/Ledger-SHA--256_Blockchain_DLT-0d1b2a?style=for-the-badge&logo=ethereum&logoColor=00ff9f"/>
<img src="https://img.shields.io/badge/Domain-Skin_Cancer_%7C_ISIC_2020-1a1a2e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/CFA-Causality_Fault_Engine-ff6b35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Platform-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black"/>

<br/><br/>

```
╔═══════════════════════════════════════════════════════════════╗
║  "When AI makes a life-or-death decision — who is liable?"   ║
║         This system answers with cryptographic proof.         ║
╚═══════════════════════════════════════════════════════════════╝
```

</div>

---

## Overview

The **Auditable AI Diagnostic Pipeline** (BCF System) is a clinical AI governance framework built around a single question that current medical AI deployments cannot answer: **when an AI-assisted diagnosis is wrong, who is legally and ethically responsible — the model vendor, the treating clinician, or the deploying institution?**

The system wraps a ResNet50-based skin cancer classifier around three interlocking accountability components:

**Grad-CAM Explainability** generates a pixel-level heatmap for every inference, spatially highlighting the image regions that drove the prediction. This visual justification is presented to the doctor before any accept/override decision is recorded, satisfying the human oversight requirement for high-risk AI systems.

**Blockchain Fault Chain (BCF)** records every AI prediction, confidence score, doctor action, model version, and UTC timestamp as a SHA-256 hash-linked immutable block. No log entry can be silently altered after the fact — any modification is cryptographically detectable. Every event simultaneously persists to a human-readable CSV audit log.

**Causality Framework Algorithm (CFA)** cross-references the logged confidence score against a governance-defined trust threshold and the doctor's recorded action to deterministically assign fault across four possible verdicts: Success, AI System's Fault, Doctor's Fault, or Uncertainty / Data Problem.

This is not a standard image classification project. It is a working prototype of the accountability architecture that clinical AI systems must adopt to comply with the EU AI Act (Article 13 transparency obligations), FDA Software as a Medical Device (SaMD) guidance, and emerging hospital AI governance standards.

---

## Research Objective

> **Can a lightweight accountability wrapper — combining cryptographic immutable logging, visual explainability, and deterministic causality analysis — provide a complete, legally defensible audit trail sufficient to assign responsibility for errors in AI-assisted clinical diagnosis?**

---

## Full System Architecture

The notebook is split across two cells, each an independently executable module:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                   CELL 0 — LIVE DIAGNOSTIC PIPELINE                     ║
╚══════════════════════════════════════════════════════════════════════════╝

  [Step 0–2]  Install deps · import libraries · create uploads/ · init CSV
                                     │
                                     ▼
  [Step 3]  ResNet50  (pretrained=True, eval mode)
            Register hooks on model.layer4[-1].conv3
            ├── forward_hook  → captures feature maps on forward pass
            └── backward_hook → captures gradients on backward pass
                                     │
                                     ▼
  [Step 5]  Image Upload  (google.colab.files.upload)
            │
            ├── PIL.Image.open → convert RGB
            ├── transforms: Resize(224×224) → ToTensor
            │               Normalize(mean=[0.485,0.456,0.406],
            │                         std =[0.229,0.224,0.225])
            └── input_tensor = preprocessed.unsqueeze(0)
                                     │
                                     ▼
            ResNet50 forward pass
            F.softmax(output) → conf, pred_idx
                                     │
                                     ▼
  ┌──────────────── GRAD-CAM ────────────────────────────────────────────┐
  │  output[0, pred_idx].backward(retain_graph=True)                     │
  │  pooled_grads = mean(gradients, dim=[0,2,3])          → shape [C]   │
  │  features[0, i, :, :] *= pooled_grads[i]  for i in C               │
  │  heatmap = mean(features[0], dim=0)                   → shape [H,W] │
  │  ReLU(heatmap) → normalize [0,1] → uint8 [0,255]                    │
  │  PIL resize to original image dims (bilinear)                        │
  │  jet colormap → overlay = 0.5×original + 0.5×colored                │
  │  plt.figure(12×5) → subplot[1] original | subplot[2] Grad-CAM       │
  └──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  Doctor views: Original image + Grad-CAM heatmap side-by-side
                                     │
                                     ▼
  [Step 6]  Doctor Input (interactive)
            ├── decision    : "accept" or "reject"
            ├── reason      : free-text clinical note
            └── ground_truth: biopsy result / known label
                                     │
                                     ▼
  [Step 7]  BCF Log Entry Assembly
            record_str = filename + pred_idx + confidence +
                         decision + reason + ground_truth + utcnow
            record_hash = SHA-256(record_str)
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
  [Step 8]  CSV Persist              [Step 9]  Blockchain Persist
  bcf_accountability_log.csv         Blockchain.add_log_entry(log_data)
  (human-readable, queryable)        (SHA-256 tamper-evident chain)
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                        is_chain_valid() → integrity check
                        print → df.tail(5) display


╔══════════════════════════════════════════════════════════════════════════╗
║                CELL 1 — BCF CAUSALITY SIMULATION ENGINE                 ║
╚══════════════════════════════════════════════════════════════════════════╝

  CONFIG: AI_SYSTEM_NAME = "Skin Cancer Detection AI (ISIC 2020)"
          AI_CONFIDENCE_THRESHOLD = 80.0%
                                     │
                                     ▼
  Blockchain() → Genesis Block (index 0)
  {"note": "System Started for: Skin Cancer Detection AI ..."}
  previous_hash = "0"
                                     │
              ┌───────────────────────────────────┐
              │  For each of 3 simulation cases:  │
              │  add_log_entry(payload dict)       │
              │  generate_audit_report(ledger, i)  │
              └───────────────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
              causality_framework_          Section A/B/C
              algorithm(log, outcome,       structured print
              threshold=80.0)
                                     │
                                     ▼
              Final is_chain_valid() across all blocks
```

---

## Component 1 — ResNet50 + Grad-CAM Explainability

### Model Specification

| Property | Value |
|---|---|
| Architecture | ResNet50 — 50-layer deep residual network |
| Weights | ImageNet pretrained via `torchvision.models.resnet50(pretrained=True)` |
| Runtime mode | Inference only — `model.eval()` |
| Input resolution | `224 × 224 × 3` |
| Normalization | ImageNet statistics · mean `[0.485, 0.456, 0.406]` · std `[0.229, 0.224, 0.225]` |
| Output | Softmax distribution across 1,000 ImageNet classes |
| Clinical target | Skin Cancer Detection mapped to ISIC 2020 malignant/benign labelling |

### Grad-CAM — Mechanism

Gradient-weighted Class Activation Mapping (Grad-CAM) produces a coarse spatial heatmap that identifies the image regions most responsible for the model's predicted class. It works by weighting each spatial activation map in the final convolutional layer by the gradient of the predicted class score with respect to that map, then collapsing across channels.

**Why `model.layer4[-1].conv3`?** This is the last convolutional layer before global average pooling. It is the optimal hook point because it carries both (a) the highest-level semantic representations the network has learned, and (b) the highest spatial resolution available before spatial information is permanently collapsed. Hooks placed after GAP produce non-spatial attributions that cannot be meaningfully overlaid on an input image.

```python
# Hook registration
last_conv_layer = model.layer4[-1].conv3
last_conv_layer.register_forward_hook(forward_hook)   # → stores features
last_conv_layer.register_backward_hook(backward_hook) # → stores gradients

# Grad-CAM computation
output[0, pred_idx].backward(retain_graph=True)

pooled_grads = torch.mean(gradients, dim=[0, 2, 3])   # global avg pool over H, W → [C]
for i in range(features.shape[1]):
    features[0, i, :, :] *= pooled_grads[i]           # weight each activation map

heatmap = torch.mean(features[0], dim=0)              # collapse channels → [H, W]
heatmap = np.maximum(heatmap, 0)                      # ReLU — keep only positive influence
heatmap /= torch.max(heatmap)                         # normalize to [0, 1]
heatmap  = np.uint8(255 * heatmap.numpy())            # scale to uint8

# Overlay assembly
heatmap  = Image.fromarray(heatmap).resize(img.size)  # upsample to original resolution
colored  = np.uint8(255 * cmap(heatmap_np/255.0)[:,:,:3])   # jet colormap
overlay  = np.uint8(np.array(img) * 0.5 + colored * 0.5)    # 50/50 alpha blend
```

The doctor is shown the original image and the Grad-CAM overlay **before** entering their decision. This is not cosmetic — it is the explainability gate that ensures no accept/override decision is logged without visual model justification having been presented.

---

## Component 2 — Blockchain Fault Chain (BCF) Immutable Ledger

### Block Data Structure

```python
Block {
    index         : int        # sequential position in chain
    timestamp     : str        # UTC  '%Y-%m-%d %H:%M:%S'
    data          : dict       # BCF log payload
    previous_hash : str        # SHA-256 of preceding block
    hash          : str        # SHA-256 of json.dumps(self, sort_keys=True)
}
```

The block hash is computed over a canonical JSON string of all four fields using `json.dumps(..., sort_keys=True)`. `sort_keys=True` is not optional — without it, Python dict serialization can vary between environments and Python versions, producing different hashes for semantically identical blocks and breaking chain validation.

### Chain Integrity Protocol

```python
def is_chain_valid():
    for i in range(1, len(chain)):
        current  = chain[i]
        previous = chain[i-1]

        # Check 1: content integrity — has this block been modified?
        if current.hash != current.compute_hash():
            return False, f"Record Tampered at Block {current.index}"

        # Check 2: chain integrity — has a block been inserted, removed, or reordered?
        if current.previous_hash != previous.hash:
            return False, f"Link Broken at Block {current.index}"

    return True, "All Records are Secure and Untouched"
```

The double verification catches two distinct attack classes. A single hash recompute check detects content modification but not block removal. A single chain link check detects structural attacks but not in-place edits where the attacker also recomputes the modified block's hash. Both checks together are necessary and sufficient for tamper detection in a single-party ledger.

### BCF Log Payload Schema

| Field | Type | Description |
|---|---|---|
| `Hashed_PatientID` | `str` | De-identified patient reference |
| `Malignancy_Confidence` | `float` | AI model softmax confidence (%) |
| `AI_Prediction_Class` | `str` | `"Malignant"` or `"Benign"` |
| `Time_Taken_ms` | `int` | Model inference wall-clock latency |
| `Doctor_Action` | `str` | `"Confirm"` or `"Override"` |
| `Override_Reason` | `str` | Clinician's stated justification (override cases only) |
| `Model_Version` | `str` | Deployed model version — e.g. `v2.5.0` |
| `Doctor_ID` | `str` | Treating clinician identifier |

**Genesis Block (index 0):** Anchors the chain with `previous_hash = "0"` and records system initialization with `AI_SYSTEM_NAME`. Every subsequent diagnostic event extends from this anchor.

### CSV Audit Log — Dual Persistence

Module 1 simultaneously persists to a flat file `bcf_accountability_log.csv` with columns: `filename`, `prediction`, `confidence`, `doctor_decision`, `doctor_reason`, `ground_truth`, `timestamp`, `hash`.

The record-level hash in the CSV is computed from raw clinical data fields:
```python
record_str  = f"{filename}{pred_idx}{conf}{decision}{reason}{ground_truth}{utcnow}"
record_hash = hashlib.sha256(record_str.encode()).hexdigest()
```

CSV provides human-readable, filterable, exportable records for routine compliance workflows. The blockchain provides cryptographic tamper-evidence that makes those records legally defensible. Neither is sufficient alone — CSV can be silently edited; blockchain without human-readable output is operationally opaque.

---

## Component 3 — Causality Framework Algorithm (CFA)

The CFA is the accountability reasoning engine. It accepts a completed log entry from the blockchain, the clinical outcome, and the governance threshold, then traverses a deterministic decision tree to assign fault to the responsible party.

**Trust threshold:** `AI_CONFIDENCE_THRESHOLD = 80.0%`

This threshold is the system's binary definition of "confident enough to be trusted." When AI confidence is at or above 80%, the AI's recommendation is considered a reliable signal and responsibility shifts based on what the doctor chose to do with it. Below 80%, the AI is operating outside its reliable performance band and the error is classified as a data/training deficiency.

### Fault Decision Tree

```
                       Diagnosis Correct?
                              │
               ┌──────────────┴──────────────┐
              YES                            NO
               │                              │
          [ SUCCESS ]           AI Confidence ≥ 80.0% ?
                                              │
                               ┌──────────────┴──────────────┐
                              YES                            NO
                               │                              │
                      Doctor_Action ?              [ UNCERTAINTY /
                               │                    DATA PROBLEM ]
                ┌──────────────┴──────────────┐   Hospital must
           "Override"                    "Confirm"   retrain model
                │                              │
       [ DOCTOR'S FAULT ]        [ AI SYSTEM'S FAULT ]
       Human oversight failure    Algorithm / vendor error
```

### Verdict Reference

| Verdict | Conditions | Liable Party | Clinical Interpretation |
|---|---|---|---|
| `SUCCESS` | `Diagnosis_Correct = True` | None | Pipeline worked as intended |
| `AI SYSTEM'S FAULT` | `conf ≥ 80%` · `action = Confirm` · wrong | AI vendor | AI was confident and wrong; doctor reasonably trusted it |
| `DOCTOR'S FAULT` | `conf ≥ 80%` · `action = Override` · wrong | Clinician | Doctor ignored a confident correct AI signal |
| `UNCERTAINTY / DATA PROBLEM` | `conf < 80%` · wrong | Hospital / institution | Edge case outside training distribution; retraining required |

### Audit Report Output Format

`generate_audit_report()` produces a three-section structured report designed to be readable by hospital administrators, legal counsel, ethics boards, and regulatory auditors — not only ML engineers:

```
══════════════════════════════════════════════════
--- FINAL AUDIT REPORT: Patient ID SKIN_PXXXX ---
══════════════════════════════════════════════════

[SECTION A: WHO IS RESPONSIBLE?]
  FINAL JUDGMENT : <verdict>
  REASON         : <plain-English justification>

[SECTION B: IMMUTABLE EVIDENCE (The Facts)]
  Log Time       : YYYY-MM-DD HH:MM:SS
  AI Score       : XX.X%
  AI Prediction  : Malignant / Benign
  Diagnosis Time : XXX milliseconds
  Doctor's Choice: Confirm / Override
  Doctor's Note  : <override_reason if present>
  Actual Outcome : CORRECT  or  WRONG (Error Type)

[SECTION C: RECORD CHECK]
  Record Security: SECURE / TAMPERED — <chain validation message>
──────────────────────────────────────────────────
```

---

## Simulation Scenarios

Module 2 runs three pre-defined clinical scenarios against a live BCF ledger, covering every non-trivial path through the CFA decision tree:

### Simulation 1 — AI System's Fault
```
Patient          : SKIN_P1001
AI Confidence    : 92.5%   [above 80% threshold]
AI Prediction    : Malignant
Inference Time   : 150 ms
Doctor Action    : Confirm
Model Version    : v2.5.0   Dr. Smith
Actual Outcome   : WRONG — False Positive (Unnecessary Surgery Risk)

Verdict          : AI SYSTEM'S FAULT
Reason           : AI was highly confident (92.5%) but incorrect.
                   Doctor reasonably trusted the system.
                   AI model vendor bears responsibility.
```

### Simulation 2 — Doctor's Fault
```
Patient          : SKIN_P1002
AI Confidence    : 85.0%   [above 80% threshold]
AI Prediction    : Malignant
Inference Time   : 180 ms
Doctor Action    : Override
Override Note    : "Non-pigmented lesion look; thought it was benign."
Model Version    : v2.5.0   Dr. Jones
Actual Outcome   : WRONG — False Negative (Delayed Cancer Treatment)

Verdict          : DOCTOR'S FAULT
Reason           : Doctor ignored a confident (85.0%) and correct AI signal.
                   Dr. Jones bears responsibility for the clinical override.
```

### Simulation 3 — Success
```
Patient          : SKIN_P1003
AI Confidence    : 98.0%
AI Prediction    : Benign
Inference Time   : 120 ms
Doctor Action    : Confirm
Model Version    : v2.5.0   Dr. Allen
Actual Outcome   : CORRECT

Verdict          : SUCCESS
Reason           : System and doctor worked perfectly together.
```

**Final security check** after all three cases: `is_chain_valid()` traverses every block and confirms full cryptographic integrity across the chain.

---

## Regulatory and Governance Context

This system directly addresses the accountability gap in currently deployed medical AI. Most clinical AI products generate predictions without creating any structured, tamper-proof evidence trail — making post-incident error attribution legally ambiguous and technically impossible.

The BCF pipeline maps to three regulatory requirements specifically:

**EU AI Act — Article 13 (Transparency).** High-risk AI systems must be transparent, enabling those responsible for oversight to interpret the system's output. Grad-CAM satisfies the interpretability obligation; the structured audit log satisfies the transparency record-keeping obligation.

**FDA SaMD Guidance (Software as a Medical Device).** The FDA requires that AI/ML-based SaMD systems implement a Total Product Lifecycle (TPLC) approach including performance monitoring and clear documentation of human-AI decision boundaries. The BCF ledger creates the evidence base for lifecycle auditing; the CFA explicitly defines the human-AI decision boundary via the trust threshold.

**Clinical Negligence Standard of Proof.** For a medical negligence claim involving AI to succeed or fail, courts need evidence of: (a) what the AI predicted and with what confidence, (b) what the clinician knew and chose, and (c) what the actual outcome was. The three-section BCF audit report provides exactly this evidence in a tamper-proof, timestamped, cryptographically signed form.

---

## Quick Start

> **Requirements:** Google account · a dermatology/skin lesion image for Module 1 testing.

**Step 1 — Open in Colab**

Upload `FINAL_DL_PROJECT.ipynb` to [Google Colab](https://colab.research.google.com). GPU runtime is optional — single-image ResNet50 inference runs acceptably on Colab's free CPU tier.

**Step 2 — Run Cell 0 (Live Diagnostic Pipeline)**

```python
!pip install torch torchvision pillow matplotlib pandas
```

Upload a skin lesion image when prompted. Review the Grad-CAM overlay, then enter doctor decision, optional reason, and ground truth biopsy result. The entry is SHA-256 hashed and written to both `bcf_accountability_log.csv` and the in-session blockchain.

**Step 3 — Run Cell 1 (Causality Simulation Engine)**

No user input required. `run_bcf_simulation()` executes automatically, printing three structured audit reports followed by a final chain integrity verification.

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` | ResNet50 forward and backward pass · gradient computation |
| `torchvision` | Pretrained ResNet50 weights · image preprocessing transforms |
| `Pillow` | Image loading · format conversion · Grad-CAM overlay compositing |
| `matplotlib` | Side-by-side original + Grad-CAM display · jet colormap |
| `numpy` | Heatmap array operations · uint8 scaling · alpha blending |
| `pandas` | CSV audit log creation, row append, and tail display |
| `hashlib` | SHA-256 block hashing and per-record fingerprinting |
| `json` | Canonical block serialization with `sort_keys=True` |
| `datetime` / `time` | UTC timestamps for blocks and CSV log entries |

---

## Project Structure

```
FINAL_DL_PROJECT.ipynb
│
├── Cell 0  ──  MODULE 1: Live Diagnostic Pipeline
│   │
│   ├── Step 0    pip install torch torchvision pillow matplotlib pandas
│   ├── Step 1    Library imports
│   ├── Step 2    Create uploads/ · initialize bcf_accountability_log.csv
│   ├── Step 3    Load ResNet50 (pretrained, eval)
│   │             Register forward_hook + backward_hook on layer4[-1].conv3
│   ├── Step 4    Block class · Blockchain class  (SHA-256 linked ledger)
│   ├── Step 5    Upload image → preprocess → forward pass → Grad-CAM → display
│   ├── Step 6    Doctor decision input  (decision / reason / ground_truth)
│   ├── Step 7    BCF log assembly + SHA-256 record hash
│   ├── Step 8    CSV persistence  (bcf_accountability_log.csv)
│   └── Step 9    Blockchain append → is_chain_valid() → df.tail(5)
│
└── Cell 1  ──  MODULE 2: BCF Causality Simulation Engine
    │
    ├── Config    AI_SYSTEM_NAME · AI_CONFIDENCE_THRESHOLD = 80.0%
    ├── Block     index · timestamp · data · previous_hash · hash
    │             to_dict() for deterministic serialization
    ├── Blockchain  genesis block · add_log_entry() · is_chain_valid()
    ├── CFA       causality_framework_algorithm(log, outcome, threshold)
    │             → 4-verdict deterministic fault tree
    ├── Report    generate_audit_report()  → Section A / B / C
    └── Sim       run_bcf_simulation()
                  ├── Case 1: AI Fault   (92.5% · Confirm · False Positive)
                  ├── Case 2: Dr. Fault  (85.0% · Override · False Negative)
                  ├── Case 3: Success    (98.0% · Confirm · Correct)
                  └── Final is_chain_valid() sweep
```

---

## Key Design Decisions

**`layer4[-1].conv3` as the Grad-CAM hook target.** The final convolutional layer before global average pooling is the optimal trade-off between semantic richness and spatial resolution. Earlier layers retain spatial granularity but carry low-level features (edges, textures) that are not clinically meaningful. The GAP layer and beyond collapse all spatial information into a single vector, making overlay visualization impossible. `layer4[-1].conv3` is the last moment where both properties coexist.

**`json.dumps(..., sort_keys=True)` for block hashing.** Python dictionaries are insertion-ordered in Python 3.7+ but this ordering is an implementation detail that may not hold across all serialization paths, pickle versions, or future runtime changes. Without `sort_keys=True`, two environments could produce different SHA-256 hashes for an identical block and `is_chain_valid()` would incorrectly report tampering. Canonical serialization is a correctness requirement, not an optimization.

**Double-check integrity protocol in `is_chain_valid()`.** Recomputing each block's hash detects content modification but not structural attacks (block removal, duplication, reordering) — because a removed block simply disappears from the chain without leaving a mismatched hash behind. Checking `previous_hash` linkage independently detects structural attacks. Both checks together are necessary and sufficient.

**`AI_CONFIDENCE_THRESHOLD = 80.0` as a named governance constant.** The threshold is not a magic number buried in a conditional — it is a module-level constant with an explicit comment that it was updated (from 75% to 80%). In production, this value represents a versioned governance policy reviewed and signed by hospital ethics boards, AI vendors, and regulators. Its location as a named constant signals that changing it requires a deliberate governance decision, because changing it directly changes who is found at fault in audit outcomes.

**Plain-English Section A justifications.** The CFA returns human-readable verdict strings (`"The doctor ignored a confident AI..."`), not machine codes. Audit reports generated by this system are consumed by administrators, lawyers, ethics committees, and regulators — not ML engineers. Readable justifications are a design requirement, not a convenience.

**Dual CSV + blockchain persistence.** A CSV alone is legally indefensible — it can be opened in Excel and edited without trace. A blockchain alone is operationally inaccessible — querying it requires code. The dual architecture provides human accessibility via CSV for routine governance work and cryptographic tamper-evidence via the blockchain for legal and regulatory proceedings.

---

## Limitations and Future Work

**ResNet50 uses ImageNet weights, not ISIC fine-tuning.** Module 1 demonstrates the accountability pipeline architecture with a pretrained backbone. Clinical deployment requires replacing the 1,000-class ImageNet head with a binary malignant/benign classification head and fine-tuning on the full ISIC 2020 labeled dataset.

**In-memory, single-party blockchain.** The current ledger is a Python object with no persistence across sessions and no distributed consensus. A production system requires a permissioned blockchain (Hyperledger Fabric, Quorum, or a private Ethereum network) shared across the hospital, AI vendor, and regulatory authority — ensuring that no single party can rewrite history.

**Binary fault attribution.** The CFA assigns 100% of fault to one party. Real clinical negligence typically involves shared and partial responsibility — the AI may have been poorly calibrated while the doctor may have had inadequate training on AI outputs. A probabilistic fault distribution weighted by confidence margin, override history, and model drift metrics would be more legally realistic.

**No model performance drift detection.** Model version `v2.5.0` is logged as a static string. A production system requires continuous monitoring of confidence calibration and false-positive/negative rates over rolling time windows, with automatic escalation when statistical drift is detected, triggering model review and retraining before further deployment.

**Patient ID tokenization.** `Hashed_PatientID` in the simulation is a plain string. HIPAA and GDPR-compliant deployment requires keyed HMAC tokenization managed by the hospital identity system, with the patient mapping stored separately and never written to the audit ledger itself.

---

## License

[MIT License](LICENSE) — free to use, extend, and build upon.

---

<div align="center">

```
┌──────────────────────────────────────────────────────────────┐
│  PyTorch · Grad-CAM · SHA-256 DLT · Causality Framework     │
│       Accountability infrastructure for clinical AI          │
└──────────────────────────────────────────────────────────────┘
```

*If this project was useful, consider giving it a ⭐*

</div>
