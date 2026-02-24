<div align="center">

```
 █████╗ ██╗     ██████╗ █████╗ ██████╗ ███████╗███████╗██████╗
██╔══██╗██║    ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗
███████║██║    ██║     ███████║██████╔╝█████╗  █████╗  ██████╔╝
██╔══██║██║    ██║     ██╔══██║██╔══██╗██╔══╝  ██╔══╝  ██╔══██╗
██║  ██║██║    ╚██████╗██║  ██║██║  ██║███████╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

████████╗██╗    ██╗██╗███╗   ██╗
╚══██╔══╝██║    ██║██║████╗  ██║
   ██║   ██║ █╗ ██║██║██╔██╗ ██║
   ██║   ██║███╗██║██║██║╚██╗██║
   ██║   ╚███╔███╔╝██║██║ ╚████║
   ╚═╝    ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝
```

### *Your AI-powered career intelligence system. Know your gaps. Ace the interview. Own your future.*

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourname/ai-career-twin/blob/main/AI_Career_Twin.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-2ECC71?style=for-the-badge)](LICENSE)
[![Free API](https://img.shields.io/badge/API-Free%20Tier-FF6B6B?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/app/apikey)

<br/>

> **The average job seeker applies to 100+ roles blindly.**
> **This system tells you exactly what's missing — before you apply.**

</div>

---

<br/>

## ◈ What Is This

**AI Career Twin** is a Google Colab notebook that acts as your personal career intelligence system. You feed it your resume and a job description. It does the rest.

In ~90 seconds, it produces:

- A **match score** showing how well you fit the role right now
- A **skill gap map** — exactly what you're missing, prioritized by how long each takes to learn
- A **phase-based learning roadmap** built around your specific gaps
- A **tailored interview question bank** with strong answer frameworks and red flags
- A **5-year career trajectory** with role progression and salary predictions
- A **downloadable HTML + JSON report** you can save and share

It does not give generic advice. Everything is computed against your actual resume and the actual job description you provide — powered by **Google Gemini 1.5 Flash**.

No local GPU. No paid API tier. No setup beyond opening a Colab tab.

<br/>

---

## ◈ The Six Engines

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   RESUME  ──────────────────────────────────────────────────►  │
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│   │  ENGINE 1    │   │  ENGINE 2    │   │    ENGINE 3      │  │
│   │  Resume      │──►│  JD Parser   │──►│   Gap Detector   │  │
│   │  Analyzer    │   │              │   │                  │  │
│   └──────────────┘   └──────────────┘   └──────────────────┘  │
│          │                                        │            │
│          ▼                                        ▼            │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│   │  ENGINE 6    │   │  ENGINE 5    │   │    ENGINE 4      │  │
│   │  Career      │◄──│  Interview   │◄──│   Roadmap        │  │
│   │  Trajectory  │   │  Simulator   │   │   Generator      │  │
│   └──────────────┘   └──────────────┘   └──────────────────┘  │
│                                                                 │
│   JD  ──────────────────────────────────────────────────────►  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Engine 1 — Resume Analyzer**
Extracts skills, experience, education, certifications, and inferred competencies. Uses a local NLP skill database (90+ skills across 6 categories) plus Gemini to catch what pattern-matching misses — implied skills, adjacent expertise, contextual knowledge.

**Engine 2 — JD Intelligence Parser**
Parses job descriptions into structured requirements: must-haves vs. nice-to-haves, seniority level, domain, key responsibilities. Works on any format — raw paste, PDF upload, or `.docx` file.

**Engine 3 — Gap Detector**
Computes exact set difference between your skills and the role's requirements. Outputs a match score (0–100%), readiness level, and a gap priority map with estimated learning time per skill — calibrated to your experience level.

**Engine 4 — Roadmap Generator**
Gemini builds a multi-phase learning plan around your specific gaps, your current experience level, and the target role's seniority. Each phase has concrete actions, free resources, and a milestone definition.

**Engine 5 — Interview Simulator**
Generates 8 tailored questions across 4 types — Technical, Behavioral, System Design, Situational — calibrated to the target role and your seniority level. Each question includes what to cover, a sample strong answer framework, and what a red-flag answer looks like.

**Engine 6 — Career Trajectory**
Predicts your 6-point career arc (Now → Year 5) with role titles, salary ranges in USD, readiness scores, and skills to add per year. Rendered as interactive charts and a visual milestone timeline.

<br/>

---

## ◈ Quick Start

**Step 1 — Get your free Gemini API key** (30 seconds)

```
https://aistudio.google.com/app/apikey
```

No credit card. No billing. The free tier handles this entire project.

**Step 2 — Open the notebook in Colab**

```
https://colab.research.google.com/github/yourname/ai-career-twin/blob/main/AI_Career_Twin.ipynb
```

Or upload `AI_Career_Twin.ipynb` manually at [colab.research.google.com](https://colab.research.google.com).

**Step 3 — Add your API key** (pick one method)

| Method | How |
|--------|-----|
| Paste directly | Open Cell 2, set `API_KEY = "your-key-here"` |
| Colab Secrets *(recommended)* | Left sidebar → 🔑 icon → add `GEMINI_API_KEY` |
| Environment variable | `os.environ['GEMINI_API_KEY'] = 'your-key'` before Cell 2 |

**Step 4 — Add your resume and job description**

Open **Cell 6** and paste into the two text fields:

```python
PASTE_RESUME = """
[paste your resume here]
"""

PASTE_JD = """
[paste the job description here]
"""
```

Or upload `.pdf`, `.docx`, or `.txt` files when prompted.

**Step 5 — Run everything**

```
Runtime → Run All   (or Ctrl+F9)
```

Done. Results appear in ~90 seconds.

> **No resume yet?** Leave the paste fields empty. Built-in demo data runs automatically so you can see the full output before adding your own content.

<br/>

---

## ◈ Cell-by-Cell Guide

| Cell | Name | What It Does |
|------|------|-------------|
| **1** | Install & Import | Installs 7 packages, imports all libraries, detects Colab environment |
| **2** | API Key | Configures Gemini — auto-detects from Secrets, env, or prompts you |
| **3** | Core Engine | Defines data models, 90-skill database, document parser, JSON utilities |
| **4** | AI Brain | All 6 Gemini analysis functions with fallback logic for every path |
| **5** | Viz Engine | Radar chart, gap bar chart, roadmap timeline, trajectory plots, interview cards |
| **6** | Input | Paste text, upload files (PDF/DOCX/TXT), or run on demo data |
| **7** | 🔴 Run Analysis | Executes all 6 engines in sequence with live progress reporting |
| **8** | Dashboard | Profile cards, match score, skill radar, gap priority chart |
| **9** | Roadmap + Trajectory | Phase-by-phase learning plan, 5-year role/salary forecast, milestone timeline |
| **10** | Interview Sim | Question bank with type/difficulty tags, answer frameworks, red flags |
| **11** | Export | Generates `career_report.html` + `career_report.json`, auto-downloads in Colab |

<br/>

---

## ◈ What the Output Looks Like

**Match Score Card**
```
         ╔════════════╗
         ║    67%     ║   ← your fit score
         ║   MATCH    ║
         ╚════════════╝
     🟡 Strong Candidate

MATCHING (14)    ✅ python  ✅ pytorch  ✅ aws  ✅ docker ...
CRITICAL GAPS (6) ❌ kubernetes  ❌ spark  ❌ mlflow  ❌ llm ...
```

**Gap Priority Chart**
```
kubernetes  ████████████████  8 weeks
spark       █████████████     5 weeks
llm         ████████          3 weeks
rag         ████████          3 weeks
mlflow      ████              1 week
```

**Learning Roadmap (example)**
```
PHASE 1 · Quick Wins Sprint · 4 weeks
  Skills: mlflow, rag
  ├─ Build experiment tracking with MLflow on an existing project
  ├─ Implement a basic RAG pipeline using LangChain + FAISS
  └─ 🎯 Milestone: Deploy a local RAG chatbot on your own data

PHASE 2 · Infrastructure Sprint · 6 weeks
  Skills: kubernetes, spark
  ├─ Complete Kubernetes the Hard Way (free)
  ├─ Process a 1GB+ dataset using PySpark locally
  └─ 🎯 Milestone: Deploy a containerized ML API to a K8s cluster
```

**Interview Question (example)**
```
[System Design] [Hard]

Q: Design a real-time ML feature store for a recommendation
   system serving 10M daily active users.

KEY POINTS: Data freshness vs. consistency trade-offs,
            online vs. offline store separation, serving latency SLAs

💡 STRONG ANSWER: Start with requirements clarification (freshness,
   scale, latency). Propose dual store (Redis online + Hive offline),
   explain write path and read path separately, then discuss trade-offs.

⚠️  RED FLAG: Jumping to implementation without scoping the problem.
```

<br/>

---

## ◈ Skill Coverage

The local skill database covers 90+ skills across 6 categories. Gemini additionally infers skills that are implied but not explicitly stated in your resume.

| Category | Examples |
|----------|---------|
| Languages | Python, Java, Go, Rust, TypeScript, SQL, Scala, R |
| Frameworks | PyTorch, TensorFlow, React, FastAPI, Spark, LangChain, Airflow |
| Cloud & DevOps | AWS, GCP, Azure, Docker, Kubernetes, Terraform, Helm |
| Databases | PostgreSQL, MongoDB, Redis, Elasticsearch, Snowflake, Pinecone |
| Concepts | LLM, RAG, fine-tuning, MLOps, system design, microservices, NLP |
| Soft Skills | Leadership, mentoring, technical writing, stakeholder management |

<br/>

---

## ◈ Reading Your Results

**Match Score**

| Score | Label | What It Means |
|-------|-------|--------------|
| 80–100% | 🟢 Highly Qualified | Apply now. Polish your pitch. |
| 60–79% | 🟡 Strong Candidate | Close 2–3 key gaps, then apply |
| 40–59% | 🟠 Needs Preparation | 1–2 months of focused study needed |
| 0–39% | 🔴 Significant Gap | 3–6 months to be competitive |

**Gap Learning Times**

Estimated at your experience level — someone with 4+ years learns Kubernetes in 4 weeks; a beginner takes 8. The roadmap accounts for this automatically.

**Career Trajectory**

The trajectory uses your current skills, experience, match score, and target role to project realistic role titles and salary bands for each year. Salary ranges are in USD, calibrated to the job's domain and seniority level.

<br/>

---

## ◈ Input Formats Supported

| Format | Resume | Job Description |
|--------|--------|----------------|
| Paste plain text | ✅ | ✅ |
| `.txt` file upload | ✅ | ✅ |
| `.pdf` file upload | ✅ | ✅ |
| `.docx` file upload | ✅ | ✅ |
| Built-in demo data | ✅ | ✅ |

For best results, paste plain text rather than uploading PDFs. Complex PDF layouts — multi-column, tables, heavy formatting — can lose structure during extraction.

<br/>

---

## ◈ Export Files

Cell 11 produces two files and auto-downloads them in Colab:

**`career_twin_report_YYYYMMDD_HHMMSS.html`**
A standalone dark-themed HTML dashboard with all results — score card, skill pills, roadmap, interview questions, and trajectory table. Opens in any browser. Share it with mentors, coaches, or peers without needing Python.

**`career_twin_report_YYYYMMDD_HHMMSS.json`**
The raw structured data from all 6 engines — skills, gaps, roadmap phases, interview questions, trajectory points. Use it to track progress over time, compare against future roles, or feed into other tools.

<br/>

---

## ◈ Dependencies

| Package | Purpose |
|---------|---------|
| `google-generativeai` | Gemini 1.5 Flash API — all 6 analysis engines |
| `PyPDF2` | PDF text extraction |
| `python-docx` | Word document parsing |
| `plotly` | Interactive charts — radar, bar, line, subplots |
| `pandas` | Data manipulation |
| `matplotlib` | Static figure support |
| `kaleido` | Plotly static image export |
| `ipywidgets` | Notebook UI widgets |

All installed automatically in Cell 1. Zero manual setup.

<br/>

---

## ◈ FAQ

**Do I need to pay for the Gemini API?**
No. Google's free tier includes 15 requests/minute and 1 million tokens/minute — more than enough to run this project dozens of times per day.

**How long does it take?**
60–90 seconds for a complete analysis. The delay is mostly API rate limiting between the 6 sequential Gemini calls. A 1.5-second buffer between calls is built in.

**Can I analyze multiple jobs against the same resume?**
Yes. Update `PASTE_JD` in Cell 6 and re-run Cells 6 through 11. Each run produces its own timestamped export file.

**Does it work outside Colab?**
Yes. Run `jupyter notebook AI_Career_Twin.ipynb` locally after installing the dependencies. File upload prompts use Colab widgets but the system auto-detects non-Colab environments and falls back gracefully.

**How accurate are the salary predictions?**
The trajectory is AI-generated based on your domain and seniority. Treat it as directional — validate with Glassdoor, Levels.fyi, or LinkedIn Salary for your specific market and location.

**What if Gemini returns broken JSON?**
Every Gemini call has a 3-layer JSON fallback parser (direct parse → markdown fence extraction → regex scan) plus a hardcoded fallback for each analysis type. The notebook will not crash on a malformed AI response.

**Is my resume data private?**
Your resume text is sent to Google's Gemini API for analysis. Review [Google's API data usage policy](https://ai.google.dev/terms) if you have concerns. Avoid including sensitive personal identifiers (SSN, home address, etc.) beyond what you'd put on a standard resume.

<br/>

---

## ◈ Known Limitations

| Limitation | Notes |
|-----------|-------|
| English only | Skill extraction and AI analysis are tuned for English-language resumes and JDs |
| Tech-focused skill DB | The local skill database covers software/data roles. Non-tech roles still work via Gemini but with less local precision |
| Gemini rate limits | Free tier: 15 req/min. The built-in 1.5s delay handles this; if you hit limits, Gemini raises a clear error |
| Salary estimates | USD-denominated, US-market calibrated by default |
| PDF parsing quality | Depends on the PDF's text layer. Scanned or image-based PDFs may extract poorly — use `.txt` export from your PDF reader instead |

<br/>

---

## ◈ License

MIT — use it, fork it, build on it.

---

<div align="center">

```
  ┌──────────────────────────────────────────────────────┐
  │   Stop guessing which jobs to apply to.              │
  │   Stop walking into interviews underprepared.        │
  │   Stop wondering why you didn't get the callback.    │
  │                                                      │
  │   Your career deserves a system.                     │
  └──────────────────────────────────────────────────────┘
```

**Built with Google Gemini · Plotly · Python**

*Know your gaps. Close them. Get the role.*

</div>
