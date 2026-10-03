# ⚡ LiftCast

> **A local-first AI workout logger and progress forecaster, built for my friend Armaan.**  
> *Entry for the DEV Hacktoberfest Weekend Challenge: "Build for a Friend"*

[![CI](https://img.shields.io/github/actions/workflow/status/Adityarane012/LiftCast/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/Adityarane012/LiftCast/actions)
[![Tests: Passing](https://img.shields.io/badge/pytest-51%20passed-10b981?style=for-the-badge&logo=pytest)](tests/)
[![Python: 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/Adityarane012/LiftCast)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Privacy: 100% Local](https://img.shields.io/badge/Privacy-100%25%20Offline%20%2F%20No%20Cloud-8b5cf6?style=for-the-badge)](CLAUDE.md)

---

## 🎯 The Problem

My friend **Armaan** is a regular lifter who has never logged a single session. Every existing logging app (Strong, Hevy, Liftoff) requires painful mid-set structured data entry (search exercise, select variant, enter weight, enter reps, tap checkmark).

The friction is too high, so Armaan logs nothing. Instead, he texts me after a workout:
> *"bench 60 8 8 7, last set died"*  
> *"aaj lat pulldown 55 pe 10 10 9"*

Because he never logs, he has no answer to the most important question in strength training:  
**"Am I actually progressing on this lift, or have I hit a plateau?"**

**LiftCast** solves this:
1. **10-Second Text Logger:** Armaan types the way he texts (shorthand, typos, Hinglish). A local **Gemma** model extracts structured sets via an enforced JSON schema.
2. **Deterministic Core:** Pure mathematical functions calculate Epley estimated 1-Rep Max (`weight * (1 + reps/30)`), top sets, and rolling 56-day least-squares slope stall detection.
3. **In-Context Tabular Forecaster:** **TabPFN** (running locally on CPU) forecasts next-session performance with 95% prediction intervals.
4. **Visual Barbell Plate Loader:** Inverts the TabPFN e1RM forecast into an actionable working weight snapped to 2.5 kg plates, complete with a visual barbell sleeve graphic showing the exact 20kg/10kg/5kg/2.5kg/1.25kg plates per side.
5. **Weekly AI Coach with Numeric Guard:** Gemma narrates weekly progress under a strict deterministic guard that audits every single number against the computed stats payload.
6. **1-Click Demo Seeder:** Reviewers and fresh clones can instantly seed a 6-month rich history across 7 lifts with genuine plateaus and progressions.

---

## 🏆 Prize Categories

| Category | Implementation & Meaningful Use |
|---|---|
| **Best Use of Gemma** | Runs locally via Ollama (`gemma4:e2b` / `gemma4:e4b`). Uses Ollama's structured JSON schema to parse free-form shorthand and Hinglish logs. Generates weekly coach recaps with an enforced regex numeric guard against hallucinated stats. |
| **Best Use of TabPFN** | Runs locally on CPU. In-context prior predicts next-session top-set e1RM scaled as a ratio to historical best across disparate lifts, providing calibrated 95% uncertainty intervals evaluated via a 40-point rolling-origin backtest. |
| **Best Use of GitHub** | Comprehensive CI matrix testing across Python 3.11, 3.12, and 3.13 on GitHub Actions. Automated verification of 51 test suites, PR templates with strict data privacy auditing, issue templates, and open-source contribution governance. |

---

## 🏛️ System Architecture

LiftCast strictly follows the principle: **AI at the edges, deterministic code at the core.**

```
       Free text note ("bench 60 8 8 7, last set died")
                           │
                           ▼
                  parser.py (Local Gemma)
                     [Enforced JSON Schema]
                           │
                           ▼
                  db.py (Local SQLite)
                 [Sessions & Sets Schema]
                           │
                           ▼
                  metrics.py (Pure Math)
             [Epley e1RM, Leakage-Free Features]
               ┌───────────┴───────────┐
               ▼                       ▼
          forecast.py              detect.py
        (Local TabPFN)      (56-Day Slope Stall Rule)
               └───────────┬───────────┘
                           ▼
                  coach.py (Gemma Narrator)
                  [Strict Numeric Guard]
                           │
                           ▼
                  app.py (Streamlit UI)
            [Dark Glassmorphic Visualization]
```

### 🗺️ Interactive Archify Diagram
An interactive, standalone visual architecture diagram with pan, zoom, and inspectable privacy boundaries is available at [`docs/assets/liftcast_architecture.html`](docs/assets/liftcast_architecture.html) (and embedded directly in Page 4 of the Streamlit app).

---

## 📊 40-Point Rolling-Origin Evaluation (§7.4)

Evaluated across the last 8 sessions of the 5 core lifts (40 out-of-sample predictions). Context strictly excludes data on or after the target date:

| Lift | Test Points ($n$) | Last Value MAE (kg) | Trend-5 MAE (kg) | TabPFN MAE (kg) | Winner |
|---|---|---|---|---|---|
| **Lat Pulldown** | 8 | 6.48 | **6.21** | 10.84 | **Trend-5** |
| **Bench Press** | 8 | **8.78** | 11.08 | 12.28 | **Last Value** |
| **Deadlift** | 8 | **10.97** | 12.81 | 15.33 | **Last Value** |
| **Bent Over Row** | 8 | 12.94 | **10.33** | 15.04 | **Trend-5** |
| **Romanian Deadlift** | 8 | **6.87** | 8.44 | 18.07 | **Last Value** |
| **Overall** | **40** | **9.21** | **9.78** | **14.31** | **Last Value** |

*Findings:*
- The **Trend-5** baseline wins on smooth, linear progress phases (Lat Pulldown and Bent Over Row).
- The **Last Value** baseline is exceptionally difficult to beat on compound movements with session noise (Bench Press and Deadlift).
- **TabPFN** provides zero-shot ratio predictions across lifts without retraining, alongside calibrated uncertainty bounds.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python:** 3.11+ (tested on Python 3.13)
- **Ollama:** Installed and running locally with Gemma:
  ```bash
  ollama pull gemma4:e2b
  ```

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Adityarane012/LiftCast.git
cd LiftCast

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# or: source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Modern Streamlit UI
```bash
$env:PYTHONPATH="src"
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the Pytest Test Suite
```bash
$env:PYTHONPATH="src"
pytest -q
```
All **51 tests** should pass in ~8 seconds.

---

## 🔒 Privacy & Local-First Guarantees

1. **Zero Cloud Egress:** All models (Gemma via Ollama, TabPFN via PyTorch CPU) run 100% locally on your machine.
2. **Sanitized Storage:** Liftoff CSV imports automatically strip `Workout Name` and personal `Notes` before writing to SQLite.
3. **No Secret Invented Numbers:** Gemma is restricted from hallucinating numbers through a strict regex numeric guard and instant deterministic fallback.

---

## 📝 License & Hackathon Notice

Built for the **DEV Hacktoberfest 2026 Weekend Challenge: "Build for a Friend"**.  
MIT License.
