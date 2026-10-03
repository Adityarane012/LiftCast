---
title: "Building LiftCast: A Local-First AI Workout Forecaster for My Friend Armaan"
published: false
description: "A 100% local workout logger that lets my friend Armaan log sets in 10 seconds using Gemma 4, forecasts progress via TabPFN, and detects strength stalls deterministically."
tags: hacktoberfest, gemma, tabpfn, github, python
canonical_url: https://github.com/Adityarane012/LiftCast
cover_image: https://raw.githubusercontent.com/Adityarane012/LiftCast/main/docs/assets/liftcast_hero.png
ai_disclosure_level: some_ai
---

## 1. What I Built

My friend **Armaan** is a college student who lifts regularly but has never logged a single session.

Every existing fitness app (Strong, Hevy, Liftoff) requires tedious mid-workout data entry: tap search, select variant, type weight, type reps, check off set, repeat 15 times. The cognitive friction while catching your breath is simply too high. Armaan logs nothing. Instead, he texts me afterward:
> *"bench 60 8 8 7, last set died"*  
> *"aaj lat pulldown 55 pe 10 10 9"*

Because he never logged, he couldn't answer the core question of progressive overload: **"Am I actually progressing, or have I hit a plateau?"**

I built **LiftCast**:
1. **10-Second Text Logger:** Armaan types the way he texts (shorthand, typos, Hinglish). A local **Gemma** model extracts structured sets via an enforced JSON schema.
2. **Deterministic Core:** Pure mathematical functions calculate Epley estimated 1-Rep Max (`weight * (1 + reps/30)`), top sets, and rolling 56-day least-squares slope stall detection.
3. **In-Context Tabular Forecaster:** **TabPFN** (running on CPU) forecasts next-session performance with 95% prediction intervals.
4. **Weekly AI Coach with Numeric Guard:** Gemma narrates weekly progress under a strict deterministic guard that audits every single number against the computed stats payload. Zero hallucinated statistics allowed.

---

## 2. Demo & User Interface

LiftCast features a modern dark glassmorphic Streamlit interface designed for quick interaction:

- **10-Second Text Logger:** Paste informal notes, hit *Parse with Gemma*, review the live confirmation table (with auto-computed e1RMs), and save to SQLite in one click.
- **Progress & Forecast Explorer:** Interactive Plotly charts showing actual session top-sets, 3-session rolling averages, TabPFN prediction intervals, and shaded red regions indicating detected stall windows.
- **Weekly Coach Recap:** Verified coach narrative accompanied by strength tier cards (Novice, Intermediate, Advanced) and time-to-tier range projections.
- **Benchmark & Architecture Explorer:** Full empirical backtest comparison and embedded interactive Archify system diagram.

---

## 3. Code Repository

- **GitHub:** [https://github.com/Adityarane012/LiftCast](https://github.com/Adityarane012/LiftCast)
- **License:** MIT
- **Architecture Diagram:** [docs/assets/liftcast_architecture.html](https://github.com/Adityarane012/LiftCast/blob/main/docs/assets/liftcast_architecture.html)

---

## 4. How I Built It

### The Architectural Separation: AI at the Edges, Math at the Core
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
```

### Local Hardware Profile
- **Laptop:** RTX 3050 Laptop GPU (4 GB VRAM) + Intel CPU.
- **VRAM Budget:** Gemma runs on GPU via Ollama (~2.2 GB VRAM). TabPFN runs strictly on CPU (`device='cpu'`) to prevent CUDA out-of-memory contention.
- **Inference Latency:** Gemma parses logs in 180–320 ms.

### The 40-Point Rolling-Origin Evaluation
We rejected arbitrary train/test splits that cause temporal leakage. Instead, we executed a rolling-origin backtest across the last 8 sessions for 5 key lifts (40 out-of-sample predictions):

| Lift | Test Sessions ($n$) | Last Value MAE (kg) | Trend-5 MAE (kg) | TabPFN MAE (kg) | Winner |
|---|---|---|---|---|---|
| **Lat Pulldown** | 8 | 6.48 | **6.21** | 10.84 | **Trend-5** |
| **Bench Press** | 8 | **8.78** | 11.08 | 12.28 | **Last Value** |
| **Deadlift** | 8 | **10.97** | 12.81 | 15.33 | **Last Value** |
| **Bent Over Row** | 8 | 12.94 | **10.33** | 15.04 | **Trend-5** |
| **Romanian Deadlift** | 8 | **6.87** | 8.44 | 18.07 | **Last Value** |
| **Overall** | **40** | **9.21** | **9.78** | **14.31** | **Last Value** |

**What the data actually told us:**
- On near-linear movements (Lat Pulldown, Row), the 5-session trend baseline wins.
- On noisy compounds (Bench, Deadlift), day-to-day session variance (2.8%–6.1%) penalizes trend extrapolation, making Last Value difficult to beat.
- TabPFN scales lifts as ratios to prior bests (`e1rm / best_so_far`), allowing cross-lift generalization and calibrated 95% uncertainty bands without retraining.

### Honest Failures & Postmortems
1. **The Flawed "PR in 21 Days" Stall Rule:** Our initial hypothesis was that lacking a PR within 3–4 sessions signified a stall. On real data, this rule flagged **22% to 78% of all workouts**! During a 6-month stretch where Bench rose from 42 kg to 64 kg, it flagged 61% of sessions as "stalled". We replaced it with a least-squares linear slope over 56 days with $m \ge 4$ sessions, which cleanly detects genuine plateaus without false positives.
2. **LLM Number Hallucinations:** Early prompt attempts had Gemma suggesting unprompted weights and supplement doses. We solved this with a strict architectural contract: code builds the JSON stats payload, Gemma narrates in $\le 120$ words, and a regex numeric guard audits every digit. If an unapproved number is detected, it falls back to a deterministic template.

---

## 5. Why Open Innovation Matters

- **100% Offline & Private:** Health notes, fatigue markers, and bodyweight never leave the user's laptop. Zero cloud egress.
- **Zero Cost (₹0):** No API subscription fees, rate limits, or vendor lock-in.
- **Enforced JSON Schemas:** Using Ollama's local structured decoding turns open models into reliable parsing compilers.

---

## 6. Prize Categories

### Best Use of Gemma
Gemma runs locally via Ollama (`gemma4:e2b`). It powers two core interfaces:
1. Converting noisy free-text logs (including shorthand and Hinglish) into structured sets via an enforced JSON schema.
2. Writing weekly coach summaries with an active numeric guard that verifies 100% of generated numbers against computed math.

### Best Use of TabPFN
TabPFN runs locally on CPU as an in-context tabular foundation model. By normalizing performance into ratio-space (`e1rm / best_so_far`), TabPFN transfers learned priors across exercises, outputting calibrated 95% prediction intervals evaluated honestly via rolling-origin backtesting.

### Best Use of GitHub
LiftCast leverages GitHub for rigorous open-source engineering and community participation:
- **Matrix CI Testing:** GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full 51-test suite across Python 3.11, 3.12, and 3.13 on every push and pull request.
- **Privacy & Quality Gates:** Custom PR template (`.github/PULL_REQUEST_TEMPLATE.md`) with explicit verification gates against data leakage and hallucinated numbers.
- **Open-Source Health:** Issue templates for bug tracking and feature requests, comprehensive `CONTRIBUTING.md`, and MIT License.

---

## 7. Armaan's Reaction (Verbatim)

> *"Wait, so I can just text 'bench 60 8 8 7' and I don't have to fiddle with menus while my arms are shaking? And it actually tells me whether I'm stuck or just having a bad day? That's insane. I'm actually going to use this."*
