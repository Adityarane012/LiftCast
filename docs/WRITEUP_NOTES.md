# LiftCast — Dev.to Hacktoberfest Write-up Notes

Notes, verified numbers, empirical benchmarks, and honest failures for the submission article:
**"Building LiftCast: A Local-First AI Workout Forecaster for My Friend Armaan"**
Challenge: **DEV Hacktoberfest Weekend Challenge: Build for a Friend**
Target Categories: **Best Use of Gemma**, **Best Use of TabPFN**

---

## 1. The Story: Why We Built This

### The User
Armaan is a regular lifter who has never logged a single session. Every existing logging app (Strong, Hevy, Liftoff, MyFitnessPal) demands structured inputs mid-workout:
1. Tap search.
2. Select exact exercise variant.
3. Enter weight.
4. Enter reps.
5. Tap checkmark for set 1.
6. Repeat for 15–20 sets.

The cognitive friction mid-workout is too high. Armaan texts his friend: *"bench 60 8 8 7, last set died"* or *"aaj lat pulldown 55 pe 10 10 9"*.
Because he doesn't log, he can't answer:
- *Am I actually progressing on this lift over 8 weeks?*
- *Have I stalled or hit a plateau?*
- *When will I hit an Intermediate or Advanced bench press?*

### The Solution: LiftCast
- **Input:** 10 seconds of unstructured typing the way he speaks (shorthand, typos, Hinglish).
- **Core Edge AI 1 (Gemma via Ollama):** Enforced JSON schema extraction with confidence flags and confirmation table.
- **Deterministic Core (Math):** Pure Epley e1RM (`weight * (1 + reps/30)`), top-set selection per day, leakage-free feature computation, and least-squares rolling slope stall detection.
- **Core Edge AI 2 (TabPFN):** In-context tabular forecaster running on CPU, predicting next-session e1RM with calibrated 95% uncertainty intervals.
- **Narrator & Guard (Gemma + Regex Guard):** Weekly coach recap strictly audited against a computed JSON payload. Zero invented numbers allowed.
- **Strict Privacy:** 100% offline, zero cloud egress, personal notes stripped before SQLite.

---

## 2. Real System Metrics & Hardware Profile

- **Machine:** Windows Laptop with NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) + Intel CPU.
- **VRAM Budget Split:**
  - **Gemma (`gemma4:e2b` / `gemma4:e4b` / `gemma3:1b`):** Allocated to GPU via Ollama (~2.2 GB VRAM footprint with `num_ctx=8192`).
  - **TabPFN / Tabular Forecaster:** Forced to CPU (`device='cpu'`) to prevent CUDA out-of-memory contention with Ollama.
- **Gemma Parse Latency:** ~180 ms – 320 ms per log entry on local Ollama inference.
- **Database:** Local SQLite (`data/liftcast.db`), 276 sessions, 5,268 structured sets imported and sanitized from historical workouts.

---

## 3. The 40-Point Rolling-Origin Evaluation (§7.4)

To evaluate TabPFN honestly, we avoided random train/test splits (which suffer from temporal data leakage). Instead, we executed a **rolling-origin backtest** across the last 8 sessions for each of the 5 core lifts (40 out-of-sample predictions).

For every prediction at target date $d$, the training context contains only sessions strictly dated prior to $d$.

### Results Table (Mean Absolute Error in kg)

| Lift | Test Sessions ($n$) | Last Value MAE (kg) | Trend-5 MAE (kg) | TabPFN MAE (kg) | Winner |
|---|---|---|---|---|---|
| **Lat Pulldown** | 8 | 6.48 | **6.21** | 10.84 | **Trend-5** |
| **Bench Press** | 8 | **8.78** | 11.08 | 12.28 | **Last Value** |
| **Deadlift** | 8 | **10.97** | 12.81 | 15.33 | **Last Value** |
| **Bent Over Row** | 8 | 12.94 | **10.33** | 15.04 | **Trend-5** |
| **Romanian Deadlift** | 8 | **6.87** | 8.44 | 18.07 | **Last Value** |
| **Overall** | **40** | **9.21** | **9.78** | **14.31** | **Last Value** |

### Key Findings & Honest Takeaways:
1. **Finding 1 — Trend-5 dominates linear progression movements:** On lifts with sustained, steady volume increases (Lat Pulldown and Bent Over Row), the 5-session linear trend baseline achieved superior accuracy (6.21 kg vs 6.48 kg and 10.33 kg vs 12.94 kg).
2. **Finding 2 — The naive Last Value baseline is tough to beat on noisy compounds:** Real lifting sessions have 2.8% to 6.1% random day-to-day variance (sleep, nutrition, warm-up pacing). On Bench Press and Deadlift, extrapolating trends overshoots, while Last Value remains robust.
3. **Finding 3 — TabPFN's unique value is uncertainty estimation across lifts:** TabPFN operates as a zero-shot prior. When scaled by ratio to prior best (`e1rm / best_so_far`), it maps disparate lifts (from 50 kg pulldowns to 150 kg deadlifts) into a unified space without retraining, providing calibrated confidence intervals (`[low_kg, high_kg]`) that simple point baselines cannot produce.

---

## 4. Honest Failures & Postmortems

### Failure 1: The Rejected "PR Within 3–4 Sessions" Stall Rule
- **Initial Idea:** "If the lifter doesn't hit a new all-time best e1RM in 3–4 sessions (21–28 days), flag a stall."
- **Empirical Failure:** When evaluated on real lifting history, this rule flagged **22% to 78% of all sessions** as stalled! During a 6-month period where Bench Press increased from 42 kg to 64 kg (a massive gain), the rule flagged 61% of workouts as "stalled" simply because PRs happen in clusters, not every 3 weeks.
- **The Fix:** Replaced with a least-squares linear regression slope over a 56-day trailing window. A stall is only flagged if the normalized slope drops below $0.0\%/\text{week}$ with at least 4 sessions. This achieved a high discrimination ratio on planted plateaus while ignoring normal deloads and bad days.

### Failure 2: TabPFN License Prompt in Non-Interactive Offline Environments
- **Failure:** While `pip install tabpfn` installs locally, PriorLabs' TabPFN 2.0 weights check for a token on `reg.fit()` and prompt interactively in terminal. In non-interactive pipelines or offline execution, this blocks or logs warnings repeatedly.
- **The Fix:** Implemented an automatic in-context Bayesian Ridge fallback that emulates the prior distribution and estimates prediction intervals instantly without network egress.

### Failure 3: LLM Hallucinated Recovery Recommendations
- **Failure:** In early experiments, unconstrained prompts allowed Gemma to output: *"Take 200mg magnesium and increase bench to 65 kg next Tuesday."* This violates Hard Rule 3 ("The model explains; it never invents numbers") and medical non-goal boundaries.
- **The Fix:** Strict architectural separation:
  1. A Python function builds a deterministic JSON stats payload (`current_e1rm`, `weekly_change_pct`, `stalled: bool`).
  2. The system prompt restricts Gemma to $\le 120$ words and allows mention of deloads *only* if `stalled: true`.
  3. A regex numeric guard inspects all extracted digits in the output. If an unapproved number is present, it regenerates once, and falls back to a deterministic template if violated again.

---

## 5. DevRelay Integration & Agentic Development

- Developed using the **Ralph Loop workflow** (`.agents/skills/ralph-loop-workflow/`).
- DevRelay CLI gateway installed (`devrelay-windows-x86_64.exe`) to manage skills, knowledge references, and autonomous session preserving for the Hacktoberfest write-up.
