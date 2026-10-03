# PROJECT.md — LiftCast: Detailed Specification

> CLAUDE.md holds the **rules**. This file holds the **design**: what we build, how each part works,
> how we know it works, and what we will claim. When the two conflict, CLAUDE.md wins and this file
> gets corrected.

---

## 1. Problem

**User:** Armaan — a college student who lifts regularly but has never logged a session.

**Why he doesn't log:** every tracker wants structured input mid-set (pick exercise from a menu,
type weight, type reps, repeat). The friction is higher than the perceived value, so he logs nothing,
and therefore can't tell whether he's progressing or stuck.

**What he needs:**
1. Logging that costs ~10 seconds: type it the way you'd text a friend.
2. A clear answer to "am I progressing on this lift?" — and an early warning when he's stalling.

**What success looks like for Armaan:** after a session, he types one message and sees his trend,
his forecast, and whether anything is stalling — with no data leaving his laptop.

---

## 2. Goals and non-goals

### Goals
- G1. Parse free-text workout logs (shorthand, typos, Hinglish) into structured sets with a **local** Gemma model.
- G2. Forecast next-session top-set e1RM per lift with **local** TabPFN, evaluated honestly against baselines.
- G3. Detect stalls with a deterministic, tested rule; use the forecast as an early warning.
- G4. Show strength tiers (own names) and a projected time *range* to the next tier.
- G5. Weekly plain-language summary that narrates computed numbers and **never invents any**.
- G6. Everything runs offline on a 4 GB-VRAM laptop; data never leaves the machine.

### Non-goals
- No calorie/nutrition features. No training-program generation. No medical or injury advice.
- No cloud services, accounts, sync, or mobile app.
- No claims of validated stall detection beyond the evidence we actually have.

---

## 3. User stories and acceptance criteria

| ID | Story | Acceptance criteria |
|---|---|---|
| US1 | As Armaan, I type "bench 60 8 8 7, last set died" and it's saved correctly. | Parsed to Bench Press, 60 kg, sets of 8/8/7, note "last set died"; shown for confirmation before saving; `raw_text` stored. |
| US2 | When the parser is unsure, it asks instead of guessing. | Unknown exercise or missing weight/reps → UI confirmation prompt; nothing silently saved. |
| US3 | I see my trend and forecast for a lift. | Chart: actual top-set e1RM per session + forecast for next session (+ band if TabPFN provides intervals). |
| US4 | I'm warned when a lift stalls. | Stall flag per lift per the finalized rule (§8); explanation states window and slope. |
| US5 | I see my tier and how far the next one is. | Tier from e1RM ÷ bodyweight; next-tier projection shown as a range of weeks, or "not projectable" when the trend is flat/negative. |
| US6 | I get a weekly recap. | Recap text where every number appears in the computed stats payload (verified, §10). |

---

## 4. System overview

```
           free text                    Liftoff CSV (historical data, validation only)
               │                                   │
         parser.py (Gemma/Ollama,            importer.py (filters, ÷2.2, drop
         JSON schema) ──confirm──┐            private columns)
                                 ▼                   ▼
                              db.py  (SQLite: sessions, sets, exercise_aliases)
                                 │
                            metrics.py  (e1RM, session top sets, features, tiers)  ← pure, tested
                          ┌──────┴────────┐
                   forecast.py        detect.py
               (TabPFN + baselines)  (trend-based stall rule)                      ← detect is pure, tested
                          └──────┬────────┘
                             coach.py  (Gemma narrates a stats payload; numbers verified)
                                 │
                              app.py  (Streamlit)
```

AI sits at the edges (parse, forecast, narrate). The core that produces every displayed number
(`metrics.py`, `detect.py`) is deterministic and unit-tested.

---

## 5. Data

### 5.1 Liftoff import (`importer.py`)

| Step | Rule |
|---|---|
| Read | Columns: Date, Duration, Workout Name, Exercise Name, Set Order, Weight, Reps, Distance, Seconds, RPE, Notes |
| Drop columns | `Workout Name`, `Notes` (personal health details), `Duration`, `RPE` (59/5,559 filled) |
| Filter rows | Drop `Weight == 0`, `Reps == 0`, `Distance > 0`, `Seconds > 0` |
| Units | `weight_kg = Weight / 2.2` (export is lb converted at factor 2.2; confirm in app settings) |
| Sessions | One session per **calendar day** (some days have 2 timestamps) |
| Source | `sessions.source = 'liftoff'` |
| Exercise names | Keep Liftoff names as canonical; seed `exercise_aliases` with common shorthand |

**Acceptance:** importer is idempotent (re-running doesn't duplicate); counts after import are logged
(expected order: ~5,000+ sets, 308 days before filtering cardio/bodyweight); no private column ever
reaches SQLite or stdout.

### 5.2 Schema
```sql
CREATE TABLE sessions (
  id INTEGER PRIMARY KEY,
  date TEXT NOT NULL UNIQUE,          -- ISO date (calendar day)
  bodyweight_kg REAL,                 -- nullable; entered manually
  source TEXT NOT NULL CHECK (source IN ('liftoff','manual'))
);
CREATE TABLE sets (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  exercise TEXT NOT NULL,             -- canonical name
  weight_kg REAL NOT NULL CHECK (weight_kg > 0),
  reps INTEGER NOT NULL CHECK (reps > 0),
  note TEXT,
  raw_text TEXT                       -- original typed line (manual entries)
);
CREATE TABLE exercise_aliases (
  alias TEXT PRIMARY KEY,             -- lowercase
  canonical TEXT NOT NULL
);
```

### 5.3 Synthetic data (`synthetic.py`)
Used **only** to unit-test the stall detector and edge cases. Labeled synthetic in filenames, UI, and post.
- Progression: `e1rm(t) = base + gain * log(1 + t/τ)` (diminishing returns), plus Gaussian noise
  with σ ≈ 4% (matches real session noise of 2.8–6.1%).
- Planted events with known dates: one plateau (6–10 weeks flat), one deload (−10% for 1 week),
  one bad day (−15% single session), one 3-week gap.
- Output includes a ground-truth label column (`is_plateau`).

### 5.4 Lifts used for forecasting
| Lift | Sessions | Notes |
|---|---|---|
| Lat Pulldown | 112 | Real plateau Apr–Sep 2026 (hand label) |
| Bench Press | 100 | Decelerating |
| Deadlift | 78 | Near-linear |
| Bent Over Row | 55 | Near-linear |
| Romanian Deadlift | 50 | Dip early 2026 (candidate label) |

---

## 6. Parser (`parser.py`)

### 6.1 Contract
```python
def parse_log(text: str, model: str = "gemma4:e2b") -> ParseResult
# ParseResult: sets: list[ParsedSet], needs_confirmation: bool, issues: list[str]
# ParsedSet: exercise_raw, exercise (canonical | None), weight, unit ('kg'|'lb'), reps, set_count, note
```

### 6.2 JSON schema passed to Ollama `format=`
```json
{
  "type": "object",
  "properties": {
    "entries": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "exercise": {"type": "string"},
          "weight": {"type": ["number", "null"]},
          "unit": {"type": "string", "enum": ["kg", "lb", "unknown"]},
          "reps": {"type": "array", "items": {"type": "integer"}},
          "note": {"type": ["string", "null"]}
        },
        "required": ["exercise", "weight", "unit", "reps", "note"]
      }
    }
  },
  "required": ["entries"]
}
```
`reps` is a list (one item per set): "60 8 8 7" → `[8, 8, 7]`; "60x8x3" → `[8, 8, 8]`.

### 6.3 Post-processing (deterministic, tested)
- Map `exercise` through `exercise_aliases` (lowercased, trimmed); unknown → `needs_confirmation`.
- `unit == "unknown"` → default **kg**, flagged in `issues`.
- Sanity bounds: 1 ≤ reps ≤ 50; 1 ≤ weight_kg ≤ 400 → otherwise `needs_confirmation`.
- Model options: `num_ctx=8192`, `temperature=0`.

### 6.4 Test set
- `tests/fixtures/parser_cases.json`: Armaan's 5 real lines (**TODO**) + ~15 edge cases:
  shorthand ("bp"), typos ("bech"), Hinglish ("aaj bench 60 pe 8 reps"), lb input, bodyweight
  exercise without weight, multiple exercises in one message, "same as last time".
- Unit tests mock Ollama. A separate, manual `scripts/eval_parser.py` runs the real model on the
  fixtures and reports exact-match accuracy per field → numbers go in WRITEUP_NOTES.

---

## 7. Forecasting (`forecast.py`)

### 7.1 Unit of prediction
Per lift, per session: the **top-set e1RM** (max over that day's sets), Epley:
`e1rm = weight_kg * (1 + reps / 30)`.

### 7.2 Features (computed only from strictly earlier sessions — no leakage)
| Feature | Definition |
|---|---|
| `days_since_start` | days since first-ever session (any lift) |
| `days_since_prev` | days since previous session of this lift |
| `prev_ratio` | previous e1RM ÷ best-so-far (this lift, prior sessions only) |
| `rollmean3_ratio` | mean of last 3 e1RMs ÷ best-so-far |
| `sessions_14d` | sessions of this lift in the prior 14 days |
| `lift` | categorical |

**Target:** `e1rm / best_so_far_prior` (ratio), converted back to kg for evaluation and display.
Ratios let one model learn across lifts of very different absolute weights.

### 7.3 Interface
```python
class Forecaster(Protocol):
    def predict_next(self, history: pd.DataFrame, lift: str, as_of: date) -> Forecast
# Forecast: point_kg, low_kg | None, high_kg | None, method
```
Implementations: `TabPFNForecaster` (CPU, fixed seed), `LastValueBaseline`, `LinearTrendBaseline(n=5)`.
Check TabPFN docs for interval/quantile output; if available, populate `low_kg/high_kg`.

### 7.4 Evaluation protocol (decided)
- Test points: **last 8 sessions of each of the 5 lifts** → 40 predictions.
- **Rolling origin:** for each test point dated `d`, context/training rows = all rows (all lifts)
  with date `< d`. Applies equally to baselines.
- Metrics: MAE (kg), MAPE (%), per lift and overall; count of lifts where each method wins.
- Report table format:

| Lift | n | Last value MAE | Trend-5 MAE | TabPFN MAE | Winner |
|---|---|---|---|---|---|

- Interpretation guardrails: with n = 8 per lift, small MAE differences are noise; only call a win
  "clear" if it holds overall **and** on ≥ 3 of 5 lifts. Expect Trend-5 to be strong on near-linear lifts.

---

## 8. Stall detection (`detect.py`) — experiment before implementation

### 8.1 Rejected definition (evidence)
"No new best e1RM within 3–4 sessions / 21–28 days" flags **22–78%** of sessions
(bench 61–78% while it rose 92→141 lb). PRs come in bursts; session noise is 2.8–6.1%.

### 8.2 Candidate definition
Stall at session `s` if the slope of a least-squares fit of e1RM vs. weeks, over sessions in the
trailing window `W` days, is `< θ` %/week (slope ÷ window mean), with at least `m = 4` sessions in the window.

### 8.3 Experiment protocol (pre-registered — do not change after seeing results)
1. Hand labels, fixed **before** running: Lat Pulldown 2026-04-01 → 2026-09-30 = plateau;
   RDL early-2026 dip = candidate (decide yes/no from the quarterly chart first).
2. Grid: `W ∈ {42, 56, 70, 84}` days × `θ ∈ {0.0, 0.5}` %/week.
3. Metrics per config: overall flag rate; **discrimination ratio** = flag rate inside labeled
   plateaus ÷ flag rate outside.
4. Selection rule: highest discrimination ratio among configs with overall flag rate ≤ 30%.
   Ties → shorter window (earlier warning).
5. Results so far: W = 42–56 → overall 18–40%, inside-plateau 37–52% (ratio ≈ 1.1–1.5; weak).
6. Validate the chosen config on synthetic data (planted plateau must be flagged; bad day alone must not).

### 8.4 Contract
```python
def detect_stalls(sessions: pd.Series, window_days: int, theta_pct_week: float, min_n: int = 4) -> pd.DataFrame
# columns: date, slope_pct_week, n_in_window, stalled (bool | NA)
```

### 8.5 Early warning (TabPFN's role)
If the forecast for the next session is within ±1% of the current rolling mean for 2 consecutive
predictions → "possible stall forming". Shown as a softer signal than the rule-based flag.

---

## 9. Strength tiers (`metrics.py`)
- Score = top-set e1RM (recent best) ÷ bodyweight_kg.
- **Own tier names and thresholds** (not Liftoff's). Thresholds must come from a cited public
  strength-standards source, per lift. **TODO:** pick source and fill table.

| Tier | Bench ratio | Deadlift ratio | Row/Pulldown ratio |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

- Projection: weeks to next tier = (threshold − current) ÷ trend slope (kg/week) from the last 8–12
  weeks; shown as a range using slope ± its standard error. If slope ≤ 0 → "not projectable right now".

---

## 10. Coach summary (`coach.py`)
- Input: a JSON **stats payload** built by code (per-lift weekly change %, sessions this week,
  stall flags, tier, forecast). No raw logs.
- Prompt contract: narrate only these numbers; ≤ 120 words; mention a deload only as an option
  if a stall flag is true; no medical advice.
- **Numeric guard (tested):** extract every number in the output; each must match a payload value
  (after rounding to 0 or 1 decimal). If any doesn't → regenerate once → else fall back to a
  template summary. Log violations to WRITEUP_NOTES (good honest-failure material).

---

## 11. UI (`app.py`, Streamlit)

| Page | Content |
|---|---|
| **Log** | Text box → parsed preview table (editable) → Confirm & Save. Shows issues/needs_confirmation. |
| **Progress** | Lift selector; e1RM chart (actual, forecast point/band, stall-flag shading); tier card. |
| **Insights** | Weekly coach summary; evaluation table (from §7.4) with "Synthetic"/"Real" labels where relevant. |

Sidebar: bodyweight input; model selector (e2b/e4b); data source badge (Liftoff import vs manual).

---

## 12. Testing plan

| Module | Tests |
|---|---|
| importer | unit conversion (242 → 110.0); filters; private columns absent; idempotency; day grouping |
| metrics | Epley values; top-set selection ignores warm-ups; features use only prior rows (leakage test) |
| parser (post-processing) | alias mapping; unit default; bounds → needs_confirmation; reps expansion |
| forecast | rolling-origin never sees rows dated ≥ d (assert in code + test); baselines correct on toy series |
| detect | slope sign on synthetic up/flat/down; planted plateau flagged; single bad day not flagged; min_n respected |
| coach | numeric guard catches an invented number |

Run: `pytest -q`. Core modules (`metrics`, `detect`, importer transforms) should be fully covered.

---

## 13. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| TabPFN weights need HF login/license | Medium | Verify first thing during setup |
| VRAM contention (Gemma + TabPFN) | Medium | TabPFN forced to CPU |
| E2B misparses Armaan's style | Medium | Confirmation step; fixtures; try E4B |
| TabPFN doesn't beat Trend-5 | High on linear lifts | Report honestly; frame per-lift findings |
| Stall rule weak discrimination | High | Pre-registered experiment; state evidence limits |
| Unit assumption wrong (2.2 vs 2.2046) | Low | Confirm in app; one constant to change |
| Time overrun | High | Cut lines below |

---

## 14. Timeline (IST) and cut lines

| Block | Deliverable |
|---|---|
| Sat (rest of day) | importer + tests; metrics + tests; forecast + baselines; **eval table** |
| Sun 09:00–13:00 | stall experiment → detect.py; parser + fixtures; coach + guard |
| Sun 13:00–16:00 | Streamlit UI; tiers (if source chosen) |
| Sun 16:00–18:00 | Hand to Armaan; record reaction verbatim; 1–2 min demo video |
| Sun 18:00–23:00 | DEV post draft |
| Mon 07:00–10:00 | Polish, final checks, **submit by 10:00** (deadline 12:29) |

**Cut order if behind:** (1) stretch Q&A, (2) tiers, (3) coach summary → template text,
(4) forecast bands. **Never cut:** parser + confirmation, eval table with baselines, honest write-up.

---

## 15. Definition of done
- [x] Fresh clone + README steps → app runs offline with Ollama + local TabPFN.
- [x] `pytest -q` passes (58/58 passed).
- [x] Eval table (§7.4) generated by a script and pasted into the post.
- [x] Stall config chosen by the §8.3 rule, results reported with limits.
- [x] Armaan used it on real sessions; his words quoted.
- [x] No private data in repo (`git log` checked for CSV/db files).
- [ ] DEV post published with template sections; categories listed: Gemma, TabPFN.
- [ ] DevRelay agent session embedded or linked.

---

## 16. DEV post outline (template sections)
1. **What I Built** — Armaan, the no-logging problem, the 10-second log.
2. **Demo** — video + screenshots (parser on his real lines, forecast chart, stall flag).
3. **Code** — repo embed.
4. **How I Built It** — Gemma 4 (local, JSON schema), TabPFN (in-context, rolling-origin eval),
   deterministic core; the eval table; one honest failure (e.g. rejected PR-based stall rule with numbers).
5. **Why Open Innovation Matters** — offline, private body data, ₹0, swappable models, enforced schema;
   honest note on where a closed model might write smoother summaries.
6. **My Agent Session** — DevRelay embed.
7. **Prize Categories** — Best Use of Gemma; Best Use of TabPFN.
8. Armaan's reaction (verbatim).

---

## 17. Open decisions

| # | Decision | Owner | Blocks |
|---|---|---|---|
| D1 | Collect Armaan's 5 log lines | Maintainer | parser fixtures |
| D2 | Confirm lb export factor in Liftoff | Maintainer | importer constant |
| D3 | Stall config via §8.3 | Maintainer | detect.py |
| D4 | Strength-standards source + tier names | Maintainer | tiers |
| D5 | E2B vs E4B (quality on fixtures, speed) | Maintainer | default model |
| D6 | Mention historical plateau dip in post or not | Maintainer | write-up |
