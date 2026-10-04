# CLAUDE.md — LiftCast (working name; rename freely)

## What this is
A local-first lift logger + progress forecaster, built for one real person: my friend,
who lifts but never logs because logging apps are tedious. He types sessions the way he talks
("bench 60 8 8 7, last set died"); a local Gemma model structures it, TabPFN forecasts progress,
and deterministic code detects stalls.

Entry for the **DEV Hacktoberfest Weekend Challenge: "Build for a Friend"**.
- **Deadline:** Mon Oct 5, 2026, 06:59 UTC = **12:29 PM IST**. Target submit: **10:00 AM IST**.
- **Rule:** this repo must be created and completed inside the challenge window (started Oct 2+).
  Any commit after the deadline must be noted in README.
- **Judging:** writing quality (heaviest) > relevance > creativity > technical execution > partner tech.

## Prize categories we enter (and ONLY these)
| Category | Meaningful use required |
|---|---|
| **Best Use of Gemma** | Gemma 4 runs locally via Ollama; parses free-text logs and writes the weekly summary |
| **Best Use of TabPFN** | TabPFN (local package) forecasts next-session e1RM from tabular history; beats or honestly loses to baselines |

Do NOT add other partner tools (MongoDB, Tiger Data, Sentry, Render, DigitalOcean, etc.).
They would send data off the laptop and weaken the core "runs locally, data stays private" story.

## Hard rules (never violate)
1. **Local only.** Ollama for Gemma. `tabpfn` local package for TabPFN — **never `tabpfn-client`** (cloud API).
2. **Privacy.** `data/raw/` and `*.db` are gitignored. Never commit the Liftoff CSV.
   On import, **drop `Workout Name` and `Notes` columns** — they contain personal health details.
   Never print those columns to the terminal (agent sessions may be published).
3. **The model explains; it never invents numbers.** Gemma parses text and narrates stats that code
   computed. Every number shown to the user comes from `metrics.py` / `forecast.py`, never from the LLM.
4. **No random train/test splits.** Time-based evaluation only (see Evaluation).
5. **Units:** store everything in **kg**. Liftoff export is in lb converted with factor **2.2**
   (e.g. 242 lb = 110 kg). Convert with `/ 2.2` on import. (Still to confirm in app settings.)
6. **Honesty in claims.** If TabPFN doesn't beat baselines, say so. One or two labeled plateaus = anecdote,
   not validation. Synthetic data must be labeled as synthetic everywhere.
7. Ask before adding dependencies, changing the schema, or changing an evaluation definition.

## Environment
- Windows. Python: `C:\Python313\python.exe` (use full path; PATH conflicts). Use `.venv`.
- GPU: RTX 3050 Laptop, **4 GB VRAM**. Gemma gets the GPU; **run TabPFN on CPU** (avoid VRAM contention).
- Ollama: `gemma4:e2b` (default) and `gemma4:e4b` (compare). Always pass `options={"num_ctx": 8192}`.
  Use Ollama's `format=<JSON schema>` for structured parsing.
- UI: Streamlit. Storage: SQLite (stdlib `sqlite3`). Tests: pytest.

## Data facts (Liftoff export historical baseline — used to validate; lifter starts logging now)
- Columns: Date, Duration, Workout Name, Exercise Name, Set Order, Weight, Reps, Distance, Seconds, RPE, Notes.
- 5,559 sets, 308 workout days, Feb 2025 – Oct 2026, 113 exercises.
- Exclude: Weight == 0, Reps == 0, cardio (Distance > 0 or Seconds > 0). RPE ~empty → don't use.
  `Notes` = planned targets ("3x6-8, RIR 1-3"), not performance → dropped anyway.
- Some days have two timestamps → **group sessions by calendar day**.
- Warm-ups unlabeled → per session use **top-set e1RM** (max over sets).
- e1RM = Epley: `weight_kg * (1 + reps / 30)`.
- Forecast lifts (sessions): Lat Pulldown 112, Bench Press 100, Deadlift 78, Bent Over Row 55, Romanian Deadlift 50.
- No bodyweight column → bodyweight for tiers is entered manually.

## Architecture
```
data/raw/                 # gitignored: Liftoff CSV
data/synthetic/           # generated, labeled synthetic logs (stall-detector unit tests only)
src/liftcast/
  db.py                   # SQLite schema + access
  importer.py             # Liftoff CSV -> SQLite (units, filters, column drops)
  parser.py               # Gemma via Ollama, JSON schema -> structured sets
  metrics.py              # pure functions: e1RM, features, tiers
  forecast.py             # TabPFN + baselines behind one interface
  detect.py               # deterministic trend-based stall rule
  coach.py                # Gemma summary from computed stats only
  synthetic.py            # synthetic generator with planted plateau/deload/bad day
app.py                    # Streamlit UI
tests/                    # pytest; parser tests use fixtures (mock Ollama)
docs/WRITEUP_NOTES.md     # running log of numbers, failures, screenshots for the DEV post
```
Design principle: AI at the edges (parser, forecaster, coach); deterministic, tested core (metrics, detect).

## Schema (SQLite)
```
sessions(id, date, bodyweight_kg NULL, source)            -- source: 'liftoff' | 'manual'
sets(id, session_id, exercise, weight_kg, reps, note NULL, raw_text NULL)
exercise_aliases(alias, canonical)                         -- 'bp','flat bench' -> 'Bench Press'
```
Keep `raw_text` for every manually logged set (debugging + honest-failure section of the post).

## Parser
- Input: free text, shorthand, typos, possibly Hinglish ("aaj bench 60 pe 8 reps").
- Output (JSON schema): list of `{exercise, weight, unit, reps, sets, note}` → normalized via aliases.
- Test cases: real example shorthand lines (TODO: collect 5 from him) + edge cases in `tests/fixtures/`.
- If parse confidence is low or exercise unknown, ask the user to confirm instead of guessing.

## Forecasting
- Target: next session's top-set e1RM for a lift.
- Features: days since first session, days since previous session (this lift), previous e1RM,
  rolling mean of last 3, best-so-far, sessions in last 14 days, lift (categorical).
- One model across lifts; express e1RM as **ratio to that lift's best-so-far** to share data.
- Baselines: (a) last value, (b) linear trend over last 5 sessions.

## Evaluation (decided)
- Hold out the **last N = 8 sessions per lift** (5 lifts → 40 predictions).
- **Rolling-origin:** for each test session, context = all rows (all lifts) strictly before its date.
  TabPFN is in-context, so this costs no retraining. This prevents cross-lift temporal leakage.
- Report MAE in kg and % for TabPFN vs both baselines, per lift and overall.
- Expect the trend baseline to be strong on near-linear lifts (deadlift, row). That's a finding, not a failure.

## Stall detection (OPEN — finish before building detect.py)
- Tested on real data: "no new best e1RM in 3–4 sessions / 21–28 days" flags 22–78% of sessions
  (bench 61–78% while it rose 92→141 lb) → rejected. Session noise is 2.8–6.1% median change.
- Trend rule (slope of linear fit over rolling window < threshold) at 42–56 days: 18–40% flagged overall,
  but only 37–52% inside the known lat pulldown plateau → weak discrimination.
- **Next:** test 70- and 84-day windows. Metric: flag rate inside hand-labeled plateaus ÷ flag rate outside.
- Hand labels (set BEFORE running detector): Lat Pulldown Apr–Sep 2026; candidate RDL dip early 2026.
- Roles: rule = detects past/current stalls; TabPFN = early warning when forecast is flat.

## Features by priority
1. **Core:** importer, parser, metrics, forecast + baselines, rolling-origin eval.
2. Stall detector (after definition is settled).
3. Strength tiers by bodyweight ratio (OWN tier names — not Liftoff's) + projected date range to next tier
   (always a range, never a single date).
4. Weekly coach summary (Gemma narrates computed stats; deload mentioned only as an option).
5. Stretch: questions over the log.

## Schedule
- Sat: importer → parser → metrics → forecast + baselines + eval numbers.
- Sun AM: detector, tiers, coach, Streamlit UI.
- Sun PM: Friend logs real sessions — record reaction verbatim; record 1–2 min demo video.
- Sun night: write the DEV post. Mon by 10:00 IST: submit.

## Write-up capture (append to docs/WRITEUP_NOTES.md as we go)
- Real numbers: s/parse, tokens/s, CPU/GPU split, MAE table vs baselines.
- At least one honest failure and its fix (e.g. a misparse, a rejected stall definition).
- Screenshots: parser on real lines, forecast chart, stall flag.
- "Why open": local, private, ₹0, offline, swappable models, enforced schema. Admit where a closed
  model would likely write smoother text.

## Working style
- The maintainer drives implementation decisions. Explain reasoning and trade-offs before significant changes.
- Prefer small, reviewable commits with meaningful messages.
- Prefer simple, readable code over clever code; pure functions in the core.
