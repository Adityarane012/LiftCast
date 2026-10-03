# LiftCast - Ralph Loop Progress Tracker

Autonomous development tracker following the Ralph Loop workflow.

## Loop Iterations

- [x] **Iteration 0: Preflight & Environment Setup**
  - [x] Git repo initialized with strict privacy `.gitignore` (`data/raw/`, `*.db`)
  - [x] Python 3.13 venv with `tabpfn`, `torch` (CPU), `pandas`, `streamlit`, `plotly`, `ollama`, `pytest`
  - [x] Ollama verified locally with Gemma
  - [x] Raw data copied to `data/raw/liftoff_workout_data.csv` (untracked)
  - [x] Archify, Inspira-UI, and Animate-UI references available

- [x] **Iteration 1: Database Layer (`src/liftcast/db.py`)**
  - [x] Implement SQLite schema: `sessions`, `sets`, `exercise_aliases`
  - [x] CRUD operations & alias resolver
  - [x] Unit tests in `tests/test_db.py`

- [x] **Iteration 2: Liftoff Importer (`src/liftcast/importer.py`)**
  - [x] Filter cardio, weight=0, reps=0
  - [x] Drop private columns: `Workout Name`, `Notes` (Hard Rule 2)
  - [x] Unit conversion: `/ 2.2` to kg (Hard Rule 5)
  - [x] Group by calendar day (date)
  - [x] Seed exercise aliases
  - [x] Idempotency (no duplicate entries)
  - [x] Unit tests in `tests/test_importer.py`

- [x] **Iteration 3: Metrics & Feature Engineering (`src/liftcast/metrics.py`)**
  - [x] Epley e1RM formula: `weight_kg * (1 + reps / 30.0)`
  - [x] Top-set calculation per session
  - [x] Leakage-free feature calculation (strictly prior sessions): `days_since_start`, `days_since_prev`, `prev_ratio`, `rollmean3_ratio`, `sessions_14d`, `lift`
  - [x] Strength tiers & next tier projection with slope ± SE
  - [x] Unit tests in `tests/test_metrics.py`

- [x] **Iteration 4: Synthetic Data Generator (`src/liftcast/synthetic.py`)**
  - [x] Diminishing returns curve `base + gain * log(1 + t/tau)` + 4% Gaussian noise
  - [x] Planted plateau (8 weeks), deload (-10%), bad day (-15%), 3-week gap
  - [x] Ground truth `is_plateau` column
  - [x] Unit tests in `tests/test_synthetic.py`

- [x] **Iteration 5: Deterministic Stall Detection (`src/liftcast/detect.py`)**
  - [x] Grid experiment on window W ∈ {42, 56, 70, 84} × theta ∈ {0.0, 0.5}
  - [x] Discrimination ratio calculation (inside plateau ÷ outside)
  - [x] Rule implementation `detect_stalls()`
  - [x] Unit tests on synthetic planted plateau & bad days in `tests/test_detect.py`

- [x] **Iteration 6: Forecasting Engine & Rolling-Origin Evaluation (`src/liftcast/forecast.py`)**
  - [x] Protocol `Forecaster`
  - [x] `LastValueBaseline`, `LinearTrendBaseline(n=5)`
  - [x] `TabPFNForecaster` (CPU, fixed seed, prediction quantiles)
  - [x] Rolling-origin evaluation runner on last 8 sessions for 5 key lifts (40 test points)
  - [x] Generate comparison table: MAE (kg), MAPE (%)
  - [x] Unit tests in `tests/test_forecast.py`

- [x] **Iteration 7: Free-Text Parser (`src/liftcast/parser.py`)**
  - [x] Ollama Gemma integration with JSON schema
  - [x] Post-processing: alias resolution, default kg, bounds validation
  - [x] Low confidence / unknown exercise confirmation flagging
  - [x] Test fixtures `tests/fixtures/parser_cases.json`
  - [x] Unit tests with mocked Ollama in `tests/test_parser.py`

- [x] **Iteration 8: AI Coach Summary with Numeric Guard (`src/liftcast/coach.py`)**
  - [x] Stats payload builder
  - [x] LLM prompt contract (≤ 120 words, deload option only if stalled, no medical advice)
  - [x] Strict numeric guard (extract all numbers, verify against payload, fallback if violated)
  - [x] Unit tests in `tests/test_coach.py`

- [x] **Iteration 9: Architecture Diagram with Archify**
  - [x] Complete system dataflow architecture spec
  - [x] Finalize with Archify CLI into interactive standalone HTML
  - [x] Verified via Archify automated gates

- [x] **Iteration 10: Modern UI (`app.py`)**
  - [x] Streamlit app with Inspira UI / Animate UI dark glassmorphic styling
  - [x] Page 1: 10-Second Text Logger with live parse & confirmation table
  - [x] Page 2: Lift Progress & Forecast with interactive Plotly graphs, prediction intervals, and stall shading
  - [x] Page 3: Weekly Coach Recap with verified numeric badges & tier projections
  - [x] Page 4: Benchmark & Architecture explorer (eval table + embedded Archify interactive diagram)

- [x] **Iteration 11: End-to-End Verification & Documentation**
  - [x] `pytest -q` passing on all test suites (51 tests)
  - [x] `docs/WRITEUP_NOTES.md` updated with real numbers, eval table, honest failures
  - [x] `README.md` with complete instructions and offline execution guide

- [x] **Iteration 12: GitHub CI/CD Matrix & Open Source Governance (Best Use of GitHub)**
  - [x] Automated GitHub Actions workflow (`.github/workflows/ci.yml`) testing Python 3.11, 3.12, 3.13 matrix
  - [x] PR template with strict privacy checklist and zero-data-leakage verification
  - [x] Issue templates for bug tracking and feature requests
  - [x] MIT License and `CONTRIBUTING.md` guidelines
  - [x] Status badges updated in `README.md`

