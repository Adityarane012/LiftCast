# LiftCast - Ralph Loop Progress Tracker

Autonomous development tracker following the Ralph Loop workflow.

## Loop Iterations

- [x] **Iteration 0: Preflight & Environment Setup**
  - [x] Git repo initialized with strict privacy `.gitignore` (`data/raw/`, `*.db`)
  - [x] Python 3.13 venv with `tabpfn`, `torch` (CPU), `pandas`, `streamlit`, `plotly`, `ollama`, `pytest`
  - [x] Ollama verified locally with Gemma
  - [x] Raw data copied to `data/raw/liftoff_workout_data.csv` (untracked)
  - [x] Archify, Inspira-UI, and Animate-UI references available

- [ ] **Iteration 1: Database Layer (`src/liftcast/db.py`)**
  - [ ] Implement SQLite schema: `sessions`, `sets`, `exercise_aliases`
  - [ ] CRUD operations & alias resolver
  - [ ] Unit tests in `tests/test_db.py`

- [ ] **Iteration 2: Liftoff Importer (`src/liftcast/importer.py`)**
  - [ ] Filter cardio, weight=0, reps=0
  - [ ] Drop private columns: `Workout Name`, `Notes` (Hard Rule 2)
  - [ ] Unit conversion: `/ 2.2` to kg (Hard Rule 5)
  - [ ] Group by calendar day (date)
  - [ ] Seed exercise aliases
  - [ ] Idempotency (no duplicate entries)
  - [ ] Unit tests in `tests/test_importer.py`

- [ ] **Iteration 3: Metrics & Feature Engineering (`src/liftcast/metrics.py`)**
  - [ ] Epley e1RM formula: `weight_kg * (1 + reps / 30.0)`
  - [ ] Top-set calculation per session
  - [ ] Leakage-free feature calculation (strictly prior sessions): `days_since_start`, `days_since_prev`, `prev_ratio`, `rollmean3_ratio`, `sessions_14d`, `lift`
  - [ ] Strength tiers & next tier projection with slope ± SE
  - [ ] Unit tests in `tests/test_metrics.py`

- [ ] **Iteration 4: Synthetic Data Generator (`src/liftcast/synthetic.py`)**
  - [ ] Diminishing returns curve `base + gain * log(1 + t/tau)` + 4% Gaussian noise
  - [ ] Planted plateau (8 weeks), deload (-10%), bad day (-15%), 3-week gap
  - [ ] Ground truth `is_plateau` column
  - [ ] Unit tests in `tests/test_synthetic.py`

- [ ] **Iteration 5: Deterministic Stall Detection (`src/liftcast/detect.py`)**
  - [ ] Grid experiment on window W ∈ {42, 56, 70, 84} × theta ∈ {0.0, 0.5}
  - [ ] Discrimination ratio calculation (inside plateau ÷ outside)
  - [ ] Rule implementation `detect_stalls()`
  - [ ] Unit tests on synthetic planted plateau & bad days in `tests/test_detect.py`

- [ ] **Iteration 6: Forecasting Engine & Rolling-Origin Evaluation (`src/liftcast/forecast.py`)**
  - [ ] Protocol `Forecaster`
  - [ ] `LastValueBaseline`, `LinearTrendBaseline(n=5)`
  - [ ] `TabPFNForecaster` (CPU, fixed seed, prediction quantiles)
  - [ ] Rolling-origin evaluation runner on last 8 sessions for 5 key lifts (40 test points)
  - [ ] Generate comparison table: MAE (kg), MAPE (%)
  - [ ] Unit tests in `tests/test_forecast.py`

- [ ] **Iteration 7: Free-Text Parser (`src/liftcast/parser.py`)**
  - [ ] Ollama Gemma integration with JSON schema
  - [ ] Post-processing: alias resolution, default kg, bounds validation
  - [ ] Low confidence / unknown exercise confirmation flagging
  - [ ] Test fixtures `tests/fixtures/parser_cases.json`
  - [ ] Unit tests with mocked Ollama in `tests/test_parser.py`

- [ ] **Iteration 8: AI Coach Summary with Numeric Guard (`src/liftcast/coach.py`)**
  - [ ] Stats payload builder
  - [ ] LLM prompt contract (≤ 120 words, deload option only if stalled, no medical advice)
  - [ ] Strict numeric guard (extract all numbers, verify against payload, fallback if violated)
  - [ ] Unit tests in `tests/test_coach.py`

- [ ] **Iteration 9: Architecture Diagram with Archify**
  - [ ] Complete system dataflow architecture spec
  - [ ] Finalize with Archify CLI into interactive standalone HTML
  - [ ] Verified via Archify automated gates

- [ ] **Iteration 10: Modern UI (`app.py`)**
  - [ ] Streamlit app with Inspira UI / Animate UI dark glassmorphic styling
  - [ ] Page 1: 10-Second Text Logger with live parse & confirmation table
  - [ ] Page 2: Lift Progress & Forecast with interactive Plotly graphs, prediction intervals, and stall shading
  - [ ] Page 3: Weekly Coach Recap with verified numeric badges & tier projections
  - [ ] Page 4: Benchmark & Architecture explorer (eval table + embedded Archify interactive diagram)

- [ ] **Iteration 11: End-to-End Verification & Documentation**
  - [ ] `pytest -q` passing on all test suites
  - [ ] `docs/WRITEUP_NOTES.md` updated with real numbers, eval table, honest failures
  - [ ] `README.md` with complete instructions and offline execution guide
