# Contributing to LiftCast

Thank you for your interest in contributing to **LiftCast**! We are participating in Hacktoberfest 2026.

## Core Architectural Guarantees (Must Never Be Violated)
1. **100% Local & Private:** No user data (workouts, weights, notes, bodyweight) is ever transmitted to external cloud endpoints.
2. **AI at the Edges, Deterministic Core:** Gemma and TabPFN operate at the boundaries (natural language parsing, forecasting, narration). All underlying arithmetic (Epley 1RM, plate loads, regression slopes) must be executed by pure, deterministic Python functions.
3. **Strict Numeric Guard:** Gemma explains numbers; it never invents them. Every numeric value in coach summaries must pass the regex guard whitelist.
4. **Hardware Friendly:** LiftCast runs comfortably on a consumer laptop (tested on 4GB VRAM laptop with TabPFN on CPU and Gemma on Ollama).

---

## Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adityarane012/LiftCast.git
   cd LiftCast
   ```

2. **Set up Python Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

4. **Install and Run Ollama with Gemma:**
   ```bash
   ollama pull gemma4:e2b
   # Or for higher reasoning capacity:
   ollama pull gemma4:e4b
   ```

5. **Run the Test Suite:**
   ```bash
   # Windows PowerShell:
   $env:PYTHONPATH="src"; pytest -v
   # Linux / macOS:
   PYTHONPATH=src pytest -v
   ```

6. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## Contribution Workflow
1. Check existing issues or open a new one using the **Feature Request** or **Bug Report** templates.
2. Fork the repository and create your branch from `main`.
3. Ensure all 58 tests pass before opening a Pull Request.
4. Follow the checklist in `.github/PULL_REQUEST_TEMPLATE.md`.
