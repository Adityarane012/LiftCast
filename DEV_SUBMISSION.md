---
title: LiftCast: A Local-First AI Workout Forecaster Built for a Friend
published: false
tags: devchallenge, weekendchallenge, hf26challenge, opensource
cover_image: https://raw.githubusercontent.com/Adityarane012/LiftCast/main/docs/assets/liftcast_hero.png
---

*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

## What I Built
My friend lifts four days a week at our college gym. He never logs his workouts. Instead, he just texts me quick WhatsApp notes on his walk home: *"bench 60 8 8 7, last set died"*. Because his history lived in unstructured chats, he had no way of knowing whether his strength was actually progressing or stalled.

**LiftCast** is a local-first tracker and strength forecaster built for his workflow:
- **10-Second Shorthand Logger:** Pastes raw text, typos, or Hinglish notes. A local **Gemma** model parses them into structured sets via an enforced JSON schema.
- **In-Context Progress Forecaster:** Prior Labs' **TabPFN** runs locally on CPU to forecast his next session's top set with calibrated 95% prediction intervals.
- **Plateau Detection & Barbell Visualizer:** Uses a 56-day regression slope to flag genuine stalls (ignoring routine deloads) and renders a color-coded barbell sleeve graphic showing exact plates to load.
- **Hands-Free Audio Briefing:** Synthesizes a 10-second voice recap via ElevenLabs (with offline browser speech fallback) straight into his gym earbuds.

## Demo
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Adityarane012/LiftCast)

- **1-Click Cloud Sandbox:** Click the badge above or launch directly at **[codespaces.new/Adityarane012/LiftCast](https://codespaces.new/Adityarane012/LiftCast)** to run the app in your browser. In the terminal, run:
  ```bash
  PYTHONPATH=src streamlit run app.py
  ```
- **Local Demo Data:** In the app sidebar, click **"Seed Rich 6-Month Demo DB"** to populate 26 weeks of authentic training plateaus, forecasts, and plate calculations across 7 compound lifts.

![LiftCast Architecture](https://raw.githubusercontent.com/Adityarane012/LiftCast/main/docs/assets/architecture.png)

## Code
**Live Repository:** [github.com/Adityarane012/LiftCast](https://github.com/Adityarane012/LiftCast)

{% embed https://github.com/Adityarane012/LiftCast %}

Key components:
- `src/liftcast/forecast.py`: Local CPU TabPFN forecaster with a 40-point rolling-origin backtest.
- `src/liftcast/parser.py`: Schema-constrained Gemma parser via Ollama with heuristic regex fallback.
- `src/liftcast/detect.py`: 56-day least-squares linear slope plateau detection.
- `src/liftcast/coach.py`: Strict regex numeric guard preventing LLM stat hallucinations.

## How I Built It
- **TabPFN (Prior Labs):** In-context tabular foundation model running locally on CPU. We normalize lift history as a ratio to personal best, allowing a single prior to forecast across disparate exercises without fine-tuning.
- **Gemma (Google / Ollama):** Runs locally (`gemma4:e2b` / `gemma3:1b`) with an enforced JSON schema to extract structured exercises, units, and rep arrays.
- **Strict Numeric Guard:** Pure mathematical verification layer. Every number in coach summaries is validated against deterministic database stats before speech synthesis.

## Why Does Open Innovation Matter?
- **Data Sovereignty:** Workout logs, bodyweight, and personal notes stay in a local SQLite database (`data/liftcast.db`). Zero cloud egress.
- **100% Offline Gym Floor Reliability:** Works in basement gyms with zero cell service—local Ollama, local CPU TabPFN, and native browser speech fallback.
- **Permanent Availability:** Open weights and local inference mean no monthly API bills, no rate limits, and zero risk of vendor deprecation.

## My Agent Session
Pair-programmed with an AI coding agent to implement the math core, build 59 automated tests, and configure multi-version CI testing (Python 3.11, 3.12, 3.13) on GitHub Actions.

{% agent_session 4326f2b3-2ac2-4dfd-8d9d-9c1622013b1d %}

## Prize Categories
### Best Use of TabPFN
LiftCast uses **TabPFN** running locally on CPU to perform in-context strength progression forecasting and uncertainty estimation from historical session logs, evaluated via a 40-point rolling-origin backtest benchmark against traditional linear regression baselines.
