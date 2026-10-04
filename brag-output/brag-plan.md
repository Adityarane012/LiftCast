# Brag Plan: LiftCast

## What is this app?
LiftCast is a local-first AI workout tracker and strength progress forecaster built for a gym buddy who refuses to use clunky fitness apps, turning informal WhatsApp-style workout shorthand into structured sets and TabPFN in-context strength forecasts.

## The angle
Fitness trackers demand tedious, multi-step structured data entry while you're out of breath and have chalk on your hands. LiftCast accepts raw informal text (*"bench 60 8 8 7, last set died"*), parses it locally with open-weight Gemma, forecasts next-session performance with TabPFN on CPU, snaps weight onto visual barbell sleeves, and briefs you through your earbuds without ever touching your phone.

## Hook (first 2-3 seconds)
The absurdity of commercial fitness apps: tapping search dropdowns and typing set numbers while out of breath with chalky hands, contrasted with the reality: a quick WhatsApp text to a friend.

## Key moments (the middle)
- **10-Second Text Logger:** Pasting raw shorthand / Hinglish text and seeing local Gemma instantly structure it into canonical exercises and e1RM sets with an enforced schema.
- **TabPFN In-Context Forecaster:** Prior Labs' tabular foundation model running on CPU to project next-session top sets with 95% confidence intervals and 56-day regression plateau detection.
- **Barbell Plate Calculator & Hands-Free Audio:** Visual color-coded barbell sleeve (20kg/10kg/5kg/2.5kg plates) so zero mental math is required, paired with a 10-second gym earbud audio briefing.

## Outro / punchline
"Built for a friend. 100% offline. Zero cloud egress." — Live on GitHub with 1-click Codespaces sandbox.

## User flow worth showing
1. Entry: Paste informal chat note (*"bench 60 8 8 7, last set died"*).
2. Key action: Local Gemma parses + TabPFN forecasts next top set (77.5 kg ± 3.7 kg) + plateau detector confirms progression.
3. Result: Visual barbell sleeve shows exact plates to load + audio briefing plays in earbuds.

## Tone
- Preset: `polished`
- Creative direction: crisp, modern, confident product film for an indie local-first AI tool.
- Interpretation: clean typography, dark-mode athletic aesthetic (`#0B0F19`, `#3B82F6`, `#10B981`), restrained motion, zero generic SaaS buzzwords, highlighting real code and real UI.

## Format: landscape — 1920x1080 (30 fps)
## Duration: 20 seconds (600 frames)

## Visual identity (from the project)
- Background: `#0B0F19` (rich midnight slate)
- Card Surface: `#161F30` (subtle border `#2A3B5C`)
- Accent Primary: `#3B82F6` (vibrant electric blue)
- Accent Success: `#10B981` (clean athletic emerald)
- Accent Warning: `#F59E0B` (plateau amber)
- Text Primary: `#F8FAFC`
- Text Muted: `#94A3B8`
- Display font: Segoe UI Bold / Inter
- Body font: Segoe UI / Roboto

## Share copy (draft)
My friend lifts 4 days a week but refused to log workouts with chalky hands. I built LiftCast: type raw shorthand, let local Gemma parse the sets, and let TabPFN forecast next week's top set on CPU. 100% offline with zero cloud egress.

## Audio direction
- Role: upbeat, confident, modern tech launch bed (`happy-beats-business-moves-vol-10-by-ende-dot-app.mp3` at 109.96 BPM).
- SFX posture: restrained motion accents (keyboard tap, card arrival snap, chime confirmation).
- Audio arc: subtle groove intro during the hook, full beat drop on LiftCast reveal, sustained rhythm during TabPFN/barbell showcase, and clean fade-out at 19.5s.

## Storyboard

### Scene 1 — The Problem: Chalky Hands & Tedious Apps — 3.5s (0.0s – 3.5s)
- Visual: Dark slate background. Split contrast: "Every fitness app wants 20 tedious inputs with chalk on your hands." Strikethrough. In pops a familiar WhatsApp message: *"bench 60 8 8 7, last set died"*.
- Audio: Keyboard clatter + message pop, transition into rhythm bed.

### Scene 2 — The Solution: LiftCast & Gemma NLP — 4.0s (3.5s – 7.5s)
- Visual: LiftCast hero logo and app frame slide in. Live demonstration of the 10-second text parser converting messy shorthand into clean, canonical sets with Epley e1RM.
- Audio: Upbeat beat drop, clean interface snap.

### Scene 3 — TabPFN In-Context Strength Forecaster — 4.5s (7.5s – 12.0s)
- Visual: Prior Labs' TabPFN card. In-context forecasting graph showing historical trend, next-session forecast (77.5 kg), and 95% confidence bounds. 56-day regression slope flags progression.
- Audio: Rhythmic drive, subtle positive chime on forecast lock.

### Scene 4 — Barbell Sleeve Visualizer & Hands-Free Audio — 4.5s (12.0s – 16.5s)
- Visual: Olympic barbell sleeve rendering showing exact color-coded plates (20kg Blue, 10kg Green, 5kg White, 2.5kg Red). Below it, the 10-second earbud audio coach briefing waves.
- Audio: Metallic barbell plate click, audio briefing waveform pulse.

### Scene 5 — Outro & Launch — 3.5s (16.5s – 20.0s)
- Visual: Full hero branding card: "LiftCast — Built for a friend. 100% offline. Zero cloud egress." GitHub repository link, 59/59 passing tests badge, and 1-click Codespaces badge.
- Audio: Music peaks and smoothly fades out with clean reverb tail.
