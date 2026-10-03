"""Render high-resolution, editorial-quality PNG architecture and flow diagrams.

Produces publication-ready raster PNG images for GitHub README and DEV.to posts.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_architecture_diagram(output_path: Path) -> None:
    """Generate high-resolution system architecture diagram PNG."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#080C14")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title & Subtitle
    ax.text(0.6, 8.4, "LIFTCAST: SYSTEM ARCHITECTURE", fontsize=20, fontweight="bold", color="#F8FAFC", family="sans-serif")
    ax.text(0.6, 8.05, "AI at the edges, deterministic math at the core • 100% offline & private", fontsize=11, color="#94A3B8", family="sans-serif")

    # Hardware & Privacy Boundary
    boundary = patches.FancyBboxPatch(
        (0.5, 1.8), 15.0, 5.8,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        linewidth=1.5, edgecolor="#1E293B", facecolor="#0B1120", linestyle="--"
    )
    ax.add_patch(boundary)
    
    # Boundary Badge
    b_badge = patches.FancyBboxPatch(
        (0.8, 7.4), 7.2, 0.38,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        linewidth=1, edgecolor="#334155", facecolor="#0F172A"
    )
    ax.add_patch(b_badge)
    ax.text(1.0, 7.52, "[ISOLATED] LOCAL HARDWARE BOUNDARY • RTX 3050 (4GB VRAM) • ZERO CLOUD EGRESS", fontsize=8, fontweight="bold", color="#94A3B8", family="monospace")

    def draw_box(x, y, w, h, title, sublabel, tag, bg_col, edge_col, tag_col="#38BDF8"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2", linewidth=1.5, edgecolor=edge_col, facecolor=bg_col)
        ax.add_patch(box)
        # Tag pill
        tag_box = patches.FancyBboxPatch((x + 0.15, y + h - 0.35), len(tag) * 0.12 + 0.3, 0.25, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=0.8, edgecolor=edge_col, facecolor="#0F172A")
        ax.add_patch(tag_box)
        ax.text(x + 0.22, y + h - 0.23, tag, fontsize=7, fontweight="bold", color=tag_col, family="monospace")
        ax.text(x + 0.15, y + h - 0.65, title, fontsize=11, fontweight="bold", color="#F8FAFC", family="sans-serif")
        ax.text(x + 0.15, y + 0.2, sublabel, fontsize=8.5, color="#94A3B8", family="sans-serif")

    # Column 1: Inputs
    draw_box(0.9, 5.2, 2.3, 1.5, "Armaan (User)", "Informal 10s text log\n'bench 60 8 8 7'", "INPUT", "#111827", "#475569", "#94A3B8")
    draw_box(0.9, 2.4, 2.3, 1.5, "importer.py", "Liftoff CSV Sanitizer\nStripped health notes", "SANITIZER", "#111827", "#475569", "#EF4444")

    # Column 2: Parser & DB
    draw_box(3.8, 5.2, 2.5, 1.5, "parser.py", "Local Gemma via Ollama\nLatency: ~240ms (GPU)", "EDGE AI", "#082F49", "#0284C7", "#38BDF8")
    draw_box(3.8, 2.4, 2.5, 1.5, "db.py (SQLite)", "Sessions, Sets, Aliases\nZero network calls", "LOCAL DB", "#064E3B", "#059669", "#34D399")

    # Column 3: Deterministic Core
    draw_box(6.9, 3.3, 2.7, 2.7, "metrics.py", "• Epley: w*(1+r/30)\n• Daily Top Sets\n• Leakage-Free Features\n• Strength Tiers (BW ratio)\n\n100% Deterministic Math", "DETERMINISTIC CORE", "#064E3B", "#10B981", "#A7F3D0")

    # Column 4: Forecaster & Stall Detection
    draw_box(10.2, 5.2, 2.5, 1.5, "forecast.py", "TabPFN on CPU\n95% Prediction Intervals", "IN-CONTEXT PRIOR", "#2E1065", "#7C3AED", "#A78BFA")
    draw_box(10.2, 2.4, 2.5, 1.5, "detect.py", "56-Day Slope < 0.0%/wk\nDiscrimination Ratio: 2.1x", "STALL DETECTOR", "#064E3B", "#059669", "#34D399")

    # Column 5: Outputs
    draw_box(13.2, 5.7, 2.1, 1.1, "app.py", "Streamlit UI\nDark Glassmorphic", "UI EXPLORER", "#1E293B", "#64748B", "#CBD5E1")
    draw_box(13.2, 4.4, 2.1, 1.1, "coach.py", "Gemma Narrator\nRegex Numeric Guard [AUDITED]", "EDGE NARRATOR", "#082F49", "#0284C7", "#38BDF8")
    draw_box(13.2, 3.1, 2.1, 1.1, "plate_loader.py", "Olympic Barbell Sleeve\n2.5kg Plate Breakdown", "ACTIONABLE", "#451A03", "#D97706", "#FBBF24")
    draw_box(13.2, 1.8, 2.1, 1.1, "voice.py", "10s Earbud Briefing\nElevenLabs + Browser TTS", "AUDIO COACH", "#451A03", "#D97706", "#FBBF24")

    # Connectors
    def connect(p1, p2, col="#38BDF8", style="-", rad=0.0):
        con = patches.ConnectionPatch(
            p1, p2, coordsA="data", coordsB="data",
            arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
            color=col, linestyle=style,
            connectionstyle=f"arc3,rad={rad}"
        )
        ax.add_patch(con)

    connect((3.2, 5.95), (3.8, 5.95), "#38BDF8")
    connect((5.05, 5.2), (5.05, 3.9), "#38BDF8")
    connect((3.2, 3.15), (3.8, 3.15), "#64748B", "--")
    connect((6.3, 3.15), (6.9, 4.2), "#34D399")
    connect((9.6, 5.3), (10.2, 5.8), "#A78BFA")
    connect((9.6, 4.0), (10.2, 3.3), "#34D399")
    connect((12.7, 5.95), (13.2, 4.95), "#A78BFA", rad=-0.1)
    connect((12.7, 3.15), (13.2, 4.8), "#34D399", rad=0.1)
    connect((12.7, 5.5), (13.2, 3.65), "#FBBF24", rad=0.15)
    connect((14.25, 4.4), (14.25, 3.9), "#FBBF24")

    # Bottom 3 Architecture Pillars
    def draw_pillar(x, y, w, h, dot_col, title, bullets):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1, edgecolor="#1E293B", facecolor="#0F172A")
        ax.add_patch(box)
        dot = patches.Circle((x + 0.25, y + h - 0.3), 0.08, facecolor=dot_col)
        ax.add_patch(dot)
        ax.text(x + 0.45, y + h - 0.35, title, fontsize=10, fontweight="bold", color="#F8FAFC", family="sans-serif")
        curr_y = y + h - 0.65
        for b in bullets:
            ax.text(x + 0.25, curr_y, f"• {b}", fontsize=8, color="#94A3B8", family="sans-serif")
            curr_y -= 0.25

    draw_pillar(0.5, 0.3, 4.7, 1.2, "#38BDF8", "100% Offline & Private by Design", [
        "Runs locally on laptop with 4 GB VRAM GPU",
        "Personal health notes stripped before SQLite storage",
        "Gemma runs via Ollama; TabPFN runs locally on CPU"
    ])
    draw_pillar(5.6, 0.3, 4.8, 1.2, "#34D399", "Deterministic Scientific Core", [
        "Epley e1RM and 56-day slopes are pure Python math",
        "40-point rolling origin evaluation prevents leakage",
        "AI strictly operates at boundaries (parse, forecast, narrate)"
    ])
    draw_pillar(10.8, 0.3, 4.7, 1.2, "#FBBF24", "Actionable Real-World Output", [
        "Inverts e1RM into 2.5kg plate loading sleeve visuals",
        "10s hands-free audio briefing for chalky gym hands",
        "Strict regex numeric guard eliminates hallucinations"
    ])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


def create_plate_loader_diagram(output_path: Path) -> None:
    """Generate high-resolution plate loader flow diagram PNG."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#080C14")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.6, 5.4, "INVERTED WORKING WEIGHT & BARBELL SLEEVE FLOW", fontsize=18, fontweight="bold", color="#F8FAFC")
    ax.text(0.6, 5.05, "Transforming abstract e1RM forecasts into gym-floor barbell loading (20kg bar + 2.5kg plates)", fontsize=10.5, color="#94A3B8")

    # Step Boxes
    def step_box(x, y, w, h, step_num, title, lines, edge_col, fill_col):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1.5, edgecolor=edge_col, facecolor=fill_col)
        ax.add_patch(box)
        ax.text(x + 0.2, y + h - 0.35, f"{step_num}. {title}", fontsize=11, fontweight="bold", color=edge_col)
        curr_y = y + h - 0.75
        for l in lines:
            ax.text(x + 0.2, curr_y, l, fontsize=9.5, color="#F8FAFC" if "kg" in l else "#94A3B8")
            curr_y -= 0.35

    step_box(0.6, 2.0, 3.0, 2.6, "1", "TabPFN Forecast", [
        "e1RM = 82.5 kg",
        "Target: 8 reps",
        "In-context zero-shot prior"
    ], "#8B5CF6", "#1E1B4B")

    step_box(4.2, 2.0, 3.2, 2.6, "2", "Epley Inversion", [
        "weight = e1RM / (1 + r/30)",
        "Raw: 65.13 kg",
        "Snap to 2.5kg: 65.0 kg"
    ], "#10B981", "#064E3B")

    step_box(8.0, 2.0, 3.1, 2.6, "3", "Barbell Sleeve Math", [
        "(65kg - 20kg bar) / 2",
        "= 22.5 kg per sleeve",
        "Greedy plate breakdown"
    ], "#F59E0B", "#451A03")

    # Sleeve Visual Graphic Box
    sleeve_box = patches.FancyBboxPatch((11.7, 2.0), 2.7, 2.6, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1.5, edgecolor="#38BDF8", facecolor="#0F172A")
    ax.add_patch(sleeve_box)
    ax.text(11.9, 4.25, "4. Visual Barbell Sleeve", fontsize=11, fontweight="bold", color="#38BDF8")

    # Draw Barbell Sleeve
    # Shaft
    ax.add_patch(patches.Rectangle((11.9, 3.1), 0.5, 0.25, facecolor="#64748B"))
    # Collar
    ax.add_patch(patches.Rectangle((12.4, 2.7), 0.15, 1.05, facecolor="#94A3B8"))
    # Sleeve
    ax.add_patch(patches.Rectangle((12.55, 3.15), 1.6, 0.15, facecolor="#CBD5E1"))
    # 20kg Blue Plate
    ax.add_patch(patches.Rectangle((12.65, 2.5), 0.25, 1.45, facecolor="#2563EB", edgecolor="#1D4ED8", linewidth=1))
    ax.text(12.77, 3.15, "20", fontsize=8, color="#FFFFFF", fontweight="bold", ha="center")
    # 2.5kg Green Plate
    ax.add_patch(patches.Rectangle((13.0, 2.85), 0.18, 0.75, facecolor="#16A34A", edgecolor="#15803D", linewidth=1))
    ax.text(13.09, 3.15, "2.5", fontsize=7, color="#FFFFFF", fontweight="bold", ha="center")

    ax.text(11.9, 2.3, "Per side: [20kg, 2.5kg]", fontsize=9, color="#34D399", fontweight="bold")

    # Connectors
    for x_start, x_end in [(3.6, 4.2), (7.4, 8.0), (11.1, 11.7)]:
        con = patches.ConnectionPatch((x_start, 3.3), (x_end, 3.3), coordsA="data", coordsB="data", arrowstyle="-|>", mutation_scale=14, linewidth=2, color="#F59E0B")
        ax.add_patch(con)

    # Bottom Gym Usability Card
    card = patches.FancyBboxPatch((0.6, 0.4), 13.8, 1.2, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1, edgecolor="#1E293B", facecolor="#111827")
    ax.add_patch(card)
    ax.text(0.9, 1.15, "GYM FLOOR USABILITY: ZERO TOUCH REQUIRED", fontsize=11, fontweight="bold", color="#FBBF24")
    ax.text(0.9, 0.82, "Armaan doesn't have to fiddle with phone sliders or mental arithmetic with chalky hands between heavy sets.", fontsize=9.5, color="#CBD5E1")
    ax.text(0.9, 0.55, "ElevenLabs voice coach speaks directly into his earbuds: 'Next bench target: 65 kg for 8 reps. Load one 20 and one 2.5 on each side.'", fontsize=9.5, color="#94A3B8")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


def create_evaluation_diagram(output_path: Path) -> None:
    """Generate high-resolution rolling-origin backtest protocol PNG."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#080C14")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.6, 5.4, "40-POINT ROLLING-ORIGIN BACKTEST PROTOCOL (§7.4)", fontsize=18, fontweight="bold", color="#F8FAFC")
    ax.text(0.6, 5.05, "Zero temporal data leakage • 8 out-of-sample test sessions across 5 core lifts", fontsize=10.5, color="#94A3B8")

    # Bad Split Box
    b_bad = patches.FancyBboxPatch((0.6, 0.8), 6.5, 3.9, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1.5, edgecolor="#EF4444", facecolor="#111827")
    ax.add_patch(b_bad)
    ax.text(0.9, 4.3, "[FAILED] RANDOM TRAIN/TEST SPLIT (TEMPORAL LEAKAGE)", fontsize=10.5, fontweight="bold", color="#F87171")
    ax.text(0.9, 3.95, "Future strength gains leak into past training sets.", fontsize=9, color="#9CA3AF")

    # Draw timeline with random dots
    ax.plot([1.0, 6.5], [3.2, 3.2], color="#374151", linewidth=3)
    x_pts = np.linspace(1.2, 6.3, 10)
    for i, pt in enumerate(x_pts):
        is_test = i in [1, 4, 7]
        col = "#EF4444" if is_test else "#3B82F6"
        ax.plot(pt, 3.2, marker="o", markersize=8, color=col)

    ax.text(1.0, 2.5, "• Blue = Train sessions, Red = Test sessions", fontsize=9, color="#EF4444", family="monospace")
    ax.text(1.0, 2.0, "A random 80/20 split uses Month 4 workouts to", fontsize=8.5, color="#94A3B8")
    ax.text(1.0, 1.7, "predict Month 2, destroying evaluation integrity.", fontsize=8.5, color="#94A3B8")
    ax.text(1.0, 1.2, "Result: Artificially low MAE that fails in the real gym.", fontsize=8.5, color="#FCA5A5")

    # Good Split Box (Rolling Origin)
    b_good = patches.FancyBboxPatch((7.6, 0.8), 6.8, 3.9, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1.5, edgecolor="#10B981", facecolor="#111827")
    ax.add_patch(b_good)
    ax.text(7.9, 4.3, "[VERIFIED] ROLLING-ORIGIN BACKTEST (STRICT CAUSALITY)", fontsize=10.5, fontweight="bold", color="#34D399")
    ax.text(7.9, 3.95, "Predict target session t using strictly history < t.", fontsize=9, color="#A7F3D0")

    # Draw 3 step bars
    y_steps = [3.3, 2.6, 1.9]
    train_lens = [2.5, 3.2, 3.9]
    for idx, (ys, tl) in enumerate(zip(y_steps, train_lens)):
        step_num = [1, 2, 8][idx]
        ax.text(7.9, ys - 0.05, f"t = {step_num}", fontsize=8.5, color="#94A3B8", family="monospace")
        ax.add_patch(patches.Rectangle((8.6, ys - 0.1), tl, 0.22, facecolor="#059669", alpha=0.85))
        ax.plot(8.6 + tl + 0.25, ys + 0.01, marker="o", markersize=8, color="#F59E0B")
        ax.text(8.6 + tl + 0.45, ys - 0.05, f"Predict t_{step_num}", fontsize=8, color="#FBBF24", family="monospace")

    b_card = patches.FancyBboxPatch((7.9, 1.05), 6.2, 0.65, boxstyle="round,pad=0.05,rounding_size=0.1", linewidth=1, edgecolor="#1E293B", facecolor="#0F172A")
    ax.add_patch(b_card)
    ax.text(8.1, 1.35, "• 5 Lifts × 8 Sessions = 40 strict out-of-sample predictions", fontsize=8.5, color="#F8FAFC")
    ax.text(8.1, 1.15, "• Zero data leakage • TabPFN compared honestly to baselines", fontsize=8.5, color="#34D399")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


def create_stall_diagram(output_path: Path) -> None:
    """Generate high-resolution empirical stall detection diagram PNG."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#080C14")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.6, 5.4, "STALL DETECTION: EMPIRICAL RULE BENCHMARK (§8)", fontsize=18, fontweight="bold", color="#F8FAFC")
    ax.text(0.6, 5.05, "Why the naive 'No PR in 3–4 weeks' rule failed • Least-squares linear slope over 56 days", fontsize=10.5, color="#94A3B8")

    # Left: Naive PR Rule
    b_bad = patches.FancyBboxPatch((0.6, 0.8), 6.5, 3.9, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1.5, edgecolor="#EF4444", facecolor="#111827")
    ax.add_patch(b_bad)
    ax.text(0.9, 4.3, "[REJECTED] NAIVE PR RULE: NO PR IN 21-28 DAYS", fontsize=10.5, fontweight="bold", color="#F87171")
    ax.text(0.9, 3.95, "Flags workout as stalled if no all-time best e1RM within 3–4 sessions.", fontsize=8.5, color="#9CA3AF")

    # Plot noisy upward trend with false alarm shading
    x = np.linspace(1.0, 6.5, 30)
    y = 1.8 + 0.3 * (x - 1.0) + 0.15 * np.sin(x * 5)
    ax.plot(x, y, color="#EF4444", linewidth=2)
    # Shaded false alarms
    ax.axvspan(2.2, 3.0, ymin=0.3, ymax=0.65, color="#EF4444", alpha=0.25)
    ax.axvspan(4.0, 4.8, ymin=0.45, ymax=0.75, color="#EF4444", alpha=0.25)
    ax.text(2.6, 2.8, "False Stall!", fontsize=8, color="#FCA5A5", ha="center", family="monospace")
    ax.text(4.4, 3.4, "False Stall!", fontsize=8, color="#FCA5A5", ha="center", family="monospace")

    b_bad_sub = patches.FancyBboxPatch((0.9, 1.05), 5.9, 0.7, boxstyle="round,pad=0.05,rounding_size=0.1", linewidth=1, edgecolor="#374151", facecolor="#0F172A")
    ax.add_patch(b_bad_sub)
    ax.text(1.1, 1.45, "• Bench press rose 42kg → 64kg, yet flagged 61% of workouts!", fontsize=8.5, color="#FCA5A5", family="monospace")
    ax.text(1.1, 1.2, "• PRs arrive in clusters; natural session noise is 2.8%–6.1%.", fontsize=8.5, color="#9CA3AF")

    # Right: 56-day LS slope rule
    b_good = patches.FancyBboxPatch((7.6, 0.8), 6.8, 3.9, boxstyle="round,pad=0.1,rounding_size=0.15", linewidth=1.5, edgecolor="#10B981", facecolor="#111827")
    ax.add_patch(b_good)
    ax.text(7.9, 4.3, "[CHOSEN] 56-DAY LEAST-SQUARES SLOPE RULE", fontsize=10.5, fontweight="bold", color="#34D399")
    ax.text(7.9, 3.95, "Fit slope of e1RM vs. weeks: flag if slope < 0.0%/wk (min 4 sessions).", fontsize=8.5, color="#A7F3D0")

    # Plot trend with regression fit
    x2 = np.linspace(8.0, 13.8, 30)
    y2 = 1.8 + 0.3 * (x2 - 8.0) + 0.15 * np.sin(x2 * 5)
    ax.plot(x2, y2, color="#10B981", linewidth=2)
    # Regression line
    ax.plot([8.0, 13.8], [1.8, 3.5], color="#34D399", linestyle="--", linewidth=1.8)
    ax.text(11.0, 3.3, "Slope = +0.8%/week (Progressing)", fontsize=8.5, color="#34D399", family="monospace")

    b_good_sub = patches.FancyBboxPatch((7.9, 1.05), 6.2, 0.7, boxstyle="round,pad=0.05,rounding_size=0.1", linewidth=1, edgecolor="#065F46", facecolor="#0F172A")
    ax.add_patch(b_good_sub)
    ax.text(8.1, 1.45, "• 2.1x discrimination ratio inside genuine plateaus", fontsize=8.5, color="#A7F3D0", family="monospace")
    ax.text(8.1, 1.2, "• Correctly ignores routine 1-week deloads (-10%) & single bad days (-15%).", fontsize=8.5, color="#9CA3AF")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    create_architecture_diagram(assets_dir / "architecture.png")
    create_plate_loader_diagram(assets_dir / "plate_loader_flow.png")
    create_evaluation_diagram(assets_dir / "evaluation_protocol.png")
    create_stall_diagram(assets_dir / "stall_detection.png")
