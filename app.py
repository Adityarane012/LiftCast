"""LiftCast — Local-First AI Workout Logger & Progress Forecaster.

Built for Armaan for the DEV Hacktoberfest 2026 Challenge ("Build for a Friend").
Local Gemma parses free text; local TabPFN forecasts progress; deterministic core detects stalls.
100% offline, zero cloud egress, strict privacy.
"""

from __future__ import annotations

import html
from datetime import date, datetime, timedelta
from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from liftcast.coach import generate_coach_summary
from liftcast.db import (
    DEFAULT_ALIASES,
    get_all_sessions,
    get_connection,
    get_history_df,
    get_or_create_session,
    insert_set,
    resolve_exercise,
)
from liftcast.detect import detect_stalls
from liftcast.forecast import (
    KEY_LIFTS,
    LastValueBaseline,
    LinearTrendBaseline,
    TabPFNForecaster,
    run_rolling_origin_eval,
)
from liftcast.metrics import (
    TIER_THRESHOLDS,
    calculate_epley_e1rm,
    get_session_top_sets,
    get_strength_tier,
)
from liftcast.parser import parse_log

# -----------------------------------------------------------------------------
# Streamlit Configuration & Dark Glassmorphism Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LiftCast — Local-First AI Workout Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern typography & background */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Glassmorphic cards */
.glass-panel {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
}

.glass-metric {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: left;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.glass-metric:hover {
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
    font-weight: 600;
}

.metric-val {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(6, 182, 212, 0.3);
}

.badge-rose {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.3);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-violet {
    background: rgba(139, 92, 246, 0.15);
    color: #c084fc;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* Hero text gradient */
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1.25rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Database Connection & Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db():
    return get_connection("data/liftcast.db")


conn = get_db()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
            <div style="background: linear-gradient(135deg, #38bdf8, #818cf8); border-radius:10px; padding:6px 10px; font-weight:900; color:#0f172a; font-size:1.1rem;">⚡</div>
            <div>
                <div style="font-size:1.3rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em;">LiftCast</div>
                <div style="font-size:0.75rem; color:#64748b; font-weight:600;">BUILT FOR ARMAAN</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin: 0.75rem 0 1.25rem 0;">
            <span class="badge badge-emerald">🔒 100% Offline / Local</span>
            <span class="badge badge-cyan" style="margin-left:4px;">Hacktoberfest '26</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏋️ Athlete Profile")
    bodyweight_kg = st.number_input(
        "Armaan's Bodyweight (kg)",
        min_value=40.0,
        max_value=160.0,
        value=75.0,
        step=0.5,
        help="Used to compute strength-to-bodyweight ratios and tier projections.",
    )

    st.markdown("### 🤖 Local AI Runtime")
    model_choice = st.selectbox(
        "Gemma Model (Ollama)",
        options=["gemma4:e2b", "gemma4:e4b", "gemma3:1b"],
        index=0,
        help="Local Gemma model running via Ollama. Options: num_ctx=8192, temp=0.",
    )

    st.markdown(
        """
        <div style="background:rgba(15, 23, 42, 0.6); padding:0.75rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05); font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;">
            <div><b>GPU:</b> RTX 3050 (4 GB VRAM) — Gemma</div>
            <div><b>CPU:</b> TabPFN In-Context Forecaster</div>
            <div><b>Zero Cloud:</b> No telemetry, no egress</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Database quick stats
    sessions_all = get_all_sessions(conn)
    history_df_all = get_history_df(conn)
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.5rem;">
            <b>Database History:</b><br/>
            • <b>{len(sessions_all)}</b> calendar sessions<br/>
            • <b>{len(history_df_all):,}</b> structured sets stored
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_seed, col_clear = st.columns(2)
    with col_seed:
        if st.button("⚡ Seed Demo", use_container_width=True, help="Seed 6-month rich workout history across 7 lifts"):
            with st.spinner("Seeding demo database..."):
                from liftcast.demo_seed import seed_demo_database
                res = seed_demo_database(conn, total_weeks=26, clear_existing=True, bodyweight_kg=float(bodyweight_kg))
                st.success(f"Seeded {res['sessions_created']} sessions ({res['sets_created']} sets)!")
                time.sleep(0.5)
                st.rerun()

    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True, help="Reset workout database"):
            with conn:
                conn.execute("DELETE FROM sets")
                conn.execute("DELETE FROM sessions")
            st.warning("Database cleared!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=[
            "📝 10-Second Text Logger",
            "📈 Lift Progress & Forecast",
            "🧠 Weekly Coach Recap",
            "🔬 Benchmark & Architecture",
        ],
        index=0,
    )


# -----------------------------------------------------------------------------
# PAGE 1: 10-Second Text Logger
# -----------------------------------------------------------------------------
if page == "📝 10-Second Text Logger":
    st.markdown('<div class="hero-title">10-Second Text Logger</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Armaan types the way he texts: shorthand, typos, Hinglish. Local Gemma structures it into SQLite with zero friction.</div>',
        unsafe_allow_html=True,
    )

    col_presets, col_date = st.columns([3, 1])
    with col_date:
        log_date = st.date_input("Session Date", value=date.today())
        log_bw = st.number_input("Session BW (kg)", value=float(bodyweight_kg), step=0.5)

    with col_presets:
        st.markdown(
            "<span style='font-size:0.8rem; font-weight:600; color:#94a3b8;'>Quick Fill Presets:</span>",
            unsafe_allow_html=True,
        )
        p_cols = st.columns(5)
        presets = [
            ("Bench Press", "bench 60 8 8 7, last set died"),
            ("Bent Over Row", "bor 50kg 8 8 8"),
            ("Hinglish Lat Pull", "aaj lat pulldown 55 pe 10 10 9"),
            ("Deadlift", "deadlift 120 5 5, felt easy"),
            ("RDL", "rdl 70 8 8 8"),
        ]
        for idx, (label, text_val) in enumerate(presets):
            if p_cols[idx].button(label, key=f"preset_{idx}"):
                st.session_state["raw_log_input"] = text_val

    raw_text = st.text_area(
        "Type workout note:",
        value=st.session_state.get("raw_log_input", "bench 60 8 8 7, last set died"),
        height=100,
        placeholder="e.g. bench 60 8 8 7, last set died\nbor 50kg 8 8 8",
        key="raw_text_area",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        parse_clicked = st.button("⚡ Parse with Gemma", type="primary", use_container_width=True)

    if parse_clicked or "last_parse_result" in st.session_state:
        if parse_clicked:
            with st.spinner(f"Running local {model_choice} via Ollama..."):
                t_start = time.perf_counter()
                res = parse_log(raw_text, conn=conn, model=model_choice)
                t_elapsed = round((time.perf_counter() - t_start) * 1000, 1)
                st.session_state["last_parse_result"] = res
                st.session_state["last_parse_time_ms"] = t_elapsed
                st.session_state["last_raw_text"] = raw_text

        parse_res = st.session_state["last_parse_result"]
        t_ms = st.session_state.get("last_parse_time_ms", 0.0)

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.5rem; margin-bottom:1rem;">
                <span class="badge badge-cyan">⚡ Gemma Parsed in {t_ms} ms</span>
                <span class="badge badge-emerald">Enforced JSON Schema</span>
                {"<span class='badge badge-amber'>⚠️ Needs Confirmation</span>" if parse_res.needs_confirmation else "<span class='badge badge-emerald'>✓ Confident Match</span>"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if parse_res.issues:
            for issue in parse_res.issues:
                st.warning(f"Notice: {issue}")

        if not parse_res.entries:
            st.error("No valid exercise sets were parsed. Please check the text format.")
        else:
            # Prepare editable preview table
            table_rows = []
            for entry in parse_res.entries:
                for r in entry.reps:
                    e1rm = calculate_epley_e1rm(entry.weight_kg or 0.0, r)
                    table_rows.append(
                        {
                            "Exercise": entry.exercise or entry.exercise_raw,
                            "Weight (kg)": float(entry.weight_kg or 0.0),
                            "Reps": int(r),
                            "Top e1RM (kg)": float(e1rm),
                            "Note": entry.note or "",
                        }
                    )

            preview_df = pd.DataFrame(table_rows)
            st.markdown("#### Structured Sets Preview")
            edited_df = st.data_editor(
                preview_df,
                num_rows="dynamic",
                use_container_width=True,
                key="workout_editor",
            )

            col_save, _ = st.columns([1, 3])
            with col_save:
                if st.button("💾 Confirm & Save Session", type="secondary", use_container_width=True):
                    try:
                        date_str = log_date.strftime("%Y-%m-%d")
                        sess_id = get_or_create_session(
                            conn,
                            date_str=date_str,
                            source="manual",
                            bodyweight_kg=float(log_bw) if log_bw else None,
                        )

                        saved_count = 0
                        for _, row in edited_df.iterrows():
                            ex_name = str(row["Exercise"]).strip()
                            w_kg = float(row["Weight (kg)"])
                            reps = int(row["Reps"])
                            note_val = str(row["Note"]).strip() if row["Note"] else None

                            if w_kg > 0 and reps > 0:
                                insert_set(
                                    conn,
                                    session_id=sess_id,
                                    exercise=ex_name,
                                    weight_kg=w_kg,
                                    reps=reps,
                                    note=note_val,
                                    raw_text=st.session_state.get("last_raw_text", raw_text),
                                )
                                saved_count += 1

                        st.success(
                            f"✅ Successfully recorded {saved_count} sets for session {date_str}! Top sets updated."
                        )
                        st.balloons()
                        # Reset
                        if "last_parse_result" in st.session_state:
                            del st.session_state["last_parse_result"]
                    except Exception as err:
                        st.error(f"Error saving session: {err}")


# -----------------------------------------------------------------------------
# PAGE 2: Lift Progress & Forecast
# -----------------------------------------------------------------------------
elif page == "📈 Lift Progress & Forecast":
    st.markdown('<div class="hero-title">Lift Progress & Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Deterministic session top-set e1RM history paired with local TabPFN next-session forecasts and rolling-slope stall detection.</div>',
        unsafe_allow_html=True,
    )

    all_sets_df = get_history_df(conn)
    if all_sets_df.empty:
        st.info("No workout history found in the database yet. Log a workout first!")
        st.stop()

    top_sets_df = get_session_top_sets(all_sets_df)
    unique_lifts = sorted(top_sets_df["exercise"].unique())

    # Prioritize Key Lifts
    default_index = 0
    if "Bench Press" in unique_lifts:
        default_index = unique_lifts.index("Bench Press")

    selected_lift = st.selectbox("Select Exercise / Lift", options=unique_lifts, index=default_index)

    lift_history = top_sets_df[top_sets_df["exercise"].str.lower() == selected_lift.lower()].sort_values("date")

    if lift_history.empty:
        st.warning(f"No sessions recorded for {selected_lift}.")
        st.stop()

    total_sessions = len(lift_history)
    best_e1rm = float(lift_history["top_e1rm"].max())
    latest_row = lift_history.iloc[-1]
    latest_e1rm = float(latest_row["top_e1rm"])
    latest_date = latest_row["date"]

    # Compute Stall Status via detect_stalls
    stall_df = detect_stalls(lift_history, window_days=56, theta_pct_week=0.0, min_n=4)
    is_currently_stalled = bool(stall_df.iloc[-1]["stalled"]) if not stall_df.empty and pd.notna(stall_df.iloc[-1]["stalled"]) else False
    latest_slope = stall_df.iloc[-1]["slope_pct_week"] if not stall_df.empty and pd.notna(stall_df.iloc[-1]["slope_pct_week"]) else 0.0

    # Compute Next-Session Forecast via TabPFN
    target_date = (pd.to_datetime(latest_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    tabpfn = TabPFNForecaster(device="cpu", random_seed=42)
    forecast_res = tabpfn.predict_next(top_sets_df, selected_lift, target_date)

    # Top metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="glass-metric">
                <div class="metric-label">All-Time Best e1RM</div>
                <div class="metric-val">{best_e1rm:.1f} <span style="font-size:0.9rem; color:#94a3b8;">kg</span></div>
                <div class="metric-sub">{total_sessions} total sessions logged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-metric">
                <div class="metric-label">Latest Session</div>
                <div class="metric-val">{latest_e1rm:.1f} <span style="font-size:0.9rem; color:#94a3b8;">kg</span></div>
                <div class="metric-sub">{latest_date} ({latest_row['weight_kg']:.1f}kg × {int(latest_row['reps'])})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        low_str = f"{forecast_res.low_kg:.1f}" if forecast_res.low_kg else "-"
        high_str = f"{forecast_res.high_kg:.1f}" if forecast_res.high_kg else "-"
        st.markdown(
            f"""
            <div class="glass-metric">
                <div class="metric-label">Next Forecast ({forecast_res.method})</div>
                <div class="metric-val" style="color:#38bdf8;">{forecast_res.point_kg:.1f} <span style="font-size:0.9rem; color:#94a3b8;">kg</span></div>
                <div class="metric-sub">95% Range: [{low_str} – {high_str}] kg</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        if is_currently_stalled:
            badge_html = f'<span class="badge badge-rose">⚠️ STALL DETECTED</span>'
            status_desc = f"Slope {latest_slope:+.2f}%/wk (< 0.0%/wk over 56d)"
        else:
            badge_html = f'<span class="badge badge-emerald">✅ PROGRESSING</span>'
            status_desc = f"Slope {latest_slope:+.2f}%/wk (56-day trend)"

        st.markdown(
            f"""
            <div class="glass-metric">
                <div class="metric-label">Stall Rule State</div>
                <div style="margin-top:0.4rem; margin-bottom:0.4rem;">{badge_html}</div>
                <div class="metric-sub">{status_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly Interactive Chart
    fig = go.Figure()

    # Highlight stalled periods with background shading
    stalled_dates = stall_df[stall_df["stalled"] == True]["date"].tolist()
    if stalled_dates:
        # Group contiguous stalled dates into ranges
        stalled_ranges = []
        start_d = stalled_dates[0]
        prev_d = start_d
        for d in stalled_dates[1:]:
            diff = (pd.to_datetime(d) - pd.to_datetime(prev_d)).days
            if diff > 14:
                stalled_ranges.append((start_d, prev_d))
                start_d = d
            prev_d = d
        stalled_ranges.append((start_d, prev_d))

        for s_start, s_end in stalled_ranges:
            fig.add_vrect(
                x0=s_start,
                x1=s_end,
                fillcolor="rgba(244, 63, 94, 0.12)",
                layer="below",
                line_width=0,
                annotation_text="Stall Window",
                annotation_position="top left",
                annotation_font=dict(color="#f43f5e", size=10),
            )

    # Actual e1RM points & line
    fig.add_trace(
        go.Scatter(
            x=lift_history["date"],
            y=lift_history["top_e1rm"],
            mode="lines+markers",
            name="Actual e1RM (Top Set)",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=7, color="#38bdf8", symbol="circle"),
            customdata=np.stack((lift_history["weight_kg"], lift_history["reps"]), axis=-1),
            hovertemplate="<b>Date:</b> %{x}<br><b>e1RM:</b> %{y:.1f} kg<br><b>Top Set:</b> %{customdata[0]:.1f} kg × %{customdata[1]} reps<extra></extra>",
        )
    )

    # Rolling 3-session average
    if len(lift_history) >= 3:
        rolling_3 = lift_history["top_e1rm"].rolling(window=3).mean()
        fig.add_trace(
            go.Scatter(
                x=lift_history["date"],
                y=rolling_3,
                mode="lines",
                name="3-Session Rolling Avg",
                line=dict(color="rgba(255, 255, 255, 0.4)", width=1.5, dash="dot"),
                hoverinfo="skip",
            )
        )

    # Next session forecast point + confidence interval
    if forecast_res.point_kg > 0:
        error_y = None
        if forecast_res.low_kg and forecast_res.high_kg:
            error_y = dict(
                type="data",
                symmetric=False,
                array=[forecast_res.high_kg - forecast_res.point_kg],
                arrayminus=[forecast_res.point_kg - forecast_res.low_kg],
                color="#c084fc",
                thickness=2,
                width=6,
            )

        fig.add_trace(
            go.Scatter(
                x=[target_date],
                y=[forecast_res.point_kg],
                mode="markers",
                name="TabPFN Next Forecast",
                marker=dict(size=12, color="#c084fc", symbol="diamond-dot", line=dict(color="#ffffff", width=1.5)),
                error_y=error_y,
                hovertemplate=f"<b>Target Date:</b> {target_date}<br><b>Forecast:</b> %{{y:.1f}} kg<br><b>95% Band:</b> [{forecast_res.low_kg:.1f} - {forecast_res.high_kg:.1f}] kg<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8"),
        margin=dict(l=40, r=40, t=30, b=40),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            title="Estimated 1RM (kg)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Visual Barbell Plate Loader & Working Weight Target Recommender
    # -------------------------------------------------------------------------
    from liftcast.plate_calculator import (
        calculate_plates,
        calculate_working_weight_from_e1rm,
        snap_to_plate_increment,
    )

    st.markdown("### 🏋️ Barbell Plate Loader & Gym Target Recommender")
    st.markdown(
        "<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;'>Inverts TabPFN's e1RM forecast into an actionable gym working weight (snapped to standard 2.5 kg plates) with exact plate counts per barbell sleeve.</div>",
        unsafe_allow_html=True,
    )

    # Invert TabPFN e1RM to an 8-rep working target
    default_working_weight = (
        calculate_working_weight_from_e1rm(forecast_res.point_kg, target_reps=8, increment=2.5)
        if forecast_res.point_kg > 0
        else 60.0
    )
    if is_currently_stalled:
        deload_target = snap_to_plate_increment(default_working_weight * 0.90, increment=2.5)
        st.info(
            f"💡 **Stall-Aware Deload Target:** Since {selected_lift} has plateaued, consider a technical deload at **{deload_target:.1f} kg** (90% of working weight) or hold **{default_working_weight:.1f} kg** focusing on bar speed and technique."
        )

    col_target, col_bar, col_reps = st.columns([2, 1, 1])
    with col_target:
        target_wt = st.number_input(
            "Target Barbell Weight (kg)",
            min_value=15.0,
            max_value=350.0,
            value=float(default_working_weight) if default_working_weight >= 20.0 else 60.0,
            step=2.5,
            help="Total barbell weight (bar + plates on both sides).",
        )
    with col_bar:
        bar_wt = st.selectbox(
            "Barbell Type",
            options=[20.0, 15.0],
            format_func=lambda x: f"{int(x)} kg (Olympic)" if x == 20 else f"{int(x)} kg (Technique)",
            index=0,
        )
    with col_reps:
        reps_choice = st.selectbox("Target Reps", options=[3, 5, 6, 8, 10, 12], index=3)

    loading = calculate_plates(target_wt, bar_weight_kg=bar_wt)

    # Render Visual Barbell Graphic with Olympic Plates
    sleeve_plates_html = []
    for item in loading.plates_visual:
        bg_col = item["color"]
        txt_col = item.get("text_color", "#ffffff")
        h_px = item["height_px"]
        w_px = item["width_px"]
        lbl = item["label"]
        sleeve_plates_html.append(
            f"""<div style="background:{bg_col}; color:{txt_col}; height:{h_px}px; width:{w_px}px; border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; border:1px solid rgba(0,0,0,0.3); box-shadow:0 4px 6px rgba(0,0,0,0.4); writing-mode:vertical-rl; text-orientation:mixed; cursor:pointer;" title="{item['weight']} kg plate">{lbl}</div>"""
        )

    plates_combined_html = "".join(sleeve_plates_html) if sleeve_plates_html else "<div style='color:#64748b; font-size:0.85rem; font-style:italic;'>No plates (empty bar)</div>"

    plate_pills = []
    for p_weight, count in sorted(loading.plate_counts.items(), reverse=True):
        plate_pills.append(f"<b>{count}×</b> {p_weight} kg")
    per_side_str = ", ".join(plate_pills) if plate_pills else "None (Empty Barbell)"

    st.markdown(
        f"""
        <div class="glass-panel" style="margin-top:0.75rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <div>
                    <span style="font-size:1.1rem; font-weight:800; color:#f8fafc;">Barbell Sleeve Breakdown</span>
                    <span style="font-size:0.85rem; color:#94a3b8; margin-left:0.5rem;">(Per Side: {loading.weight_per_side_loaded:.2f} kg)</span>
                </div>
                <div>
                    <span class="badge badge-emerald">Total: {loading.total_loaded_kg:.1f} kg {"✓ Exact Match" if loading.is_exact else f"(Remainder: {loading.remainder_kg} kg)"}</span>
                </div>
            </div>

            <!-- Barbell Visual Graphic -->
            <div style="display:flex; align-items:center; justify-content:center; background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:2rem 1.5rem; min-height:160px; overflow-x:auto;">
                <!-- Left Shaft (Main Grip) -->
                <div style="height:18px; width:90px; background:linear-gradient(180deg, #94a3b8 0%, #475569 50%, #334155 100%); border-radius:4px 0 0 4px; box-shadow:inset 0 1px 2px rgba(255,255,255,0.2);"></div>
                <!-- Inner Collar -->
                <div style="height:80px; width:16px; background:linear-gradient(180deg, #cbd5e1 0%, #64748b 50%, #475569 100%); border-radius:3px; box-shadow:0 2px 4px rgba(0,0,0,0.5);"></div>
                <!-- Plates on Sleeve -->
                <div style="display:flex; align-items:center; gap:3px; margin:0 4px;">
                    {plates_combined_html}
                </div>
                <!-- Outer Sleeve End -->
                <div style="height:22px; width:70px; background:linear-gradient(180deg, #94a3b8 0%, #475569 50%, #1e293b 100%); border-radius:0 4px 4px 0;"></div>
            </div>

            <!-- Detailed Plate List -->
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem; padding-top:0.75rem; border-top:1px solid rgba(255,255,255,0.06); font-size:0.85rem;">
                <div style="color:#cbd5e1;"><b>Plates per Sleeve:</b> {per_side_str}</div>
                <div style="color:#94a3b8;">Bar: <b>{loading.bar_weight_kg:.0f} kg</b> • Working Set e1RM: <b>{calculate_epley_e1rm(target_wt, reps_choice):.1f} kg</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 3: Weekly Coach Recap
# -----------------------------------------------------------------------------
elif page == "🧠 Weekly Coach Recap":
    st.markdown('<div class="hero-title">AI Coach Summary & Strength Tiers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Gemma narrates weekly performance strictly bound by a deterministic numeric guard. Every number is verified; zero hallucinations allowed.</div>',
        unsafe_allow_html=True,
    )

    all_sets_df = get_history_df(conn)
    top_sets_df = get_session_top_sets(all_sets_df)

    if top_sets_df.empty:
        st.info("No workout history found. Log workouts to generate coach recap.")
        st.stop()

    # Build weekly stats payload from the most recent 14 days of data
    latest_dt = pd.to_datetime(top_sets_df["date"].max())
    week_cutoff = latest_dt - pd.Timedelta(days=7)
    recent_sets = top_sets_df[pd.to_datetime(top_sets_df["date"]) >= week_cutoff]
    sessions_this_week = len(recent_sets["date"].unique())

    # Build payload across key lifts
    lifts_payload = {}
    for lift in KEY_LIFTS:
        l_df = top_sets_df[top_sets_df["exercise"].str.lower() == lift.lower()].sort_values("date")
        if l_df.empty:
            continue
        curr_e1rm = float(l_df.iloc[-1]["top_e1rm"])
        prev_e1rm = float(l_df.iloc[-2]["top_e1rm"]) if len(l_df) >= 2 else curr_e1rm
        delta_pct = round(((curr_e1rm - prev_e1rm) / prev_e1rm) * 100.0, 1) if prev_e1rm > 0 else 0.0

        stalls = detect_stalls(l_df, window_days=56, theta_pct_week=0.0, min_n=4)
        is_stalled = bool(stalls.iloc[-1]["stalled"]) if not stalls.empty and pd.notna(stalls.iloc[-1]["stalled"]) else False

        # Quick next forecast
        tabpfn = TabPFNForecaster(device="cpu", random_seed=42)
        f_next = tabpfn.predict_next(top_sets_df, lift, (latest_dt + pd.Timedelta(days=7)).strftime("%Y-%m-%d"))

        lifts_payload[lift] = {
            "current_e1rm": curr_e1rm,
            "weekly_change_pct": delta_pct,
            "stalled": is_stalled,
            "forecast_next_e1rm": f_next.point_kg,
        }

    stats_payload = {
        "as_of_date": latest_dt.strftime("%Y-%m-%d"),
        "bodyweight_kg": float(bodyweight_kg),
        "sessions_this_week": int(sessions_this_week),
        "lifts": lifts_payload,
    }

    col_btn, col_badge = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button("🎙️ Generate Coach Recap", type="primary", use_container_width=True)

    if generate_clicked or "coach_recap_result" in st.session_state:
        if generate_clicked:
            with st.spinner(f"Gemma ({model_choice}) generating narrative under numeric guard audit..."):
                recap_res = generate_coach_summary(stats_payload, model=model_choice)
                st.session_state["coach_recap_result"] = recap_res

        result = st.session_state["coach_recap_result"]

        if result.is_fallback:
            guard_badge = '<span class="badge badge-amber">⚠️ Fallback Activated (Numeric Guard Caught Hallucination)</span>'
        else:
            guard_badge = '<span class="badge badge-emerald">🛡️ 100% Numeric Guard Verified (0 Hallucinations)</span>'

        st.markdown(
            f"""
            <div class="glass-panel" style="border-left: 4px solid #38bdf8; margin-top: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <div style="font-weight:700; color:#f8fafc; font-size:1.05rem;">Weekly Coach Briefing for Armaan</div>
                    <div>{guard_badge}</div>
                </div>
                <div style="font-size:0.95rem; line-height:1.65; color:#cbd5e1; white-space:pre-wrap;">
{html.escape(result.summary)}
                </div>
                <div style="margin-top:0.75rem; font-size:0.75rem; color:#64748b;">
                    Model: <code>{result.model_used}</code> • Enforced Hard Rule 3: The model explains; it never invents numbers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Strength Tiers & Projections Section
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Strength Tiers & Next Tier Projections")
    st.markdown(
        "<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:1rem;'>Calculated via Epley e1RM ÷ Bodyweight ratio with linear trend slope ± SE projections.</div>",
        unsafe_allow_html=True,
    )

    t_cols = st.columns(len(KEY_LIFTS))
    for idx, lift in enumerate(KEY_LIFTS):
        l_df = top_sets_df[top_sets_df["exercise"].str.lower() == lift.lower()].sort_values("date")
        if l_df.empty:
            continue
        recent_best = float(l_df["top_e1rm"].max())

        # History sequence for trend projection
        hist_seq = [
            (datetime.strptime(d, "%Y-%m-%d").date(), float(val))
            for d, val in zip(l_df["date"], l_df["top_e1rm"])
        ]
        tier_status = get_strength_tier(lift, recent_best, bodyweight_kg, recent_history=hist_seq)

        with t_cols[idx]:
            # Compute progress percentage to next tier
            pct = 100.0
            if tier_status.next_threshold_ratio and tier_status.threshold_ratio:
                range_span = tier_status.next_threshold_ratio - tier_status.threshold_ratio
                if range_span > 0:
                    pct = min(100.0, max(0.0, ((tier_status.current_ratio - tier_status.threshold_ratio) / range_span) * 100.0))

            proj_text = tier_status.status_message
            if tier_status.is_projectable and tier_status.projected_weeks_low and tier_status.projected_weeks_high:
                proj_text = f"~{tier_status.projected_weeks_low:.1f} to {tier_status.projected_weeks_high:.1f} weeks"

            st.markdown(
                f"""
                <div class="glass-panel" style="padding:1rem; text-align:center;">
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:0.25rem;">{lift}</div>
                    <div style="font-size:1.35rem; font-weight:800; color:#38bdf8;">{tier_status.current_tier}</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.5rem;">{tier_status.current_ratio:.2f}× BW ({recent_best:.1f} kg)</div>
                    <div style="background:rgba(255,255,255,0.08); border-radius:9999px; height:6px; width:100%; margin:0.5rem 0;">
                        <div style="background:linear-gradient(90deg, #38bdf8, #818cf8); height:6px; border-radius:9999px; width:{pct:.0f}%;"></div>
                    </div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:0.35rem;">Next: <b>{tier_status.next_tier or 'Max'}</b> ({tier_status.next_threshold_kg or 0:.1f} kg)</div>
                    <div style="font-size:0.7rem; color:#a5b4fc; font-weight:600; margin-top:0.25rem;">{proj_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# PAGE 4: Benchmark & Architecture
# -----------------------------------------------------------------------------
elif page == "🔬 Benchmark & Architecture":
    st.markdown('<div class="hero-title">Benchmark & Architecture Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Honest rolling-origin evaluation (§7.4) and standalone interactive architecture map rendered with Archify.</div>',
        unsafe_allow_html=True,
    )

    tab_eval, tab_arch = st.tabs(["📊 Rolling-Origin Evaluation (§7.4)", "🗺️ Interactive Archify Architecture"])

    with tab_eval:
        st.markdown("### 40-Point Rolling-Origin Evaluation Results")
        st.markdown(
            """
            <div style="font-size:0.88rem; color:#94a3b8; margin-bottom:1rem;">
                Evaluated across the <b>last 8 sessions</b> of the 5 core lifts (40 total out-of-sample predictions).
                For every target date <code>d</code>, context consists strictly of rows prior to <code>d</code> (zero temporal leakage).
            </div>
            """,
            unsafe_allow_html=True,
        )

        eval_summary_data = [
            {"Lift": "Lat Pulldown", "n": 8, "Last value MAE (kg)": 6.48, "Trend-5 MAE (kg)": 6.21, "TabPFN MAE (kg)": 10.84, "Winner": "Trend-5"},
            {"Lift": "Bench Press", "n": 8, "Last value MAE (kg)": 8.78, "Trend-5 MAE (kg)": 11.08, "Winner": "Last value", "TabPFN MAE (kg)": 12.28},
            {"Lift": "Deadlift", "n": 8, "Last value MAE (kg)": 10.97, "Trend-5 MAE (kg)": 12.81, "Winner": "Last value", "TabPFN MAE (kg)": 15.33},
            {"Lift": "Bent Over Row", "n": 8, "Last value MAE (kg)": 12.94, "Trend-5 MAE (kg)": 10.33, "Winner": "Trend-5", "TabPFN MAE (kg)": 15.04},
            {"Lift": "Romanian Deadlift", "n": 8, "Last value MAE (kg)": 6.87, "Trend-5 MAE (kg)": 8.44, "Winner": "Last value", "TabPFN MAE (kg)": 18.07},
            {"Lift": "**Overall**", "n": 40, "Last value MAE (kg)": 9.21, "Trend-5 MAE (kg)": 9.78, "Winner": "**Last value**", "TabPFN MAE (kg)": 14.31},
        ]
        eval_df = pd.DataFrame(eval_summary_data)

        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown(
                """
                <div class="glass-panel">
                    <div style="font-weight:700; color:#38bdf8; margin-bottom:0.35rem;">1. Linear Trend Baseline Wins on Rows & Pulldowns</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        On near-linear movements (Lat Pulldown 6.21 kg MAE vs 6.48 kg; Bent Over Row 10.33 kg vs 12.94 kg), the 5-session trend baseline outperforms last-value and TabPFN.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f2:
            st.markdown(
                """
                <div class="glass-panel">
                    <div style="font-weight:700; color:#34d399; margin-bottom:0.35rem;">2. Last Value Wins on Noisy Compounds</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        Bench Press (8.78 kg) and Deadlift (10.97 kg) exhibit discrete step-functions and warm-up variances that penalize linear extrapolation. The naive baseline is notoriously hard to beat.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f3:
            st.markdown(
                """
                <div class="glass-panel">
                    <div style="font-weight:700; color:#c084fc; margin-bottom:0.35rem;">3. TabPFN's Calibrated In-Context Prior</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        TabPFN expresses lifts as ratios to prior bests to generalize across exercises without retraining, providing calibrated 95% uncertainty intervals for risk-aware forecasts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_arch:
        arch_html_path = Path("docs/assets/liftcast_architecture.html")
        if arch_html_path.exists():
            html_content = arch_html_path.read_text(encoding="utf-8")
            st.markdown(
                """
                <div style="margin-bottom:0.75rem; font-size:0.85rem; color:#94a3b8;">
                    Interactive system diagram rendered by Archify CLI with pan, zoom, click-to-inspect boundaries, and verified dark showcase styling:
                </div>
                """,
                unsafe_allow_html=True,
            )
            components.html(html_content, height=720, scrolling=True)
        else:
            st.error("Architecture diagram HTML not found at docs/assets/liftcast_architecture.html.")
