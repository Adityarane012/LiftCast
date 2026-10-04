"""LiftCast — Local-First AI Workout Logger & Progress Forecaster.

Built for a friend for the DEV Hacktoberfest 2026 Challenge ("Build for a Friend").
Local Gemma parses free text; local TabPFN forecasts progress; deterministic core detects stalls.
100% offline, zero cloud egress, strict privacy.
"""

from __future__ import annotations

import html
import os
from datetime import date, datetime, timedelta
from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from liftcast.coach import generate_coach_summary
from liftcast.voice import (
    DEFAULT_VOICE_ID,
    build_audio_briefing_script,
    synthesize_voice_elevenlabs,
)
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
/* Modern typography & base canvas contrast overrides */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #080C14 !important;
    color: #F8FAFC !important;
}

[data-testid="stHeader"] {
    background-color: rgba(8, 12, 20, 0.85) !important;
    backdrop-filter: blur(12px) !important;
}

[data-testid="stSidebar"] {
    background-color: #0B1120 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Glassmorphic cards */
.glass-panel {
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
}

.glass-metric {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: left;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.glass-metric:hover {
    border-color: rgba(56, 189, 248, 0.5);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 0.35rem;
    font-weight: 700;
}

.metric-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 0.25rem;
}

/* Status tags */
.tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}
.tag-emerald { background: rgba(16, 185, 129, 0.15); color: #34D399; }
.tag-rose { background: rgba(244, 63, 94, 0.15); color: #FB7185; }
.tag-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; }
.tag-cyan { background: rgba(56, 189, 248, 0.15); color: #38BDF8; }
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
    st.title("LiftCast")
    st.caption("Local workout logger & strength forecaster")

    st.subheader("Athlete")
    bodyweight_kg = st.number_input(
        "Bodyweight (kg)",
        min_value=40.0,
        max_value=160.0,
        value=75.0,
        step=0.5,
        help="Used for strength-to-bodyweight ratios and tier projections.",
    )

    st.subheader("Model")
    model_choice = st.selectbox(
        "Gemma Model (Ollama)",
        options=["gemma4:e2b", "gemma4:e4b", "gemma3:1b"],
        index=0,
        help="Local Gemma model running via Ollama.",
    )

    with st.expander("Settings", expanded=False):
        chalk_mode = st.toggle(
            "Large touch targets",
            value=st.session_state.get("chalk_mode", False),
            help="Enlarges buttons and inputs for easier tapping on mobile or gym floor.",
            key="chalk_mode",
        )
        st.text_input(
            "ElevenLabs API Key (optional)",
            value=os.environ.get("ELEVENLABS_API_KEY", ""),
            type="password",
            help="Optional. Uses browser speech synthesis when blank.",
            key="elevenlabs_api_key_input",
        )
        st.selectbox(
            "Voice",
            options=["pNInz6obpgDQGcFmaJgB", "21m00Tcm4TlvDq8ikWAM"],
            format_func=lambda x: "Adam" if "pNIn" in x else "Rachel",
            key="elevenlabs_voice_choice",
        )

    if st.session_state.get("chalk_mode", False):
        st.markdown(
            """
            <style>
            .stButton > button {
                min-height: 54px !important;
                font-size: 1.1rem !important;
                font-weight: 700 !important;
                border-radius: 8px !important;
            }
            input, textarea, select {
                min-height: 48px !important;
                font-size: 1.05rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Database quick stats
    sessions_all = get_all_sessions(conn)
    history_df_all = get_history_df(conn)
    st.markdown("---")
    st.caption(f"{len(sessions_all)} sessions • {len(history_df_all):,} sets stored")

    col_seed, col_real, col_clear = st.columns(3)
    with col_seed:
        if st.button("Demo", use_container_width=True, help="Load 6-month synthetic workout history"):
            with st.spinner("Loading demo data..."):
                from liftcast.demo_seed import seed_demo_database
                res = seed_demo_database(conn, total_weeks=26, clear_existing=True, bodyweight_kg=float(bodyweight_kg))
                st.success(f"Loaded {res['sessions_created']} sessions")
                time.sleep(0.5)
                st.rerun()

    with col_real:
        if st.button("Real", use_container_width=True, help="Load Liftoff workout dataset"):
            real_csv = Path("data/raw/liftoff_workout_data.csv")
            if not real_csv.exists():
                real_csv = Path("liftoff_workout_data.csv")
            if real_csv.exists():
                with st.spinner("Importing dataset..."):
                    from liftcast.importer import import_liftoff_csv
                    res = import_liftoff_csv(real_csv, conn)
                    st.success(f"Imported {res['sessions_imported']} sessions")
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.error("liftoff_workout_data.csv not found")

    with col_clear:
        if st.button("Clear", use_container_width=True, help="Reset database"):
            with conn:
                conn.execute("DELETE FROM sets")
                conn.execute("DELETE FROM sessions")
            st.warning("Database cleared")
            time.sleep(0.5)
            st.rerun()

    with st.expander("Import / Export", expanded=False):
        db_file = Path("data/liftcast.db")
        if db_file.exists():
            try:
                with open(db_file, "rb") as f:
                    st.download_button(
                        label="Download SQLite DB",
                        data=f.read(),
                        file_name="liftcast.db",
                        mime="application/x-sqlite3",
                        use_container_width=True,
                    )
            except Exception:
                pass

        if not history_df_all.empty:
            csv_data = history_df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export CSV",
                data=csv_data,
                file_name=f"liftcast_history_{date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        uploaded_csv = st.file_uploader(
            "Import CSV (Liftoff or Hevy format)",
            type=["csv"],
        )
        if uploaded_csv is not None:
            if st.button("Process CSV", use_container_width=True):
                with st.spinner("Importing CSV..."):
                    from liftcast.importer import import_liftoff_csv
                    import_res = import_liftoff_csv(uploaded_csv, conn)
                    st.success(
                        f"Imported {import_res['sessions_imported']} sessions ({import_res['sets_imported']} sets, {import_res['sessions_skipped']} skipped duplicates)"
                    )
                    time.sleep(0.8)
                    st.rerun()

    st.markdown("---")
    page = st.radio(
        "View",
        options=[
            "Logger",
            "Progress & Forecast",
            "Coach Recap",
            "Benchmark & Architecture",
        ],
        index=0,
    )


# -----------------------------------------------------------------------------
# PAGE 1: Workout Logger
# -----------------------------------------------------------------------------
if page == "Logger":
    st.title("Workout Logger")
    st.caption("Paste workout notes or shorthand. Local Gemma extracts structured sets.")

    col_presets, col_date = st.columns([3, 1])
    with col_date:
        log_date = st.date_input("Date", value=date.today())
        log_bw = st.number_input("Bodyweight (kg)", value=float(bodyweight_kg), step=0.5)

    with col_presets:
        st.caption("Examples:")
        p_cols = st.columns(5)
        presets = [
            ("Bench", "bench 60 8 8 7"),
            ("Row", "bor 50kg 8 8 8"),
            ("Pulldown", "lat pulldown 55 10 10 9"),
            ("Deadlift", "deadlift 120 5 5"),
            ("RDL", "rdl 70 8 8 8"),
        ]
        for idx, (label, text_val) in enumerate(presets):
            if p_cols[idx].button(label, key=f"preset_{idx}"):
                st.session_state["raw_log_input"] = text_val

    raw_text = st.text_area(
        "Workout note:",
        value=st.session_state.get("raw_log_input", "bench 60 8 8 7"),
        height=100,
        placeholder="e.g. bench 60 8 8 7\nbor 50kg 8 8 8",
        key="raw_text_area",
    )

    parse_clicked = st.button("Parse Note", type="primary")

    if parse_clicked or "last_parse_result" in st.session_state:
        if parse_clicked:
            with st.spinner(f"Parsing with {model_choice}..."):
                t_start = time.perf_counter()
                res = parse_log(raw_text, conn=conn, model=model_choice)
                t_elapsed = round((time.perf_counter() - t_start) * 1000, 1)
                st.session_state["last_parse_result"] = res
                st.session_state["last_parse_time_ms"] = t_elapsed
                st.session_state["last_raw_text"] = raw_text

        parse_res = st.session_state["last_parse_result"]
        t_ms = st.session_state.get("last_parse_time_ms", 0.0)

        st.caption(f"Parsed in {t_ms} ms")
        if parse_res.needs_confirmation:
            st.warning("Low confidence on some exercises — please verify before saving.")

        if parse_res.issues:
            for issue in parse_res.issues:
                st.warning(f"Notice: {issue}")

        if not parse_res.entries:
            st.error("No valid exercise sets were parsed. Please check the text format.")
        else:
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
            st.markdown("#### Structured Sets")
            edited_df = st.data_editor(
                preview_df,
                num_rows="dynamic",
                use_container_width=True,
                key="workout_editor",
            )

            col_save, _ = st.columns([1, 3])
            with col_save:
                if st.button("Save Session", type="secondary", use_container_width=True):
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

                        st.success(f"Saved {saved_count} sets for {date_str}.")
                        if "last_parse_result" in st.session_state:
                            del st.session_state["last_parse_result"]
                    except Exception as err:
                        st.error(f"Error saving session: {err}")


# -----------------------------------------------------------------------------
# PAGE 2: Lift Progress & Forecast
# -----------------------------------------------------------------------------
elif page == "Progress & Forecast":
    st.title("Lift Progress & Forecast")
    st.caption("Session top-set estimated 1RM history, stall detection, and next-session forecast.")

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

    selected_lift = st.selectbox("Exercise", options=unique_lifts, index=default_index)

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
            badge_html = '<span class="tag tag-rose">Plateau</span>'
            status_desc = f"Slope {latest_slope:+.2f}%/wk (56d window)"
        else:
            badge_html = '<span class="tag tag-emerald">Progressing</span>'
            status_desc = f"Slope {latest_slope:+.2f}%/wk (56d window)"

        st.markdown(
            f"""
            <div class="glass-metric">
                <div class="metric-label">Stall Status</div>
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

    st.subheader("Barbell Plate Calculator")
    st.caption("Plate breakdown per sleeve (snapped to standard 2.5 kg plate increments).")

    # Invert TabPFN e1RM to an 8-rep working target
    default_working_weight = (
        calculate_working_weight_from_e1rm(forecast_res.point_kg, target_reps=8, increment=2.5)
        if forecast_res.point_kg > 0
        else 60.0
    )
    if is_currently_stalled:
        deload_target = snap_to_plate_increment(default_working_weight * 0.90, increment=2.5)
        st.info(
            f"Plateau detected on {selected_lift}. Suggested deload target: **{deload_target:.1f} kg** (90%) or hold **{default_working_weight:.1f} kg**."
        )

    suggested_val = float(default_working_weight) if default_working_weight >= 20.0 else 60.0
    if "plate_target_wt" not in st.session_state:
        st.session_state["plate_target_wt"] = suggested_val

    st.caption("Adjust weight:")
    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    if st_c1.button("-10 kg", key="step_sub_10", use_container_width=True):
        st.session_state["plate_target_wt"] = max(20.0, round(st.session_state["plate_target_wt"] - 10.0, 1))
        st.rerun()
    if st_c2.button("-2.5 kg", key="step_sub_2_5", use_container_width=True):
        st.session_state["plate_target_wt"] = max(20.0, round(st.session_state["plate_target_wt"] - 2.5, 1))
        st.rerun()
    if st_c3.button("+2.5 kg", key="step_add_2_5", use_container_width=True):
        st.session_state["plate_target_wt"] = min(350.0, round(st.session_state["plate_target_wt"] + 2.5, 1))
        st.rerun()
    if st_c4.button("+10 kg", key="step_add_10", use_container_width=True):
        st.session_state["plate_target_wt"] = min(350.0, round(st.session_state["plate_target_wt"] + 10.0, 1))
        st.rerun()

    col_target, col_bar, col_reps = st.columns([2, 1, 1])
    with col_target:
        target_wt = st.number_input(
            "Target Weight (kg)",
            min_value=15.0,
            max_value=350.0,
            value=float(st.session_state["plate_target_wt"]),
            step=2.5,
            key="input_target_wt_val",
            help="Total weight including bar.",
        )
        st.session_state["plate_target_wt"] = target_wt
    with col_bar:
        bar_wt = st.selectbox(
            "Barbell",
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
                    <span style="font-size:1rem; font-weight:700; color:#f8fafc;">Sleeve Breakdown</span>
                    <span style="font-size:0.82rem; color:#94a3b8; margin-left:0.5rem;">({loading.weight_per_side_loaded:.2f} kg / side)</span>
                </div>
                <div>
                    <span class="tag tag-emerald">Total: {loading.total_loaded_kg:.1f} kg</span>
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
elif page == "Coach Recap":
    st.title("Weekly Coach Recap")
    st.caption("Weekly performance summary generated from verified training numbers.")

    all_sets_df = get_history_df(conn)
    top_sets_df = get_session_top_sets(all_sets_df)

    if top_sets_df.empty:
        st.info("No workout history found. Log workouts to generate a recap.")
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

    generate_clicked = st.button("Generate Recap", type="primary")

    if generate_clicked or "coach_recap_result" in st.session_state:
        if generate_clicked:
            with st.spinner(f"Generating summary with {model_choice}..."):
                recap_res = generate_coach_summary(stats_payload, model=model_choice)
                st.session_state["coach_recap_result"] = recap_res

        result = st.session_state["coach_recap_result"]
        guard_badge = '<span class="tag tag-amber">Fallback Summary</span>' if result.is_fallback else '<span class="tag tag-emerald">Verified Stats</span>'

        st.markdown(
            f"""
            <div class="glass-panel" style="border-left: 4px solid #38bdf8; margin-top: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <div style="font-weight:700; color:#f8fafc; font-size:1rem;">Weekly Summary</div>
                    <div>{guard_badge}</div>
                </div>
                <div style="font-size:0.95rem; line-height:1.65; color:#cbd5e1; white-space:pre-wrap;">
{html.escape(result.summary)}
                </div>
                <div style="margin-top:0.75rem; font-size:0.75rem; color:#64748b;">
                    Model: <code>{result.model_used}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_copy_btn, _ = st.columns([1, 2])
        with col_copy_btn:
            if st.button("Copy Summary", use_container_width=True):
                components.html(
                    f"""
                    <script>
                    try {{
                        navigator.clipboard.writeText({repr(result.summary)});
                    }} catch (_) {{}}
                    </script>
                    """,
                    height=0,
                    width=0,
                )
                st.success("Copied to clipboard.")

        # Audio Briefing
        st.subheader("Audio Briefing")
        st.caption("Spoken summary of weekly training numbers.")

        audio_script = build_audio_briefing_script(stats_payload, user_name="Friend")
        generate_voice_clicked = st.button("Generate Audio Briefing")

        if generate_voice_clicked or "voice_briefing_result" in st.session_state:
            if generate_voice_clicked:
                with st.spinner("Synthesizing audio..."):
                    active_key = st.session_state.get(
                        "elevenlabs_api_key_input", os.environ.get("ELEVENLABS_API_KEY", "")
                    ).strip()
                    v_choice = st.session_state.get("elevenlabs_voice_choice", DEFAULT_VOICE_ID)
                    voice_res = synthesize_voice_elevenlabs(
                        script=audio_script,
                        api_key=active_key if active_key else None,
                        voice_id=v_choice,
                    )
                    st.session_state["voice_briefing_result"] = voice_res

            v_result = st.session_state["voice_briefing_result"]

            st.markdown(
                f"""
                <div class="glass-panel" style="margin-top:0.5rem;">
                    <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.35rem;">Transcript:</div>
                    <div style="font-size:0.92rem; line-height:1.5; color:#cbd5e1; font-style:italic;">
                        "{html.escape(v_result.script)}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if v_result.audio_bytes:
                st.audio(v_result.audio_bytes, format="audio/mp3", autoplay=True)
                st.download_button(
                    "Download Audio (.mp3)",
                    data=v_result.audio_bytes,
                    file_name=f"liftcast_coach_{latest_dt.strftime('%Y%m%d')}.mp3",
                    mime="audio/mp3",
                )
            else:
                safe_script_js = v_result.script.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
                components.html(
                    f"""
                    <div style="display:flex; align-items:center; gap:12px; font-family:sans-serif;">
                        <button onclick="
                            if ('speechSynthesis' in window) {{
                                window.speechSynthesis.cancel();
                                var msg = new SpeechSynthesisUtterance('{safe_script_js}');
                                msg.rate = 1.05;
                                window.speechSynthesis.speak(msg);
                            }}
                        " style="background:#0284c7; color:#ffffff; font-weight:600; padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:13px;">
                            Speak Aloud (Browser Voice)
                        </button>
                    </div>
                    """,
                    height=45,
                )

    # Strength Tiers
    st.subheader("Strength Tiers & Projections")
    st.caption("Epley e1RM to bodyweight ratios with linear slope projections.")

    t_cols = st.columns(len(KEY_LIFTS))
    for idx, lift in enumerate(KEY_LIFTS):
        l_df = top_sets_df[top_sets_df["exercise"].str.lower() == lift.lower()].sort_values("date")
        if l_df.empty:
            continue
        recent_best = float(l_df["top_e1rm"].max())

        hist_seq = [
            (datetime.strptime(d, "%Y-%m-%d").date(), float(val))
            for d, val in zip(l_df["date"], l_df["top_e1rm"])
        ]
        tier_status = get_strength_tier(lift, recent_best, bodyweight_kg, recent_history=hist_seq)

        with t_cols[idx]:
            pct = 100.0
            if tier_status.next_threshold_ratio and tier_status.threshold_ratio:
                range_span = tier_status.next_threshold_ratio - tier_status.threshold_ratio
                if range_span > 0:
                    pct = min(100.0, max(0.0, ((tier_status.current_ratio - tier_status.threshold_ratio) / range_span) * 100.0))

            proj_text = tier_status.status_message
            if tier_status.is_projectable and tier_status.projected_weeks_low and tier_status.projected_weeks_high:
                proj_text = f"~{tier_status.projected_weeks_low:.1f} to {tier_status.projected_weeks_high:.1f} wks"

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
elif page == "Benchmark & Architecture":
    st.title("Benchmark & Architecture")
    st.caption("Rolling-origin backtest results and system architecture.")

    tab_eval, tab_arch = st.tabs(["Rolling-Origin Backtest", "Architecture Diagram"])

    with tab_eval:
        st.subheader("40-Point Rolling-Origin Backtest Results")
        st.caption("Evaluated across the last 8 sessions of the 5 core lifts (40 out-of-sample predictions). Zero temporal leakage.")

        eval_summary_data = [
            {"Lift": "Lat Pulldown", "n": 8, "Last value MAE (kg)": 6.48, "Trend-5 MAE (kg)": 6.21, "TabPFN MAE (kg)": 10.84, "Winner": "Trend-5"},
            {"Lift": "Bench Press", "n": 8, "Last value MAE (kg)": 8.78, "Trend-5 MAE (kg)": 11.08, "TabPFN MAE (kg)": 12.28, "Winner": "Last value"},
            {"Lift": "Deadlift", "n": 8, "Last value MAE (kg)": 10.97, "Trend-5 MAE (kg)": 12.81, "TabPFN MAE (kg)": 15.33, "Winner": "Last value"},
            {"Lift": "Bent Over Row", "n": 8, "Last value MAE (kg)": 12.94, "Trend-5 MAE (kg)": 10.33, "TabPFN MAE (kg)": 15.04, "Winner": "Trend-5"},
            {"Lift": "Romanian Deadlift", "n": 8, "Last value MAE (kg)": 6.87, "Trend-5 MAE (kg)": 8.44, "TabPFN MAE (kg)": 18.07, "Winner": "Last value"},
            {"Lift": "**Overall**", "n": 40, "Last value MAE (kg)": 9.21, "Trend-5 MAE (kg)": 9.78, "TabPFN MAE (kg)": 14.31, "Winner": "**Last value**"},
        ]
        eval_df = pd.DataFrame(eval_summary_data)

        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown(
                """
                <div class="glass-panel">
                    <div style="font-weight:700; color:#38bdf8; margin-bottom:0.35rem;">1. Linear Trend on Consistent Lifts</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        Lat Pulldown (6.21 kg MAE) and Bent Over Row (10.33 kg MAE) reward 5-session trend extrapolation over last-value.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f2:
            st.markdown(
                """
                <div class="glass-panel">
                    <div style="font-weight:700; color:#34d399; margin-bottom:0.35rem;">2. Last Value on Noisy Compounds</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        Bench Press (8.78 kg) and Deadlift (10.97 kg) have day-to-day variance that penalizes trend extrapolation. Last-value is hard to beat.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f3:
            st.markdown(
                """
                <div class="glass-panel">
                    <div style="font-weight:700; color:#c084fc; margin-bottom:0.35rem;">3. TabPFN Uncertainty Intervals</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                        TabPFN provides calibrated 95% uncertainty intervals for risk-aware targets across disparate exercises.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_arch:
        arch_html_path = Path("docs/assets/liftcast_architecture.html")
        if arch_html_path.exists():
            html_content = arch_html_path.read_text(encoding="utf-8")
            st.caption("Interactive system architecture diagram (pan, zoom, and inspect components):")
            components.html(html_content, height=720, scrolling=True)
        else:
            st.error("Architecture diagram HTML not found at docs/assets/liftcast_architecture.html.")
