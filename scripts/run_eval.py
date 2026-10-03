"""Run the 40-Point Rolling-Origin Backtest Evaluation (§7.4) and print markdown."""

from __future__ import annotations
import sys
from liftcast.db import get_connection, get_history_df
from liftcast.metrics import get_session_top_sets
from liftcast.forecast import run_rolling_origin_eval, KEY_LIFTS


def df_to_markdown(df) -> str:
    try:
        return df.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        headers = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
        return "\n".join(lines)


def main() -> None:
    from pathlib import Path
    from liftcast.importer import import_liftoff_csv
    
    db_path = "data/liftcast.db"
    conn = get_connection(db_path)
    df = get_history_df(conn)
    
    raw_csv = Path("data/raw/liftoff_workout_data.csv")
    if df.empty and raw_csv.exists():
        print(f"Database empty. Auto-seeding from {raw_csv}...")
        import_liftoff_csv(raw_csv, conn)
        df = get_history_df(conn)
        
    if df.empty:
        print("Database is empty and no raw CSV found to seed.")
        sys.exit(1)

    top_sets_df = get_session_top_sets(df)
    summary_df, detailed_df = run_rolling_origin_eval(
        top_sets_df,
        lifts=KEY_LIFTS,
        n_test_sessions=8,
    )

    if summary_df.empty:
        print("Could not compute rolling origin evaluation.")
        sys.exit(1)

    print("\n### 40-Point Rolling-Origin Backtest Results (§7.4)\n")
    print(df_to_markdown(summary_df))
    print(f"\nTotal out-of-sample predictions evaluated: {len(detailed_df)}\n")


if __name__ == "__main__":
    main()

