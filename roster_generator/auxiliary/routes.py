"""
Route Flight-Time Catalogue

Computes median scheduled and actual flight times for every route present in
the Markov transition tables. Results are grouped by
(origin, destination, airline, wake) with an operator-agnostic "ALL" fallback.
When actual times are disabled, the actual-time output column mirrors the
scheduled median to preserve schema compatibility.

Output:
  - routes{suffix}.csv
      columns: orig_id, dest_id, airline_id, wake_type, scheduled_time, time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roster_generator.config import PipelineConfig
from roster_generator.output import (
    DEFAULT_OUTPUT_MODE,
    OutputConfig,
    add_output_arguments,
    reset_log_file,
    roster_print,
)
from roster_generator.time_window import DEFAULT_REFTZ, parse_datetime_series_to_reftz

# --- Column aliases ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
AC_WAKE_COL = "AC_WAKE"
DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"
ATD_COL = "ATD_REFTZ"
ATA_COL = "ATA_REFTZ"


# --- Data preparation ---


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _prepare_flights(
    df: pd.DataFrame,
    reftz: str = DEFAULT_REFTZ,
    *,
    actual_times: bool = False,
    output_config: OutputConfig | None = None,
) -> pd.DataFrame:
    """Normalise columns, remap ZZZ airlines, compute flight durations."""
    required = [AC_REG_COL, AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL, STD_COL, STA_COL]
    if actual_times:
        required.extend([ATD_COL, ATA_COL])
    _require_columns(
        df,
        required,
        "schedule",
    )

    # Remap placeholder airline "ZZZ" -> actual registration
    if AIRLINE_COL in df.columns and AC_REG_COL in df.columns:
        zzz_mask = df[AIRLINE_COL].astype(str).str.upper().str.strip() == "ZZZ"
        zzz_count = int(zzz_mask.sum())
        if zzz_count:
            df.loc[zzz_mask, AIRLINE_COL] = df.loc[zzz_mask, AC_REG_COL].astype(str).str.strip()
        roster_print(f"[Routes] Remapped {zzz_count} flights: AC_OPER='ZZZ' -> AC_REG", config=output_config)

    df[AC_WAKE_COL] = df[AC_WAKE_COL].fillna("").astype(str).str.upper().str.strip()

    # Parse scheduled times first; actuals are optional.
    for col in [STD_COL, STA_COL]:
        df[col] = parse_datetime_series_to_reftz(df[col], reftz)
    df = df.dropna(subset=[STD_COL, STA_COL]).copy()

    # Compute scheduled duration (minutes)
    df["SCHEDULED_FLIGHT_TIME"] = (df[STA_COL] - df[STD_COL]).dt.total_seconds() / 60.0
    df = df[df["SCHEDULED_FLIGHT_TIME"] > 0].copy()

    if actual_times:
        for col in [ATD_COL, ATA_COL]:
            df[col] = parse_datetime_series_to_reftz(df[col], reftz)
        df = df.dropna(subset=[ATD_COL, ATA_COL]).copy()
        df["ACTUAL_FLIGHT_TIME"] = (df[ATA_COL] - df[ATD_COL]).dt.total_seconds() / 60.0
        df = df[df["ACTUAL_FLIGHT_TIME"] > 0].copy()
    else:
        df["ACTUAL_FLIGHT_TIME"] = df["SCHEDULED_FLIGHT_TIME"]

    return df


# --- Route statistics ---

def _build_per_operator_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median scheduled and actual flight times per (origin, dest, airline, wake)."""
    group_cols = [DEP_COL, ARR_COL, AIRLINE_COL, AC_WAKE_COL]
    stats = df.groupby(group_cols).agg(
        SCHEDULED_FLIGHT_TIME=("SCHEDULED_FLIGHT_TIME", "median"),
        ACTUAL_FLIGHT_TIME=("ACTUAL_FLIGHT_TIME", "median"),
    ).reset_index()
    return stats


def _build_operator_agnostic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median flight times with airline_id = 'ALL' (fallback)."""
    group_cols = [DEP_COL, ARR_COL, AC_WAKE_COL]
    stats = df.groupby(group_cols).agg(
        SCHEDULED_FLIGHT_TIME=("SCHEDULED_FLIGHT_TIME", "median"),
        ACTUAL_FLIGHT_TIME=("ACTUAL_FLIGHT_TIME", "median"),
    ).reset_index()
    stats[AIRLINE_COL] = "ALL"
    return stats


def _rename_and_round(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to output schema and round times to integer minutes."""
    df = df.rename(columns={
        DEP_COL: "orig_id",
        ARR_COL: "dest_id",
        AIRLINE_COL: "airline_id",
        AC_WAKE_COL: "wake_type",
        "SCHEDULED_FLIGHT_TIME": "scheduled_time",
        "ACTUAL_FLIGHT_TIME": "time",
    })
    df["scheduled_time"] = df["scheduled_time"].round().astype(int)
    df["time"] = df["time"].round().astype(int)
    return df


# --- Public API ---

def generate_routes(config: PipelineConfig) -> None:
    """Compute median route flight times from the schedule, filtered to Markov routes.

    Only routes present in the Markov transition table are kept.  Both
    per-operator and operator-agnostic (``ALL``) statistics are produced.
    When ``config.actual_times`` is false, the output ``time`` column mirrors
    scheduled medians.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If ``config.schedule_file`` or the markov analysis file do not exist.
    ValueError
        If no flights match the Markov routes.
    """
    roster_print("[Routes] --- ROUTE FLIGHT-TIME CATALOGUE ---", config=config)

    markov_path = config.analysis_path("markov")
    output_path = config.output_path("routes")

    if not config.schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {config.schedule_file}")
    if not markov_path.exists():
        raise FileNotFoundError(f"Markov file not found: {markov_path}")

    # 1. Load markov routes
    roster_print(f"[Routes] Markov file: {markov_path}", config=config)
    markov_df = pd.read_csv(markov_path)
    routes = markov_df[[DEP_COL, ARR_COL, AC_WAKE_COL]].drop_duplicates()
    roster_print(f"[Routes] {len(routes)} unique route/wake combinations", config=config)

    # 2. Load and prepare schedule
    roster_print(f"[Routes] Schedule: {config.schedule_file}", config=config)
    if config.actual_times:
        roster_print("[Routes] Using actual flight-time medians", config=config)
    else:
        roster_print("[Routes] Using scheduled-only medians for route time", config=config)
    df = pd.read_csv(config.schedule_file)
    df = _prepare_flights(
        df,
        reftz=config.reftz,
        actual_times=config.actual_times,
        output_config=config,
    )

    if df.empty:
        raise ValueError("No valid flights after normalisation.")

    roster_print(f"[Routes] {len(df)} valid flights", config=config)

    # 3. Filter schedule to markov routes
    merged = df.merge(routes, on=[DEP_COL, ARR_COL, AC_WAKE_COL], how="inner")

    if merged.empty:
        raise ValueError("No flights match the Markov routes.")

    roster_print(f"[Routes] {len(merged)} flights matching Markov routes", config=config)

    # 4. Build statistics
    roster_print("[Routes] Computing median flight times...", config=config)
    per_op = _build_per_operator_stats(merged)
    agnostic = _build_operator_agnostic_stats(merged)
    combined = pd.concat([per_op, agnostic], ignore_index=True)
    combined = _rename_and_round(combined)

    # 5. Persist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Summary
    roster_print(f"[Routes] Saved: {output_path} ({len(combined)} routes)", config=config)
    roster_print(f"[Routes] Wake types: {sorted(combined['wake_type'].unique())}", config=config)
    roster_print("[Routes] --- SUCCESS ---", config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route flight-time catalogue builder")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule CSV")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    add_output_arguments(parser)
    args = parser.parse_args()

    output_mode = args.output_mode or DEFAULT_OUTPUT_MODE
    cfg = PipelineConfig(
        schedule_file=Path(args.schedule),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
        output_mode=output_mode,
        log_file=args.log_file,
    )
    reset_log_file(config=cfg)
    generate_routes(cfg)
