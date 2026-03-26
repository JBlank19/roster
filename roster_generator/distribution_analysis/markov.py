"""
Markov Chain Transition Tables

Builds hourly destination-probability tables from historical flight sequences.
Each aircraft's flight chain is grouped by (airline, wake, prev_origin, origin)
and stratified by departure hour to capture time-of-day route patterns.

Two table tiers are produced:
  - Primary:  P(dest | prev_origin, origin, hour)  - memory-aware
  - Fallback: P(dest | origin, hour)               - memoryless

The public entry point ``generate_markov`` orchestrates both the Markov build
and the initial condition sampling (delegated to ``InitialConditionModel``).

Outputs:
  - markov{suffix}.csv (primary + fallback rows, DEP_HOUR_REFTZ)
  - initial_conditions{suffix}.csv
  - phys_ta{suffix}.csv
"""

from __future__ import annotations

import random
from types import MappingProxyType

import numpy as np
import pandas as pd

from roster_generator.config import MarkovContext, PipelineConfig
from roster_generator.time_window import (
    DEFAULT_REFTZ,
    DEFAULT_WINDOW_LENGTH_HOURS,
    hour_of_shifted_day,
    minute_of_shifted_day,
    parse_datetime_series_to_reftz,
    shift_series_by_window_start,
)
from .initial_conditions import InitialConditionModel

# --- Column aliases ---

AC_REG_COL = "AC_REG"
AIRLINE_COL = "AC_OPER"
AC_WAKE_COL = "AC_WAKE"
DEP_COL = "DEP_ICAO"
ARR_COL = "ARR_ICAO"
STD_COL = "STD_REFTZ"
STA_COL = "STA_REFTZ"
TABLE_KIND_COL = "TABLE_KIND"
PREV_COL = "PREV_ICAO"
HOUR_COL = "DEP_HOUR_REFTZ"
COUNT_COL = "COUNT"
WEIGHT_COL = "WEIGHT"
PROB_COL = "PROB"

MARKOV_EXPORT_COLUMNS = [
    TABLE_KIND_COL,
    AIRLINE_COL,
    AC_WAKE_COL,
    PREV_COL,
    DEP_COL,
    ARR_COL,
    HOUR_COL,
    COUNT_COL,
    WEIGHT_COL,
    PROB_COL,
]


# --- Helpers ---

def _to_minute_bin_preserve_day(value_mins):
    """Convert to integer minute bin, preserving sign for previous-day values."""
    value = float(value_mins)
    if value < 0:
        return int(np.floor(value))
    return int(round(value))


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


# --- Data preparation ---

def _prepare_base_flights(
    df,
    airline_filter=None,
    *,
    reftz: str = DEFAULT_REFTZ,
    window_start_mins: int = 0,
    window_length_mins: int = DEFAULT_WINDOW_LENGTH_HOURS * 60,
):
    """Normalise raw schedule columns, apply filters, and drop unusable rows.

    Handles the "ZZZ" sentinel airline (remaps to AC_REG), normalises wake
    categories, and removes same-airport flights.
    """
    _require_columns(
        df,
        [AC_REG_COL, AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL, STD_COL, STA_COL],
        "schedule",
    )

    # Remap placeholder airline "ZZZ" -> actual registration
    if AIRLINE_COL in df.columns and AC_REG_COL in df.columns:
        zzz_mask = df[AIRLINE_COL].astype(str).str.upper().str.strip() == "ZZZ"
        zzz_count = int(zzz_mask.sum())
        if zzz_count:
            df.loc[zzz_mask, AIRLINE_COL] = df.loc[zzz_mask, AC_REG_COL].astype(str).str.strip()
        print(f"  Remapped {zzz_count} flights: AC_OPER='ZZZ' -> AC_REG")

    for c in [AC_REG_COL, AIRLINE_COL, DEP_COL, ARR_COL]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df[AIRLINE_COL] = df[AIRLINE_COL].str.upper()
    df[AC_WAKE_COL] = df[AC_WAKE_COL].fillna("").astype(str).str.strip().str.upper()

    df["STD"] = parse_datetime_series_to_reftz(df[STD_COL], reftz)
    df["STA"] = parse_datetime_series_to_reftz(df[STA_COL], reftz)
    df = df.dropna(subset=["STD", "STA"])
    df["STD"] = shift_series_by_window_start(df["STD"], window_start_mins)
    df["STA"] = shift_series_by_window_start(df["STA"], window_start_mins)

    if int(window_length_mins) < 24 * 60:
        std_mins = minute_of_shifted_day(df["STD"])
        df = df[(std_mins >= 0) & (std_mins < int(window_length_mins))].copy()
        if df.empty:
            raise ValueError("No usable flights inside configured REFTZ window.")

    df["DEP_HOUR_REFTZ"] = hour_of_shifted_day(df["STD"]).astype(int)

    if airline_filter:
        af = str(airline_filter).strip().upper()
        before = len(df)
        df = df[df[AIRLINE_COL] == af].copy()
        print(f"  Airline filter {af}: {len(df)} rows from {before}")

    df = df[df[DEP_COL] != df[ARR_COL]].copy()
    if df.empty:
        raise ValueError("No usable flights after normalization/filtering.")

    return df


# --- Markov table construction ---

def _prepare_markov_source(base_df: pd.DataFrame) -> pd.DataFrame:
    """Sort flights and derive previous-origin/hour columns used by Markov tables."""
    df = base_df.sort_values(by=[AC_REG_COL, "STD"]).reset_index(drop=True).copy()
    df[PREV_COL] = df.groupby(AC_REG_COL, sort=False)[DEP_COL].shift(1)
    if HOUR_COL not in df.columns:
        df[HOUR_COL] = df["STD"].dt.hour
    return df


def _build_primary_markov_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate primary Markov counts with previous-origin memory."""
    primary_df = df[df[PREV_COL].notna()].copy()
    if primary_df.empty:
        return pd.DataFrame(columns=MARKOV_EXPORT_COLUMNS)

    grouped = (
        primary_df.groupby(
            [AIRLINE_COL, AC_WAKE_COL, PREV_COL, DEP_COL, ARR_COL, HOUR_COL],
            sort=False,
        )
        .size()
        .reset_index(name=COUNT_COL)
    )
    grouped[TABLE_KIND_COL] = "primary"
    grouped[WEIGHT_COL] = grouped[COUNT_COL].astype(float)
    totals = grouped.groupby(
        [TABLE_KIND_COL, AIRLINE_COL, AC_WAKE_COL, PREV_COL, DEP_COL, HOUR_COL],
        sort=False,
    )[WEIGHT_COL].transform("sum")
    grouped[PROB_COL] = grouped[WEIGHT_COL] / totals
    return grouped[MARKOV_EXPORT_COLUMNS].copy()


def _build_fallback_markov_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fallback Markov counts without previous-origin memory."""
    grouped = (
        df.groupby([AIRLINE_COL, AC_WAKE_COL, DEP_COL, ARR_COL, HOUR_COL], sort=False)
        .size()
        .reset_index(name=COUNT_COL)
    )
    grouped[TABLE_KIND_COL] = "fallback"
    grouped[PREV_COL] = ""
    grouped[WEIGHT_COL] = grouped[COUNT_COL].astype(float)
    totals = grouped.groupby(
        [TABLE_KIND_COL, AIRLINE_COL, AC_WAKE_COL, PREV_COL, DEP_COL, HOUR_COL],
        sort=False,
    )[WEIGHT_COL].transform("sum")
    grouped[PROB_COL] = grouped[WEIGHT_COL] / totals
    return grouped[MARKOV_EXPORT_COLUMNS].copy()


def _validate_markov_bias_map(
    bias_map: object,
    base_probs: dict[str, float],
    context: MarkovContext,
) -> dict[str, float]:
    """Validate user Markov biases and default missing destinations to 1.0."""
    if bias_map is None:
        return {dest: 1.0 for dest in base_probs}
    if not isinstance(bias_map, dict):
        raise ValueError(
            f"Invalid Markov manipulation for {context.table_kind} "
            f"{context.airline} {context.origin} hour {context.dep_hour_reftz}: "
            f"expected dict or None, got {type(bias_map).__name__}"
        )

    unknown = sorted(set(bias_map.keys()) - set(base_probs.keys()))
    if unknown:
        raise ValueError(
            f"Unknown Markov destinations for {context.table_kind} "
            f"{context.airline} {context.origin} hour {context.dep_hour_reftz}: {unknown}"
        )

    out: dict[str, float] = {dest: 1.0 for dest in base_probs}
    for dest, raw_bias in bias_map.items():
        bias = float(raw_bias)
        if not np.isfinite(bias) or bias <= 0:
            raise ValueError(
                f"Invalid Markov bias for destination {dest!r}: {raw_bias!r}. "
                "Biases must be finite and strictly positive."
            )
        out[str(dest)] = bias
    return out


def _apply_markov_manipulation(
    grouped_df: pd.DataFrame,
    markov_manipulation_fn,
) -> pd.DataFrame:
    """Apply row-wise support-preserving Markov manipulation."""
    if grouped_df.empty:
        return grouped_df.reindex(columns=MARKOV_EXPORT_COLUMNS).copy()

    output_frames: list[pd.DataFrame] = []
    group_cols = [TABLE_KIND_COL, AIRLINE_COL, AC_WAKE_COL, PREV_COL, DEP_COL, HOUR_COL]

    for group_key, frame in grouped_df.groupby(group_cols, sort=False, dropna=False):
        table_kind, airline, wake, prev_origin, origin, dep_hour = group_key
        base_counts = {
            str(row.ARR_ICAO): int(row.COUNT)
            for row in frame.itertuples(index=False)
        }
        total_count = float(sum(base_counts.values()))
        if total_count <= 0:
            raise ValueError(f"Invalid Markov row with non-positive total count: {group_key}")

        base_probs = {
            dest: float(count / total_count)
            for dest, count in base_counts.items()
        }
        normalized_prev = None if str(table_kind) == "fallback" else str(prev_origin)
        context = MarkovContext(
            table_kind=str(table_kind),
            airline=str(airline),
            wake=str(wake),
            prev_origin=normalized_prev,
            origin=str(origin),
            dep_hour_reftz=int(dep_hour),
            base_probs=MappingProxyType(base_probs.copy()),
            base_counts=MappingProxyType(base_counts.copy()),
        )
        bias_map = _validate_markov_bias_map(
            markov_manipulation_fn(dict(base_probs), context),
            base_probs,
            context,
        )

        frame = frame.copy()
        frame[WEIGHT_COL] = frame.apply(
            lambda row: float(row[COUNT_COL]) * bias_map[str(row[ARR_COL])],
            axis=1,
        )
        total_weight = float(frame[WEIGHT_COL].sum())
        if total_weight <= 0:
            raise ValueError(f"Invalid Markov row with non-positive total weight: {group_key}")
        frame[PROB_COL] = frame[WEIGHT_COL] / total_weight
        output_frames.append(frame[MARKOV_EXPORT_COLUMNS])

    return pd.concat(output_frames, ignore_index=True)


def _pack_markov_tables(markov_df: pd.DataFrame):
    """Pack Markov CSV rows into runtime lookup tables using WEIGHT values."""
    markov_hourly = {}
    markov_fallback_hourly = {}

    if markov_df.empty:
        return markov_hourly, markov_fallback_hourly

    for row in markov_df.itertuples(index=False):
        table_kind = getattr(row, TABLE_KIND_COL, "primary")
        op = str(row.AC_OPER)
        wake = str(row.AC_WAKE)
        prev = str(row.PREV_ICAO)
        dep = str(row.DEP_ICAO)
        arr = str(row.ARR_ICAO)
        hour = int(getattr(row, HOUR_COL))
        weight = float(getattr(row, WEIGHT_COL, getattr(row, COUNT_COL)))

        if table_kind == "fallback":
            fkey = (op, wake, dep)
            if fkey not in markov_fallback_hourly:
                markov_fallback_hourly[fkey] = {}
            if hour not in markov_fallback_hourly[fkey]:
                markov_fallback_hourly[fkey][hour] = {}
            markov_fallback_hourly[fkey][hour][arr] = (
                markov_fallback_hourly[fkey][hour].get(arr, 0.0) + weight
            )
            continue

        pkey = (op, wake, prev, dep)
        if pkey not in markov_hourly:
            markov_hourly[pkey] = {}
        if hour not in markov_hourly[pkey]:
            markov_hourly[pkey][hour] = {}
        markov_hourly[pkey][hour][arr] = markov_hourly[pkey][hour].get(arr, 0.0) + weight

    return markov_hourly, markov_fallback_hourly


def _build_markov_tables(base_df, markov_manipulation_fn=None):
    """Build primary/fallback Markov tables and apply optional user manipulation.

    Returns:
        final_markov: DataFrame with combined primary and fallback rows for CSV export.
        markov_hourly: nested dict  (op, wake, prev, dep) -> hour -> {arr: weight}
        markov_fallback_hourly: nested dict  (op, wake, dep) -> hour -> {arr: weight}
    """
    markov_manipulation_fn = markov_manipulation_fn or (lambda params, context: None)
    df = _prepare_markov_source(base_df)

    primary_df = _build_primary_markov_counts(df)
    fallback_df = _build_fallback_markov_counts(df)
    combined_df = pd.concat([primary_df, fallback_df], ignore_index=True)
    final_markov = _apply_markov_manipulation(combined_df, markov_manipulation_fn)
    markov_hourly, markov_fallback_hourly = _pack_markov_tables(final_markov)
    return final_markov, markov_hourly, markov_fallback_hourly


# --- Public API ---

def generate_markov(config: PipelineConfig, airline_filter: str | None = None) -> None:
    """Run Markov chain analysis and generate synthetic initial conditions.

    Pipeline:
      1. Load and normalise the schedule.
      2. Build Markov transition tables.
      3. Build empirical initial-condition distributions.
      4. Inject Markov tables into the IC model (needed for first-flight destinations).
      5. Sample one synthetic fleet and write all outputs.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.
    airline_filter : str or None, optional
        Restrict analysis to a single ICAO airline code.

    Raises
    ------
    FileNotFoundError
        If ``config.schedule_file`` does not exist.
    ValueError
        If no usable flights remain after normalization and filtering.
    """
    seed = config.seed
    suffix = config.suffix

    print("[Markov] --- Synthetic Initial Conditions + Continuation ---")
    print(f"[Markov] Using SEED={seed}")
    if airline_filter:
        print(f"[Markov] Filtering for airline: {airline_filter}")

    np.random.seed(seed)
    random.seed(seed)

    if not config.schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {config.schedule_file}")

    df = pd.read_csv(config.schedule_file)
    print(f"[Markov] Schedule: {config.schedule_file}")

    base_df = _prepare_base_flights(
        df,
        airline_filter=airline_filter,
        reftz=config.reftz,
        window_start_mins=config.window_start_mins,
        window_length_mins=config.window_length_mins,
    )
    print(f"[Markov]   {base_df[AC_REG_COL].nunique()} unique aircraft in normalized schedule")

    # Step 1: Markov transitions
    final_markov, markov_hourly, markov_fallback_hourly = _build_markov_tables(
        base_df,
        config.markov_manipulation_fn,
    )

    # Step 2: Initial conditions (needs Markov tables for destination sampling)
    model = InitialConditionModel(
        base_df,
        seed=seed,
        window_length_mins=config.window_length_mins,
    )
    model.build_all()
    model.set_markov_tables(markov_hourly, markov_fallback_hourly)
    model.apply_manipulation(config.manipulation_fn)

    ic_df = model.sample_initial_conditions()

    # Step 3: Persist outputs
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    initial_conditions_path = config.analysis_path("initial_conditions")
    ic_df.to_csv(initial_conditions_path, index=False)

    markov_path = config.analysis_path("markov")
    final_markov.to_csv(markov_path, index=False)

    phys_ta_path = config.output_path("phys_ta")
    phys_ta_df = model.phys_ta_df.sort_values(["airline_id", "aircraft_wake"]).reset_index(drop=True)
    phys_ta_df.to_csv(phys_ta_path, index=False)

    prior_rows = int(ic_df["PRIOR_STD_REFTZ_MINS"].notna().sum())
    prior_only_rows = int((ic_df["PRIOR_ONLY"].astype(int) == 1).sum())
    single_flights = int(ic_df["SINGLE_FLIGHT"].fillna(0).astype(int).sum())

    print(f"[Markov] Fleet size (synthetic): {len(ic_df)}")
    print(f"[Markov]   Prior rows: {prior_rows}")
    print(f"[Markov]   Prior-only rows: {prior_only_rows}")
    print(f"[Markov]   Single-flight rows: {single_flights}")
    print(f"[Markov] Saved: {initial_conditions_path}")
    print(f"[Markov] Saved: {markov_path} ({len(final_markov)} transitions)")
    print(f"[Markov] Saved: {phys_ta_path} ({len(phys_ta_df)} rows)")
