"""
Fleet Catalogue

Extracts the fleet configuration (aircraft registry) from the sampled
initial-conditions file and writes it as a flat table.

Input:
  - initial_conditions{suffix}.csv  (analysis_dir)

Output:
  - fleet{suffix}.csv
      columns: AC_REG, AC_OPER, AC_WAKE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from roster_generator.config import PipelineConfig
from roster_generator.output import (
    DEFAULT_OUTPUT_MODE,
    add_output_arguments,
    reset_log_file,
    roster_print,
)

# --- Column aliases ---

REQUIRED_COLS = ["AC_REG", "AC_OPER", "AC_WAKE"]


# --- Public API ---

def generate_fleet(config: PipelineConfig) -> None:
    """Extract the fleet catalogue from the initial-conditions file.

    Steps:

    1. Load ``initial_conditions{suffix}.csv`` from ``analysis_dir``.
    2. Select columns ``AC_REG``, ``AC_OPER``, ``AC_WAKE``.
    3. Drop duplicate registrations.
    4. Save the result to ``output_dir/fleet{suffix}.csv``.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If the initial-conditions file does not exist.
    SystemExit
        If required columns are missing from the input file.
    """
    roster_print("[Fleet] --- FLEET CATALOGUE ---", config=config)

    input_path = config.analysis_path("initial_conditions")
    output_path = config.output_path("fleet")

    if not input_path.exists():
        roster_print(f"[Fleet] Error: initial conditions file not found ({input_path}).", config=config)
        sys.exit(1)

    # 1. Load initial conditions
    roster_print(f"[Fleet] Initial conditions: {input_path}", config=config)
    df = pd.read_csv(input_path)

    # 2. Validate required columns
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        roster_print(f"[Fleet] Error: missing columns in initial conditions: {missing}", config=config)
        sys.exit(1)

    # 3. Extract and deduplicate by aircraft registration
    fleet = (
        df[REQUIRED_COLS]
        .drop_duplicates(subset=["AC_REG"])
        .sort_values("AC_REG")
        .reset_index(drop=True)
    )

    # 4. Persist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fleet.to_csv(output_path, index=False)

    roster_print(f"[Fleet] Saved: {output_path} ({len(fleet)} aircraft)", config=config)
    roster_print(fleet.head().to_string(index=False), config=config)
    roster_print("[Fleet] --- SUCCESS ---", config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fleet catalogue builder")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    # schedule_file is not used here but PipelineConfig requires it
    parser.add_argument("--schedule", type=str, default="", help="Path to schedule CSV (unused)")
    add_output_arguments(parser)
    args = parser.parse_args()

    output_mode = args.output_mode or DEFAULT_OUTPUT_MODE
    cfg = PipelineConfig(
        schedule_file=Path(args.schedule) if args.schedule else Path("."),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
        output_mode=output_mode,
        log_file=args.log_file,
    )
    reset_log_file(config=cfg)
    generate_fleet(cfg)
