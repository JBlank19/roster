"""
Roster Generator - Master Pipeline Script

Runs all pre-processing stages in order to produce a complete set of
simulation input files:

  Stage 0: Data cleaning
  Stage 1: Markov chain construction + initial conditions
  Stage 2: Turnaround and flight-time analysis
  Stage 3: Output file generation (airlines, airports, fleet, routes)
  Stage 4: Schedule generation (greedy forward construction)

Usage:
    python tutorials/tutorial_basic.py [--seed SEED] [--suffix SUFFIX]
        [--schedule-file PATH] [--output-mode {terminal,file,non-verbose}]
        [--log-file PATH]

Example:
    python tutorials/tutorial_basic.py --seed 42 --suffix 0
    python tutorials/tutorial_basic.py --seed 42 --suffix 0 --output-mode file
    python tutorials/tutorial_basic.py --output-mode non-verbose

REFTZ Window Config
-------------------
PipelineConfig now supports:
  - reftz (default: UTC)
  - window_start (default: 00:00)
  - window_length_hours (default: 24)

Generated schema now uses REFTZ names (for example, STD_REFTZ_MINS/STA_REFTZ_MINS).
Pipeline params can also disable actual-time requirements:
  - ACTUAL_TIMES (default: false)

Status output can be routed with:
  - OUTPUT_MODE (terminal, file, or non-verbose; default: terminal)
  - LOG_FILE (optional explicit file used when OUTPUT_MODE=file)

File mode writes to log/roster{suffix}.log unless LOG_FILE or --log-file is set.
Command-line output options override tutorials/params.yaml.
"""

import argparse
import os
import sys
from pathlib import Path

import roster_generator
from roster_generator.output import (
    add_output_arguments,
    reset_log_file,
    roster_print,
)
from roster_generator.time_window import load_params_yaml, resolve_window_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roster generator pipeline"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for stochastic stages (default: 42)"
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Output file suffix, e.g. '0' -> schedule_0.csv (default: none)"
    )
    parser.add_argument(
        "--schedule-file", type=str, default="input/september2023.csv",
        help=(
            "Cleaned schedule CSV to use. For BTS, first run "
            "'python tutorials/tutorial_bts.py' or "
            "'python -m roster_generator.data_cleaning.clean_bts "
            "<on_time.csv> <aircraft.csv> --output input/bts_clean.csv', "
            "then pass "
            "--schedule-file input/bts_clean.csv."
        )
    )
    add_output_arguments(parser)
    args = parser.parse_args()

    suffix = f"_{args.suffix}" if args.suffix else ""
    params_path = Path(__file__).with_name("params.yaml")
    raw_params = load_params_yaml(params_path)
    window_cfg = resolve_window_config(raw_params)
    output_mode = args.output_mode or window_cfg.output_mode
    log_file = args.log_file or window_cfg.log_file
    log_file = reset_log_file(
        output_mode=output_mode,
        log_file=log_file,
        suffix=suffix,
    ) or log_file

    roster_print(
        "[Main] Window config: "
        f"REFTZ={window_cfg.reftz}, "
        f"WINDOW_START={window_cfg.window_start}, "
        f"WINDOW_LENGTH_HOURS={window_cfg.window_length_hours}, "
        f"ACTUAL_TIMES={window_cfg.actual_times}, "
        f"OUTPUT_MODE={output_mode}",
        output_mode=output_mode,
        log_file=log_file,
    )
    if params_path.exists():
        roster_print(f"[Main] Loaded params from {params_path}", output_mode=output_mode, log_file=log_file)
    else:
        roster_print(f"[Main] {params_path} not found. Using defaults.", output_mode=output_mode, log_file=log_file)

    # ------------------------------------------------------------------
    # Stage 0: Data Cleaning
    # ------------------------------------------------------------------
    default_input_file = "input/september2023.csv"
    input_file = args.schedule_file
    if input_file == default_input_file and not os.path.exists(input_file):
        roster_print(f"[Main] {input_file} not found. Cleaning data...", output_mode=output_mode, log_file=log_file)
        roster_generator.clean_data(
            dirty_file="ECTL/Flights_20230901_20230930.csv",
            clean_file=input_file,
            output_mode=output_mode,
            log_file=log_file,
        )
        roster_print("[Main] Cleaning data done.", output_mode=output_mode, log_file=log_file)
    elif not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Schedule file not found: {input_file}. For BTS, extract the "
            "on-time schedule CSV and Schedule B-43 aircraft inventory CSV, "
            "then run: python tutorials/tutorial_bts.py or "
            "python -m roster_generator.data_cleaning.clean_bts "
            "<on_time.csv> <aircraft.csv> --output input/bts_clean.csv"
        )
    else:
        roster_print(f"[Main] {input_file} already exists. Skipping cleaning stage.", output_mode=output_mode, log_file=log_file)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    roster_print("[Main] Setting up config...", output_mode=output_mode, log_file=log_file)
    config = roster_generator.PipelineConfig(
        schedule_file=input_file,
        analysis_dir="computed",
        output_dir="output",
        seed=args.seed,
        suffix=suffix,
        reftz=window_cfg.reftz,
        window_start=window_cfg.window_start,
        window_length_hours=window_cfg.window_length_hours,
        actual_times=window_cfg.actual_times,
        output_mode=output_mode,
        log_file=log_file,
    )
    roster_print("[Main] Setting up config done.", config=config)

    # ------------------------------------------------------------------
    # Stage 1: Initial Conditions + Markov Chains
    # ------------------------------------------------------------------
    roster_print("[Main] Generating Markov chain...", config=config)
    roster_generator.generate_markov(config)
    roster_print("[Main] Generating Markov chain done.", config=config)

    # ------------------------------------------------------------------
    # Stage 2: Turnaround + Flight Time Analysis
    # ------------------------------------------------------------------
    roster_print("[Main] Analyzing scheduled turnaround...", config=config)
    roster_generator.analyze_turnaround_distribution(config)
    roster_print("[Main] Analyzing scheduled turnaround done.", config=config)

    roster_print("[Main] Analyzing scheduled flight time...", config=config)
    roster_generator.analyze_flight_time_distribution(config)
    roster_print("[Main] Analyzing scheduled flight time done.", config=config)

    # ------------------------------------------------------------------
    # Stage 3: Output File Generation
    # ------------------------------------------------------------------
    roster_print("[Main] Generating airlines...", config=config)
    roster_generator.generate_airlines(config)
    roster_print("[Main] Generating airlines done.", config=config)

    roster_print("[Main] Generating airports...", config=config)
    roster_generator.generate_airports(config)
    roster_print("[Main] Generating airports done.", config=config)

    roster_print("[Main] Generating fleet...", config=config)
    roster_generator.generate_fleet(config)
    roster_print("[Main] Generating fleet done.", config=config)

    roster_print("[Main] Generating routes...", config=config)
    roster_generator.generate_routes(config)
    roster_print("[Main] Generating routes done.", config=config)

    # ------------------------------------------------------------------
    # Stage 4: Schedule Generation
    # ------------------------------------------------------------------
    roster_print("[Main] Generating schedule...", config=config)
    roster_generator.generate_schedule(config)
    roster_print("[Main] Generating schedule done.", config=config)

    roster_print("[Main] Pipeline completed successfully!", config=config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
