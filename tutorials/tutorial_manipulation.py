"""
Tutorial: Distribution Manipulation
====================================

Shows how to use ``manipulation_fn`` to modify any distribution parameter
at runtime without touching the underlying data files.

The callback signature is::

    def manipulation_fn(params: dict[str, float], dtype: str) -> dict[str, float]

- ``params`` is a dict of the distribution's parameters.
- ``dtype`` is a string descriptor: ``"<distribution_type> <key1> <key2> ..."``.
- Return the (possibly modified) ``params`` dict.

Available distributions and their params/dtype formats:

    Distribution                 dtype example                       params keys
    -------------------------    ----------------------------------  ----------------
    Turnaround intraday          "turnaround_intraday IBE H"         location, shape, shift
    Fleet size                   "fleet_size IBE H"                  mu, sigma
    Physical turnaround min      "phys_ta_min IBE H"                 min_turnaround
    Route duration               "route_duration LEBL LEMD IBE H"   scheduled_time
    P(prior day)                 "p_prior IBE H"                     probability

By default ``manipulation_fn`` is an identity function (no changes).
To apply modifications, define your own function and pass it to
``PipelineConfig``.

Usage:
    python tutorial_manipulation.py --seed 42 --suffix manip
"""

import argparse
import os
import sys
from pathlib import Path

import roster_generator
from roster_generator.time_window import load_params_yaml, resolve_window_config


# ------------------------------------------------------------------
# Define your manipulation function here
# ------------------------------------------------------------------

def my_manipulation(params: dict[str, float], dtype: str) -> dict[str, float]:
    """Example manipulation that modifies turnaround and fleet distributions.

    The dtype string always starts with the distribution type, followed by
    the key components separated by spaces.  You can use simple string
    matching to target specific distributions, airlines, wake categories,
    or routes.
    """

    # --- Example 1: Shift turnarounds by a fixed amount ---
    # Add 5 minutes to all turnarounds while preserving distribution shape.
    # The "shift" parameter displaces the entire lognormal distribution
    # in real-space (minutes), so every sampled turnaround is 5 min longer.
    if dtype.startswith("turnaround_intraday"):
        params["shift"] = 5.0

    # --- Example 2: Scale turnaround location (log-space mean) ---
    # IMPORTANT: "location" is the log-space mean (mu) of the lognormal
    # distribution: samples ~ exp(N(location, shape^2)).
    # This means location is NOT in minutes — it's in log-minutes.
    # Multiplying location by 1.2 does NOT add 20% to turnaround times;
    # it raises the geometric mean to the power 1.2, which can be a
    # much larger effect than expected.
    # If you want to add a fixed number of minutes, use "shift" instead.
    # if dtype.startswith("turnaround_intraday"):
    #     params["location"] *= 1.2

    # --- Example 3: Batch targeting by airline ---
    # Override fleet size mean for all RYR (Ryanair) wake categories.
    # if dtype.startswith("fleet_size") and "RYR" in dtype:
    #     params["mu"] = 25.0

    # --- Example 4: Single target (airline + wake) ---
    # Double the fleet sigma for IBE Heavy specifically.
    # if dtype == "fleet_size IBE H":
    #     params["sigma"] *= 2.0

    # --- Example 5: Route-level manipulation ---
    # Add 10 minutes to all route durations departing from LEBL.
    # if dtype.startswith("route_duration LEBL"):
    #     params["scheduled_time"] += 10.0

    # --- Example 6: Modify physical turnaround minimum ---
    # Set a global minimum turnaround of 30 minutes.
    # if dtype.startswith("phys_ta_min"):
    #     params["min_turnaround"] = max(params["min_turnaround"], 30.0)

    # --- Example 7: Reduce prior-day probability for Heavy aircraft ---
    # if dtype.startswith("p_prior") and dtype.endswith("H"):
    #     params["probability"] *= 0.5

    return params


# ------------------------------------------------------------------
# Pipeline (same as tutorial_pipeline.py, but with manipulation_fn)
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roster generator pipeline with distribution manipulation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for stochastic stages (default: 42)"
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Output file suffix, e.g. '0' -> schedule_0.csv (default: none)"
    )
    args = parser.parse_args()

    suffix = f"_{args.suffix}" if args.suffix else ""
    params_path = Path(__file__).with_name("params.yaml")
    raw_params = load_params_yaml(params_path)
    window_cfg = resolve_window_config(raw_params)

    print(
        "[Main] Window config: "
        f"REFTZ={window_cfg.reftz}, "
        f"WINDOW_START={window_cfg.window_start}, "
        f"WINDOW_LENGTH_HOURS={window_cfg.window_length_hours}, "
        f"ACTUAL_TIMES={window_cfg.actual_times}"
    )

    # ------------------------------------------------------------------
    # Stage 0: Data Cleaning
    # ------------------------------------------------------------------
    input_file = "input/september2023.csv"
    if not os.path.exists(input_file):
        print(f"[Main] {input_file} not found. Cleaning data...")
        roster_generator.clean_data(
            dirty_file="ECTL/Flights_20230901_20230930.csv",
            clean_file=input_file,
        )

    # ------------------------------------------------------------------
    # Setup - pass manipulation_fn to PipelineConfig
    # ------------------------------------------------------------------
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
        manipulation_fn=my_manipulation,  # <-- the only difference
    )

    print("[Main] Using custom manipulation_fn")

    # ------------------------------------------------------------------
    # Stages 1-4: Run the full pipeline
    # ------------------------------------------------------------------
    roster_generator.generate_markov(config)
    roster_generator.analyze_turnaround_distribution(config)
    roster_generator.analyze_flight_time_distribution(config)
    roster_generator.generate_airlines(config)
    roster_generator.generate_airports(config)
    roster_generator.generate_fleet(config)
    roster_generator.generate_routes(config)
    roster_generator.generate_schedule(config)

    print("[Main] Pipeline with manipulation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
