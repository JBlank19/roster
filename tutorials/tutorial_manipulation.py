"""
Tutorial: Distribution And Markov Manipulation
==============================================

Shows how to use:

- ``manipulation_fn`` to modify scalar distribution parameters at runtime
- ``markov_manipulation_fn`` to reweight Markov destinations without
  breaking row-stochastic validity

The scalar callback signature is::

    def manipulation_fn(params: dict[str, float], dtype: str) -> dict[str, float]

- ``params`` is a dict of the distribution's parameters.
- ``dtype`` is a string descriptor: ``"<distribution_type> <key1> <key2> ..."``.
- Return the (possibly modified) ``params`` dict.

The Markov callback signature is::

    def markov_manipulation_fn(
        base_probs: dict[str, float],
        ctx: roster_generator.MarkovContext,
    ) -> dict[str, float] | None

- ``base_probs`` is the destination probability row before manipulation.
- ``ctx`` contains metadata such as airline, origin, previous origin,
  hour, and whether the row is from the primary or fallback table.
- Return multiplicative biases by destination. Missing destinations
  default to ``1.0`` and the row is renormalized automatically.

Available distributions and their params/dtype formats:

    Distribution                 dtype example                       params keys
    -------------------------    ----------------------------------  ----------------
    Turnaround intraday          "turnaround_intraday IBE H"         location, shape, shift
    Fleet size                   "fleet_size IBE H"                  mu, sigma
    Physical turnaround min      "phys_ta_min IBE H"                 min_turnaround
    Route duration               "route_duration LEBL LEMD IBE H"   scheduled_time
    P(prior day)                 "p_prior IBE H"                     probability

By default ``manipulation_fn`` is an identity function (no changes).
By default both callbacks are identities (no changes). To apply
modifications, define your own functions and pass them to ``PipelineConfig``.

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


def my_markov_manipulation(
    base_probs: dict[str, float],
    ctx: roster_generator.MarkovContext,
) -> dict[str, float]:
    """Example Markov manipulation using multiplicative destination biases.

    This callback never adds new routes. It only reweights destinations that
    already exist in the historical row and the code renormalizes the row
    afterward.

    Worked example:
      base row  = {"LEMD": 0.40, "LEVC": 0.35, "LEPA": 0.25}
      bias map  = {"LEMD": 1.5, "LEVC": 0.7}
      weighted  = {"LEMD": 0.60, "LEVC": 0.245, "LEPA": 0.25}
      final row = {"LEMD": 0.548, "LEVC": 0.224, "LEPA": 0.228}
    """

    bias: dict[str, float] = {}

    # --- Example 1: Boost the odds of a specific route in the morning ---
    # For IBE departures from LEBL between 06:00 and 11:59 REFTZ,
    # increase the odds of going to LEMD by 50%.
    if (
        ctx.airline == "IBE"
        and ctx.origin == "LEBL"
        and 6 <= ctx.dep_hour_reftz <= 11
        and "LEMD" in base_probs
    ):
        bias["LEMD"] = 1.5

    # --- Example 2: Mildly discourage another destination on the same row ---
    if ctx.origin == "LEBL" and "LEVC" in base_probs:
        bias["LEVC"] = 0.7

    return bias


# ------------------------------------------------------------------
# Pipeline (same as tutorial_pipeline.py, but with both manipulation hooks)
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
    # Setup - pass both manipulation callbacks to PipelineConfig
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
        manipulation_fn=my_manipulation,
        markov_manipulation_fn=my_markov_manipulation,
    )

    print("[Main] Using custom manipulation_fn and markov_manipulation_fn")

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
