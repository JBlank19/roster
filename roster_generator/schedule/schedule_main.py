import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from .structures import Aircraft, Flight
from .stats import GenerationStats
from .data_manager import DataManager
from .capacity import CapacityTracker
from .generator import ScheduleGenerator


def load_initial_conditions(initial_conditions_path: Path) -> List[Aircraft]:
    """Load initial conditions and create Aircraft objects.

    Parameters
    ----------
    initial_conditions_path : Path
        Path to the initial conditions CSV file.

    Returns
    -------
    List[Aircraft]
        List of initialized Aircraft objects with their starting flights.
    """
    print("[Schedule] Loading initial conditions...")
    ic_df = pd.read_csv(initial_conditions_path)
    aircraft_list: List[Aircraft] = []

    for r in ic_df.itertuples():
        is_prior_only = bool(getattr(r, "PRIOR_ONLY", 0))
        initial_flight = None
        if not is_prior_only:
            std_utc_mins = getattr(r, "STD_UTC_MINS", None)
            sta_utc_mins = getattr(r, "STA_UTC_MINS", None)
            if pd.isna(std_utc_mins) or pd.isna(sta_utc_mins):
                raise ValueError(
                    f"Invalid initial_conditions row for AC_REG={r.AC_REG}: "
                    "missing STD_UTC_MINS/STA_UTC_MINS without PRIOR_ONLY=1"
                )
            initial_flight = Flight(
                orig=r.ORIGIN,
                dest=r.DEST,
                std=int(std_utc_mins),
                sta=int(sta_utc_mins),
            )
        is_single = bool(getattr(r, "SINGLE_FLIGHT", 0))

        prior_flight = None
        if hasattr(r, "PRIOR_ORIGIN") and pd.notna(getattr(r, "PRIOR_ORIGIN", None)):
            prior_flight = Flight(
                orig=r.PRIOR_ORIGIN,
                dest=r.PRIOR_DEST,
                std=int(r.PRIOR_STD_UTC_MINS),
                sta=int(r.PRIOR_STA_UTC_MINS),
            )

        ac = Aircraft(
            reg=r.AC_REG,
            operator=r.AC_OPERATOR,
            wake=r.AC_WAKE,
            initial_flight=initial_flight,
            prior_flight=prior_flight,
            is_single_flight=is_single,
            is_prior_only=is_prior_only,
        )
        aircraft_list.append(ac)

    single_flight_count = sum(1 for ac in aircraft_list if ac.is_single_flight)
    single_flight_with_prior = sum(1 for ac in aircraft_list if ac.is_single_flight and ac.prior_flight)
    prior_only_count = sum(1 for ac in aircraft_list if ac.is_prior_only)
    print(f"[Schedule]   Loaded {len(aircraft_list)} aircraft with initial conditions")
    print(f"[Schedule]   Single-flight aircraft (will passthrough): {single_flight_count}")
    print(f"[Schedule]     With overnight arrival: {single_flight_with_prior}")
    print(f"[Schedule]   Prior-only aircraft (arrival passthrough): {prior_only_count}")

    return aircraft_list


def run_generation(
    aircraft_list: List[Aircraft],
    generator: ScheduleGenerator,
    tracker: CapacityTracker,
    stats: GenerationStats,
) -> None:
    """Run the schedule generation process for all aircraft.

    Parameters
    ----------
    aircraft_list : List[Aircraft]
        List of aircraft to generate schedules for.
    generator : ScheduleGenerator
        The schedule generator instance.
    tracker : CapacityTracker
        The capacity tracker for monitoring airport constraints.
    stats : GenerationStats
        Statistics collector for the generation process.
    """
    # Pass 1: Seed all initial/prior flights into the capacity tracker
    print("[Schedule] Seeding initial flights...")
    needs_greedy = []
    for ac in aircraft_list:
        if generator.seed_initial_flights(ac):
            needs_greedy.append(ac)

    initial_burst, initial_rolling = tracker.compute_violations()
    initial_flights = sum(len(ac.chain) for ac in aircraft_list)
    print(
        f"[Schedule]   Seeded {initial_flights} initial flights | "
        f"Initial violations: burst={initial_burst}, rolling={initial_rolling}"
    )

    # Pass 2: Greedy chain extension (now sees full initial load)
    print(f"[Schedule] Generating greedy chains for {len(needs_greedy)} aircraft...")
    for i, ac in enumerate(needs_greedy):
        generator.generate_greedy_chain(ac)
        stats.successful_chains += 1
        stats.total_flights = sum(len(a.chain) for a in aircraft_list)

        if (i + 1) % 1000 == 0:
            burst, rolling = tracker.compute_violations()
            print(
                f"[Schedule]   {i + 1}/{len(needs_greedy)} | Flights: {stats.total_flights} | "
                f"Violations: burst={burst}, rolling={rolling}"
            )

    # Count aircraft that were complete after seeding (prior-only, single-flight)
    needs_greedy_set = set(id(ac) for ac in needs_greedy)
    for ac in aircraft_list:
        if id(ac) not in needs_greedy_set and len(ac.chain) > 0:
            stats.successful_chains += 1
    stats.total_flights = sum(len(ac.chain) for ac in aircraft_list)


def format_results(aircraft_list: List[Aircraft]) -> pd.DataFrame:
    """Format the generated flight chains into a pandas DataFrame.

    Parameters
    ----------
    aircraft_list : List[Aircraft]
        List of aircraft with their generated flight chains.

    Returns
    -------
    pd.DataFrame
        Formatted schedule data ready to be saved.
    """
    all_flights = []
    for ac in aircraft_list:
        initial_index = 1 if ac.prior_flight and ac.initial_flight is not None else 0
        for i, f in enumerate(ac.chain):
            is_prior_flight = 1 if (ac.prior_flight is not None and i == 0) else 0
            is_initial_departure = 1 if (ac.initial_flight is not None and i == initial_index) else 0
            all_flights.append({
                "airline_id": ac.operator,
                "aircraft_id": ac.reg,
                "orig_id": f.orig,
                "dest_id": f.dest,
                "STD_UTC": f.std,
                "STA_UTC": f.sta,
                "first_flight": 1 if i == 0 else 0,
                "is_prior_flight": is_prior_flight,
                "is_initial_departure": is_initial_departure,
                "single_flight_real": 1 if ac.is_single_flight else 0,
                "turnaround_to_next_category": f.turnaround_to_next_category,
                "turnaround_to_next_minutes": (
                    int(f.turnaround_to_next_minutes) if f.turnaround_to_next_minutes >= 0 else np.nan
                ),
            })

    return pd.DataFrame(all_flights)


def generate_schedule(config: PipelineConfig) -> None:
    """Generate synthetic flight schedule via greedy forward construction.

    Reads initial conditions and analysis artifacts (Markov, turnaround,
    routes, airports) and builds a full day's schedule for each aircraft.

    Parameters
    ----------
    config : PipelineConfig
        Paths and parameters for the pipeline.

    Raises
    ------
    FileNotFoundError
        If required input files do not exist.
    """
    seed = config.seed
    suffix = config.suffix

    rng = random.Random(seed)
    np.random.seed(seed)

    # Pathing
    output_path = config.output_path("schedule")
    routes_path = config.output_path("routes")
    airports_path = config.output_path("airports")
    initial_conditions_path = config.analysis_path("initial_conditions")
    markov_path = config.analysis_path("markov")
    turnaround_intraday_params_path = config.analysis_path("scheduled_turnaround_intraday_params")
    turnaround_temporal_profile_path = config.analysis_path("scheduled_turnaround_temporal_profile")

    print("[Schedule] --- SCHEDULE GENERATOR (Greedy Construction) ---")
    print(f"[Schedule] Seed={seed}, Suffix='{suffix}'")
    print(f"[Schedule] Output: {output_path}")
    print(
        f"[Schedule] Inputs: IC={initial_conditions_path}, Routes={routes_path}, "
        f"Airports={airports_path}"
    )
    print(
        f"[Schedule] Turnaround params: {turnaround_intraday_params_path}, "
        f"{turnaround_temporal_profile_path}"
    )

    for path in [initial_conditions_path, routes_path, airports_path, markov_path,
                 turnaround_intraday_params_path, turnaround_temporal_profile_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    # Initialize data managers and trackers
    data = DataManager(
        rng,
        routes_path,
        airports_path,
        markov_path,
        turnaround_intraday_params_path,
        turnaround_temporal_profile_path,
    )

    # 1. Load Initial Conditions
    aircraft_list = load_initial_conditions(initial_conditions_path)

    # Shuffle for variety
    rng.shuffle(aircraft_list)

    # 2. Setup Tracking and Generation
    tracker = CapacityTracker(data.rolling_capacity, data.burst_capacity)
    stats = GenerationStats()
    stats.total_aircraft = len(aircraft_list)

    generator = ScheduleGenerator(data, tracker, stats, rng)

    # 3. Process Aircraft
    run_generation(aircraft_list, generator, tracker, stats)

    # 4. Format and Save Output
    print("[Schedule] Saving schedule...")
    df = format_results(aircraft_list)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(output_path, index=False)

    print(stats.summary())
    if data.turnaround_lookup_stats:
        ordered_keys = sorted(data.turnaround_lookup_stats.keys())
        parts = [f"{k}={int(data.turnaround_lookup_stats[k])}" for k in ordered_keys]
        print("[Schedule] Turnaround lookup diagnostics: " + ", ".join(parts))
    print(f"[Schedule] Saved: {output_path} ({len(df)} flights)")
    print("[Schedule] --- SUCCESS ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic flight schedule")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule CSV")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Analysis output directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        schedule_file=Path(args.schedule),
        analysis_dir=Path(args.analysis_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        suffix=f"_{args.suffix}" if args.suffix else "",
    )
    generate_schedule(cfg)
