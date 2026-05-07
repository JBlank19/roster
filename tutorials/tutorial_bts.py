"""Clean BTS downloads into a ROSTER schedule file.

Before running this tutorial:

- Download BTS monthly on-time data from https://transtats.bts.gov/PREZIP/
- Extract the on-time schedule CSV into BTS/
- Download Schedule B-43 Inventory from:
  https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=Nv4+Pn44vr4+Sv0n0pvny&gnoyr_VQ=GEH
- Extract the Schedule B-43 aircraft inventory CSV into BTS/

ROSTER does not download or read BTS zip archives. Put those extracted CSV
files in a folder named BTS, then run:

    python tutorials/tutorial_bts.py

Status output can be routed with OUTPUT_MODE in tutorials/params.yaml or
``--output-mode`` on the command line. ``terminal`` displays progress,
``file`` writes to log/roster.log unless LOG_FILE/``--log-file`` is set, and
``non-verbose`` suppresses routed progress messages.

The tutorial accepts friendly aliases (BTS/on_time.csv and BTS/aircraft.csv)
or the default extracted BTS filenames, such as:

    BTS/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv
    BTS/T_F41SCHEDULE_B43.csv
"""

import argparse
from pathlib import Path

import roster_generator
from roster_generator.output import add_output_arguments, reset_log_file, roster_print
from roster_generator.time_window import load_params_yaml, resolve_window_config

BTS_DIR = Path("BTS")
ON_TIME_ALIAS = BTS_DIR / "on_time.csv"
AIRCRAFT_ALIAS = BTS_DIR / "aircraft.csv"
ON_TIME_PATTERNS = ("On_Time_Reporting_Carrier_On_Time_Performance*.csv",)
AIRCRAFT_PATTERNS = ("T_F41SCHEDULE_B43.csv", "*F41SCHEDULE_B43*.csv")


def _unique_sorted_paths(paths: list[Path]) -> list[Path]:
    """Return sorted unique paths."""
    return sorted(set(paths), key=lambda path: str(path))


def _resolve_bts_csv(
    *,
    label: str,
    alias_path: Path,
    patterns: tuple[str, ...],
) -> Path:
    """Resolve one BTS source CSV from an alias or default extracted filename."""
    if alias_path.exists():
        if not alias_path.is_file():
            raise ValueError(f"{label} alias must be a file: {alias_path}")
        return alias_path

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(BTS_DIR.glob(pattern))
    matches = [path for path in _unique_sorted_paths(matches) if path.is_file()]

    if len(matches) == 1:
        return matches[0]

    pattern_text = ", ".join(str(BTS_DIR / pattern) for pattern in patterns)
    if not matches:
        raise FileNotFoundError(
            f"Could not auto-detect {label}. Expected {alias_path} or exactly "
            f"one of: {pattern_text}."
        )

    listed = ", ".join(str(path) for path in matches)
    raise ValueError(
        f"Multiple {label} candidates found: {listed}. Keep only one matching "
        "file in BTS/ or rename the intended file to the friendly alias."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean BTS downloads into a ROSTER schedule file"
    )
    add_output_arguments(parser)
    args = parser.parse_args()

    params_path = Path(__file__).with_name("params.yaml")
    raw_params = load_params_yaml(params_path)
    window_cfg = resolve_window_config(raw_params)
    output_mode = args.output_mode or window_cfg.output_mode
    log_file = args.log_file or window_cfg.log_file
    log_file = (
        reset_log_file(output_mode=output_mode, log_file=log_file)
        or log_file
    )

    schedule_csv = _resolve_bts_csv(
        label="BTS on-time schedule CSV",
        alias_path=ON_TIME_ALIAS,
        patterns=ON_TIME_PATTERNS,
    )
    aircraft_csv = _resolve_bts_csv(
        label="Schedule B-43 aircraft inventory CSV",
        alias_path=AIRCRAFT_ALIAS,
        patterns=AIRCRAFT_PATTERNS,
    )
    output_file = Path("input") / "bts_clean.csv"

    roster_print(f"Resolved BTS on-time CSV: {schedule_csv}", output_mode=output_mode, log_file=log_file)
    roster_print(f"Resolved BTS aircraft CSV: {aircraft_csv}", output_mode=output_mode, log_file=log_file)
    report = roster_generator.clean_bts_data(
        schedule_csv,
        aircraft_csv,
        output_file,
        output_mode=output_mode,
        log_file=log_file,
    )
    roster_print(report, output_mode=output_mode, log_file=log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
