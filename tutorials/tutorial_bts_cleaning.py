"""Clean BTS downloads into a ROSTER schedule file.

Before running this tutorial:

- Download BTS monthly on-time data from https://transtats.bts.gov/PREZIP/
- Extract the on-time schedule CSV into BTS/
- Download Schedule B-43 Inventory from:
  https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=Nv4+Pn44vr4+Sv0n0pvny&gnoyr_VQ=GEH
- Extract the Schedule B-43 aircraft inventory CSV into BTS/

ROSTER does not download or read BTS zip archives. Put those extracted CSV
files in a folder named BTS, then run:

    python tutorials/tutorial_bts_cleaning.py

The tutorial accepts friendly aliases (BTS/on_time.csv and BTS/aircraft.csv)
or the default extracted BTS filenames, such as:

    BTS/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv
    BTS/T_F41SCHEDULE_B43.csv
"""

from pathlib import Path

import roster_generator

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

    print(f"Resolved BTS on-time CSV: {schedule_csv}")
    print(f"Resolved BTS aircraft CSV: {aircraft_csv}")
    report = roster_generator.clean_bts_data(schedule_csv, aircraft_csv, output_file)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
