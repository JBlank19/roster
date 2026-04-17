# ROSTER - Realistic Operational Schedules Through Empirical Records

[![PyPI version](https://img.shields.io/pypi/v/roster-generator)](https://pypi.org/project/roster-generator/)
[![Python versions](https://img.shields.io/pypi/pyversions/roster-generator)](https://pypi.org/project/roster-generator/)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

ROSTER is a Python package for generating realistic flight
schedules from historical data. It is designed for ATM researchers that require synthetic schedules (with or without modification) for simulation purposes.

The goal of ROSTER is to provide a transparent and reproducible pipeline for
building these synthetic schedules, which preserve important operational structure:
airline and wake-category mixes, airport and route usage, scheduled flight
times and turnarounds, and fleet initial conditions.

## Installation

ROSTER requires Python 3.12 or newer. Install the latest released package with:

```bash
python -m pip install roster-generator
```

The installed import package is named `roster_generator`:

```python
import roster_generator
```

For local development from a source checkout:

```bash
cd roster
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Runtime dependencies are declared in `pyproject.toml` and include `pandas`,
`numpy`, `airportsdata`, `aircraft-list`, and `pytz`. Development and release
tools, including `pytest`, `build`, and `twine`, are available through the
optional `dev` extra.

## Input Data

The main ROSTER pipeline expects a cleaned flight schedule with this normalized
schema:

```text
DEP_ICAO, ARR_ICAO, STD_REFTZ, STA_REFTZ, ATD_REFTZ, ATA_REFTZ,
AC_OPER, AC_REG, AC_WAKE
```

Any dataset with those columns can be used directly as `schedule_file` in
`PipelineConfig`. The built-in cleaning step is optional: it is a convenience
utility for converting EUROCONTROL-style flight records into the normalized
ROSTER input format.

For that optional cleaner, the raw EUROCONTROL-style file must include, at
minimum, the following source columns:

```text
ADEP, ADES, FILED OFF BLOCK TIME, FILED ARRIVAL TIME,
ACTUAL OFF BLOCK TIME, ACTUAL ARRIVAL TIME,
AC Type, AC Operator, AC Registration
```

The cleaner validates airport ICAO codes, parses timestamps, adds ICAO wake
turbulence categories from aircraft type data, and writes the normalized schema
above.

## Using BTS Data

ROSTER can also clean BTS on-time performance downloads into the same normalized
schema. Download the BTS source data yourself, extract the CSV files into
`BTS/`, and run the BTS cleaning tutorial:

```bash
mkdir -p BTS
# Put the extracted BTS on-time schedule CSV and aircraft inventory CSV in BTS/.
# The tutorial auto-detects BTS/on_time.csv, BTS/aircraft.csv, or the default
# extracted BTS filenames shown below.
python tutorials/tutorial_bts_cleaning.py
python main.py --schedule-file input/bts_clean.csv --seed 42 --suffix bts
```

You can also call the cleaner directly by passing both extracted CSV paths:

```bash
python -m roster_generator.data_cleaning.clean_bts \
  BTS/On_Time_Reporting_Carrier_On_Time_Performance_\(1987_present\)_2024_12.csv \
  BTS/T_F41SCHEDULE_B43.csv \
  --output input/bts_clean.csv
```

For example, from a user workspace such as
`/home/josu/Escritorio/main_pc/roster-workspace`, the command above expects
the extracted CSV files under
`/home/josu/Escritorio/main_pc/roster-workspace/BTS`.

The two required BTS CSV inputs are:

| Need | BTS table | Where to get it | Filename |
| --- | --- | --- | --- |
| Required | Reporting Carrier On-Time Performance (1987-present) schedule | <https://transtats.bts.gov/PREZIP/> | Extracted CSV, for example `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv` |
| Required | Schedule B-43 aircraft inventory | <https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=Nv4+Pn44vr4+Sv0n0pvny&gnoyr_VQ=GEH> | Extracted CSV, for example `T_F41SCHEDULE_B43.csv` |

BTS on-time field definitions are documented at
<https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ>. BTS Schedule B-43 fields
are documented at <https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=GEH>.

ROSTER does not download, cache, or read zipped BTS support data. Both the
on-time schedule CSV and the Schedule B-43 aircraft inventory CSV are mandatory.

BTS stores its scheduled and actual clock times as local `hhmm` values. The BTS
cleaner localizes those times using each airport timezone, converts them into
the normalized ROSTER timestamp convention, and writes the standard cleaned
columns shown above.

## Quick Start

Run the full tutorial pipeline from a source checkout:

```bash
python tutorials/tutorial_pipeline.py --seed 42 --suffix demo
```

The seed controls stochastic sampling. The optional suffix is appended to
generated files, so `--suffix demo` produces outputs such as
`schedule_demo.csv`.

## Pipeline

The standard tutorial workflow runs the following stages:

1. Optionally clean historical flight records into the normalized ROSTER schema.
2. Build empirical Markov transition tables and sample fleet initial
   conditions.
3. Analyze scheduled turnaround and flight-time distributions.
4. Generate auxiliary simulator input files for airlines, airports, fleet, and
   routes.
5. Generate a synthetic schedule through greedy forward construction with
   airport capacity checks.

Intermediate analysis files are written under `computed/`, including
`initial_conditions`, `markov`, scheduled flight-time distributions, and
turnaround profiles. Simulator-facing outputs are written under `output/`,
including `airlines`, `airports`, `fleet`, `routes`, `phys_ta`, and `schedule`.

## Configuration

Runtime window behavior can be configured in `tutorials/params.yaml`:

```yaml
REFTZ: UTC
WINDOW_START: "00:00"
WINDOW_LENGTH_HOURS: 24
ACTUAL_TIMES: false
```

`REFTZ` defines the reference timezone used for time-of-day and day-boundary
logic. `WINDOW_START` and `WINDOW_LENGTH_HOURS` define the simulated operating
window. `ACTUAL_TIMES` controls whether actual timestamp columns are required
and used by stages that support them.

Programmatic workflows use `roster_generator.PipelineConfig` to define input
paths, output paths, random seed, suffix, time-window settings, and optional
manipulation callbacks.

## Schedule manipulation

ROSTER supports controlled scenario manipulation without editing the generated
analysis tables by hand:

- `manipulation_fn` modifies scalar distribution parameters at runtime, such as
  turnaround distributions, fleet-size parameters, route durations, physical
  turnaround minima, and prior-day probabilities.
- `markov_manipulation_fn` receives a `MarkovContext` and reweights existing
  destination probabilities before each Markov row is normalized.

See `tutorials/tutorial_manipulation.py` for a worked example of both hooks.

## Some Features Of ROSTER

- Written in Python 3 and installable with pip.
- Reproducible stochastic generation through explicit random seeds.
- Empirical Markov transition models for aircraft continuation behavior.
- Synthetic fleet initial-condition sampling from historical records.
- Reference-timezone operating windows with configurable start time and length.
- Wake-category handling based on aircraft type data.
- Scheduled turnaround and flight-time distribution analysis.
- Airport capacity-aware schedule construction.
- CSV outputs suitable for downstream simulation workflows.
- Unit tests covering cleaning, configuration, distributions, Markov models,
  auxiliary files, and schedule generation.

## Testing

Run the test suite from the project root:

```bash
python -m pytest
```

## Contributions

Contributions are welcome from researchers and developers. Please
keep changes focused, add or update tests for behavioural changes.

## License

ROSTER is distributed under the GNU General Public License v3. See `LICENSE`
for the full license text.
