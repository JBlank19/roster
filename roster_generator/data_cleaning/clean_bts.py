"""Clean BTS on-time data into the normalized ROSTER schedule schema.

BTS source tables used by this cleaner:

- Reporting Carrier On-Time Performance (1987-present)
  Fields: https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
  Monthly downloads: https://transtats.bts.gov/PREZIP/
  Extracted CSV example filename:
  ``On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv``

- Schedule B-43 Inventory
  Fields: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=GEH
  Download page:
  https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=Nv4+Pn44vr4+Sv0n0pvny&gnoyr_VQ=GEH
  Extracted CSV example filename: ``T_F41SCHEDULE_B43.csv``
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import airportsdata
import pandas as pd
import pytz

from .clean_data import FINAL_COLUMNS, _load_wake_map

ON_TIME_FIELDS_URL = "https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ"
PREZIP_URL = "https://transtats.bts.gov/PREZIP/"
B43_FIELDS_URL = "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=GEH"
B43_DOWNLOAD_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx?"
    "QO_fu146_anzr=Nv4+Pn44vr4+Sv0n0pvny&gnoyr_VQ=GEH"
)

ON_TIME_FILENAME_EXAMPLE = (
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv"
)
B43_FILENAME_EXAMPLE = "T_F41SCHEDULE_B43.csv"

ON_TIME_REQUIRED_COLUMNS = {
    "Year",
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "DepTime",
    "ArrTime",
    "DepDelay",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
}
B43_REQUIRED_COLUMNS = {
    "YEAR",
    "TAIL_NUMBER",
    "MANUFACTURER",
    "MODEL",
    "UNIQUE_CARRIER",
}

BTS_HELP_TEXT = f"""Provide two extracted BTS CSV files:
  1. BTS on-time schedule CSV, e.g. {ON_TIME_FILENAME_EXAMPLE}
  2. Schedule B-43 aircraft inventory CSV, e.g. {B43_FILENAME_EXAMPLE}

ROSTER does not download or read BTS zip archives. Download the BTS source data
yourself and extract the CSV files before running this cleaner.

On-time downloads: {PREZIP_URL}
Schedule B-43 download page: {B43_DOWNLOAD_URL}
"""


@dataclass
class BTSCleanReport:
    """Summary returned by :func:`clean_bts_data`."""

    output_file: Path
    schedule_csv: Path
    aircraft_csv: Path
    rows_read: int
    completed_rows: int
    rows_written: int
    dropped_cancelled_diverted: int
    dropped_airports: int
    dropped_missing_tail: int
    dropped_missing_wake: int

    def __str__(self) -> str:
        return (
            "[BTS Clean] "
            f"schedule_csv={self.schedule_csv}, "
            f"aircraft_csv={self.aircraft_csv}, "
            f"rows_read={self.rows_read}, "
            f"completed_rows={self.completed_rows}, "
            f"dropped_cancelled_diverted={self.dropped_cancelled_diverted}, "
            f"dropped_airports={self.dropped_airports}, "
            f"dropped_missing_tail={self.dropped_missing_tail}, "
            f"dropped_missing_wake={self.dropped_missing_wake}, "
            f"rows_written={self.rows_written}, "
            f"output={self.output_file}"
        )


def clean_bts_data(
    schedule_csv: str | Path,
    aircraft_csv: str | Path,
    clean_file: str | Path,
) -> BTSCleanReport:
    """Clean BTS on-time data and write the normalized ROSTER schema.

    Parameters
    ----------
    schedule_csv:
        Extracted BTS Reporting Carrier On-Time Performance CSV, for example
        ``On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv``.
    aircraft_csv:
        Extracted Schedule B-43 aircraft inventory CSV, for example
        ``T_F41SCHEDULE_B43.csv``.
    clean_file:
        Output CSV path.
    """
    output_file = Path(clean_file)
    schedule_path = _validate_input_csv(schedule_csv, "BTS on-time schedule CSV")
    aircraft_path = _validate_input_csv(
        aircraft_csv,
        "Schedule B-43 aircraft inventory CSV",
    )

    on_time = _read_required_csv(
        schedule_path,
        ON_TIME_REQUIRED_COLUMNS,
        "BTS on-time schedule CSV",
        ON_TIME_FIELDS_URL,
    )
    b43_df = _read_required_csv(
        aircraft_path,
        B43_REQUIRED_COLUMNS,
        "Schedule B-43 aircraft inventory CSV",
        B43_FIELDS_URL,
    )
    rows_read = len(on_time)

    cleaned, counters = _clean_bts_frame(on_time, b43_df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_file, index=False, date_format="%Y-%m-%d %H:%M:%S")

    report = BTSCleanReport(
        output_file=output_file,
        schedule_csv=schedule_path,
        aircraft_csv=aircraft_path,
        rows_read=rows_read,
        completed_rows=counters["completed_rows"],
        rows_written=len(cleaned),
        dropped_cancelled_diverted=counters["dropped_cancelled_diverted"],
        dropped_airports=counters["dropped_airports"],
        dropped_missing_tail=counters["dropped_missing_tail"],
        dropped_missing_wake=counters["dropped_missing_wake"],
    )
    print(report)
    return report


def _validate_input_csv(path: str | Path, label: str) -> Path:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[BTS Clean] {label} not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"[BTS Clean] {label} must be a CSV file: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            f"[BTS Clean] {label} must be a .csv file: {csv_path}. "
            "Zip archives are not accepted; extract the CSV before running ROSTER."
        )
    return csv_path


def _read_required_csv(
    csv_path: Path,
    required_columns: set[str],
    label: str,
    fields_url: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    missing = sorted(required_columns.difference(df.columns.astype(str)))
    if missing:
        raise ValueError(
            f"[BTS Clean] {label} is missing required columns: "
            f"{', '.join(missing)}. Field reference: {fields_url}"
        )
    return df


def _clean_bts_frame(on_time: pd.DataFrame, b43: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = on_time.copy()
    cancelled = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0)
    diverted = pd.to_numeric(df["Diverted"], errors="coerce").fillna(0)
    completed_mask = (cancelled == 0) & (diverted == 0)
    df = df.loc[completed_mask].copy()

    df["TAIL_NORM"] = df["Tail_Number"].map(_normalize_tail)
    before_tail = len(df)
    df = df[df["TAIL_NORM"] != ""].copy()
    after_tail = len(df)

    airports_iata = airportsdata.load("IATA")
    df["DEP_ICAO"] = df["Origin"].map(lambda value: _iata_to_icao(value, airports_iata))
    df["ARR_ICAO"] = df["Dest"].map(lambda value: _iata_to_icao(value, airports_iata))
    df["DEP_TZ"] = df["Origin"].map(lambda value: _iata_to_tz(value, airports_iata))
    df["ARR_TZ"] = df["Dest"].map(lambda value: _iata_to_tz(value, airports_iata))
    before_airports = len(df)
    df = df.dropna(subset=["DEP_ICAO", "ARR_ICAO", "DEP_TZ", "ARR_TZ"]).copy()
    after_airports = len(df)

    wake_lookup = _build_tail_wake_lookup(b43, df)
    df["AC_WAKE"] = df["TAIL_NORM"].map(wake_lookup)
    before_wake = len(df)
    df = df.dropna(subset=["AC_WAKE"]).copy()
    df = df[df["AC_WAKE"].astype(str).str.strip() != ""].copy()
    after_wake = len(df)

    df["STD_REFTZ"] = df.apply(
        lambda row: _scheduled_departure_utc(row["FlightDate"], row["CRSDepTime"], row["DEP_TZ"]),
        axis=1,
    )
    elapsed = pd.to_numeric(df["CRSElapsedTime"], errors="coerce")
    df["STA_REFTZ"] = df["STD_REFTZ"] + pd.to_timedelta(elapsed, unit="m")
    dep_delay = pd.to_numeric(df["DepDelay"], errors="coerce")
    arr_delay = pd.to_numeric(df["ArrDelay"], errors="coerce")
    df["ATD_REFTZ"] = df["STD_REFTZ"] + pd.to_timedelta(dep_delay, unit="m")
    df["ATA_REFTZ"] = df["STA_REFTZ"] + pd.to_timedelta(arr_delay, unit="m")
    df = df.dropna(subset=["STD_REFTZ", "STA_REFTZ", "ATD_REFTZ", "ATA_REFTZ"])

    df["AC_OPER"] = df["Reporting_Airline"].astype(str).str.strip()
    df["AC_REG"] = df["TAIL_NORM"]
    df = df[FINAL_COLUMNS].copy()

    counters = {
        "completed_rows": int(completed_mask.sum()),
        "dropped_cancelled_diverted": int((~completed_mask).sum()),
        "dropped_missing_tail": before_tail - after_tail,
        "dropped_airports": before_airports - after_airports,
        "dropped_missing_wake": before_wake - after_wake,
    }
    return df, counters


def _iata_to_icao(value: object, airports_iata: dict[str, dict[str, object]]) -> str | None:
    item = airports_iata.get(str(value).strip().upper())
    if not item:
        return None
    icao = item.get("icao")
    return str(icao).strip().upper() if icao else None


def _iata_to_tz(value: object, airports_iata: dict[str, dict[str, object]]) -> str | None:
    item = airports_iata.get(str(value).strip().upper())
    if not item:
        return None
    tz_name = item.get("tz")
    return str(tz_name).strip() if tz_name else None


def _normalize_tail(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    if not text or text == "NAN":
        return ""
    if text.startswith("N"):
        return text
    if text[0].isdigit():
        return f"N{text}"
    return text


def _scheduled_departure_utc(flight_date: object, hhmm: object, tz_name: object) -> pd.Timestamp:
    parsed_date = pd.to_datetime(flight_date, errors="coerce")
    if pd.isna(parsed_date):
        return pd.NaT
    parsed_time = _parse_hhmm(hhmm)
    if parsed_time is None:
        return pd.NaT
    hour, minute, day_offset = parsed_time
    local_naive = datetime(
        parsed_date.year,
        parsed_date.month,
        parsed_date.day,
        hour,
        minute,
    ) + timedelta(days=day_offset)
    try:
        local_tz = pytz.timezone(str(tz_name))
        local_aware = local_tz.localize(local_naive, is_dst=None)
    except Exception:
        return pd.NaT
    return pd.Timestamp(local_aware.astimezone(pytz.UTC).replace(tzinfo=None))


def _parse_hhmm(value: object) -> tuple[int, int, int] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split(".", 1)[0].zfill(4)
    if not text.isdigit():
        return None
    hour = int(text[:-2])
    minute = int(text[-2:])
    if hour == 24 and minute == 0:
        return 0, 0, 1
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute, 0


def _build_tail_wake_lookup(b43: pd.DataFrame, on_time: pd.DataFrame) -> dict[str, str]:
    b43_prepared = b43.copy()
    b43_prepared["TAIL_NORM"] = b43_prepared["TAIL_NUMBER"].map(_normalize_tail)
    b43_prepared["YEAR_NUM"] = pd.to_numeric(b43_prepared.get("YEAR"), errors="coerce")
    b43_prepared["UNIQUE_CARRIER_NORM"] = (
        b43_prepared.get("UNIQUE_CARRIER", "")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    b43_prepared["ICAO_TYPE"] = b43_prepared.apply(_infer_icao_from_b43_row, axis=1)
    wake_map = _load_wake_map()
    b43_prepared["AC_WAKE"] = b43_prepared["ICAO_TYPE"].map(wake_map)
    b43_prepared["AC_WAKE"] = b43_prepared["AC_WAKE"].replace({"L/M": "M", "M/H": "H"})
    b43_prepared = b43_prepared.dropna(subset=["TAIL_NORM", "AC_WAKE"])

    lookup: dict[str, str] = {}
    unique_keys = on_time[["TAIL_NORM", "Year", "Reporting_Airline"]].drop_duplicates()
    for row in unique_keys.itertuples(index=False):
        tail = row.TAIL_NORM
        if tail in lookup:
            continue
        candidates = b43_prepared[b43_prepared["TAIL_NORM"] == tail].copy()
        if candidates.empty:
            continue
        year = pd.to_numeric(pd.Series([row.Year]), errors="coerce").iloc[0]
        carrier = str(row.Reporting_Airline).strip().upper()
        candidates["SCORE"] = 0
        candidates.loc[candidates["UNIQUE_CARRIER_NORM"] == carrier, "SCORE"] -= 1000
        if not pd.isna(year):
            candidates["SCORE"] += (candidates["YEAR_NUM"] - float(year)).abs().fillna(100)
        selected = candidates.sort_values(["SCORE", "YEAR_NUM"]).iloc[0]
        lookup[tail] = str(selected["AC_WAKE"]).strip().upper()
    return lookup


def _infer_icao_from_b43_row(row: pd.Series) -> str | None:
    manufacturer = _normalize_aircraft_token(row.get("MANUFACTURER", ""))
    model = _normalize_aircraft_token(row.get("MODEL", ""))
    aircraft_type = _normalize_aircraft_token(row.get("AIRCRAFT_TYPE", ""))
    if aircraft_type and aircraft_type.isalpha():
        return aircraft_type
    return _infer_icao_from_model(manufacturer, model)


def _normalize_aircraft_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return "" if text == "NAN" else text


def _infer_icao_from_model(manufacturer: str, model: str) -> str | None:
    manufacturer = {
        "THEBOEINGCO": "BOEING",
        "THEBOEINGCOMPANY": "BOEING",
        "BOEINGCO": "BOEING",
        "BOEINGCOMPANY": "BOEING",
        "AIRBUSINDUSTRIE": "AIRBUS",
        "AIRBUSINDUSTRIES": "AIRBUS",
        "AIRBUSSAS": "AIRBUS",
        "BOMBARDIERAEROSPACE": "BOMBARDIER",
    }.get(manufacturer, manufacturer)

    if model.startswith("BD5001A10") or model.startswith("A220100"):
        return "BCS1"
    if model.startswith("BD5001A11") or model.startswith("A220300"):
        return "BCS3"
    if manufacturer == "AIRBUS" or model.startswith(("A3", "A22", "BD500")):
        if model.startswith("A300"):
            return "A306"
        match = re.match(r"^A?(31[89]|32[01])", model)
        if match:
            family = match.group(1)
            neo = model.endswith("N") or re.search(r"2[57][0-9]N", model)
            return {
                "318": "A318",
                "319": "A19N" if neo else "A319",
                "320": "A20N" if neo else "A320",
                "321": "A21N" if neo else "A321",
            }[family]
        if model.startswith("A330"):
            if re.match(r"A?3309", model):
                return "A339"
            if re.match(r"A?3308", model):
                return "A338"
            if re.match(r"A?3302", model):
                return "A332"
            if re.match(r"A?3303", model):
                return "A333"
            return "A330"
        if model.startswith("A350"):
            return "A359" if "900" in model else "A35K" if "1000" in model else "A350"

    if manufacturer == "BOEING" or model.startswith(("B7", "BOEING7")):
        boeing = model.replace("BOEING", "")
        boeing = boeing[1:] if boeing.startswith("B") else boeing
        if boeing.startswith("717"):
            return "B712"
        if boeing.startswith("727"):
            return "B727"
        if boeing.startswith("737"):
            rest = boeing[3:]
            if re.match(r"10", rest):
                return "B3XM"
            if re.match(r"9$", rest):
                return "B39M"
            if re.match(r"8$", rest):
                return "B38M"
            if re.match(r"[89][A-Z0-9]{2}", rest):
                return "B738" if rest[0] == "8" else "B739"
            if re.match(r"7[A-Z0-9]{2}", rest):
                return "B737"
            for prefix, icao in [
                ("6", "B736"),
                ("5", "B735"),
                ("4", "B734"),
                ("3", "B733"),
                ("2", "B732"),
            ]:
                if rest.startswith(prefix):
                    return icao
            return "B737"
        if boeing.startswith("747"):
            return "B748" if re.match(r"7478", boeing) else "B744"
        if boeing.startswith("757"):
            return "B752" if re.match(r"7572", boeing) else "B753"
        if boeing.startswith("767"):
            if re.match(r"7672", boeing):
                return "B762"
            if re.match(r"7673", boeing) or "300F" in boeing:
                return "B763"
            return "B764" if re.match(r"7674", boeing) else "B767"
        if boeing.startswith("777F") or re.match(r"777F", boeing):
            return "B77L"
        if boeing.startswith("777"):
            if "300ER" in boeing:
                return "B77W"
            return "B772" if re.match(r"7772", boeing) else "B773"
        if boeing.startswith("787"):
            if re.match(r"7878", boeing):
                return "B788"
            if re.match(r"7879", boeing):
                return "B789"
            return "B78X" if re.match(r"78710", boeing) else "B787"

    if manufacturer in {"BOMBARDIER", "CANADAIR", "GE"} or model.startswith("CL600"):
        if any(token in model for token in ["2B19", "2C11", "CRJ100", "CRJ200", "CRJ440"]):
            return "CRJ2"
        if any(token in model for token in ["2C10", "CRJ700", "CRJ701", "CRJ705"]):
            return "CRJ7"
        if "2D24" in model or "CRJ900" in model:
            return "CRJ9"
        if "2E25" in model or "CRJ1000" in model:
            return "CRJX"
    if manufacturer == "EMBRAER" or model.startswith(("ERJ", "EMB")):
        if re.search(r"(170|ERJ170100)", model):
            return "E170"
        if re.search(r"(175|ERJ170200)", model):
            return "E75L"
        if re.search(r"(190|ERJ190100)", model):
            return "E190"
        if re.search(r"(195|ERJ190200)", model):
            return "E195"
        if re.search(r"(145|EMB145|ERJ145)", model):
            return "E145"
        if re.search(r"(140|135|ERJ140|ERJ135)", model):
            return "E135"
    if "DHC8" in model or "DASH8" in model:
        return "DH8D" if "400" in model else "DH8C" if "300" in model else "DH8B"
    if manufacturer == "CESSNA" and (model.startswith("C208") or model.startswith("208")):
        return "C208"
    if manufacturer in {"MCDONNELLDOUGLAS", "DOUGLAS"} or model.startswith(("MD", "DC9")):
        if model.startswith("MD11"):
            return "MD11"
        if model.startswith(("MD80", "MD81", "MD82", "MD83", "MD87", "MD88", "DC9")):
            return "MD80"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean BTS on-time data into the normalized ROSTER schema.",
        epilog=BTS_HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "schedule_csv",
        help="Extracted BTS on-time schedule CSV.",
    )
    parser.add_argument(
        "aircraft_csv",
        help="Extracted Schedule B-43 aircraft inventory CSV.",
    )
    parser.add_argument("--output", required=True, help="Path to the cleaned output CSV.")
    args = parser.parse_args()
    clean_bts_data(
        args.schedule_csv,
        args.aircraft_csv,
        args.output,
    )


if __name__ == "__main__":
    main()
