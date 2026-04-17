"""Tests for BTS cleaning support."""

from pathlib import Path

import pandas as pd
import pytest

from roster_generator.data_cleaning.clean_bts import clean_bts_data


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _on_time_row(**overrides):
    row = {
        "Year": "2024",
        "FlightDate": "2024-12-01",
        "Reporting_Airline": "B6",
        "Tail_Number": "535JB",
        "Origin": "JFK",
        "Dest": "LAX",
        "CRSDepTime": "0800",
        "CRSArrTime": "1100",
        "DepTime": "0810",
        "ArrTime": "1115",
        "DepDelay": "10",
        "ArrDelay": "15",
        "Cancelled": "0",
        "Diverted": "0",
        "CRSElapsedTime": "360",
    }
    row.update(overrides)
    return row


def _b43_row(**overrides):
    row = {
        "YEAR": "2024",
        "TAIL_NUMBER": "N535JB",
        "MANUFACTURER": "AirbusIndustrie",
        "AIRCRAFT_TYPE": "",
        "MODEL": "A320-232",
        "NUMBER_OF_SEATS": "150",
        "UNIQUE_CARRIER": "B6",
    }
    row.update(overrides)
    return row


def _without(row: dict[str, str], column: str) -> dict[str, str]:
    copy = row.copy()
    del copy[column]
    return copy


def test_clean_bts_data_from_required_schedule_and_aircraft_csvs(tmp_path):
    source_dir = tmp_path / "BTS"
    schedule_csv = _write_csv(
        source_dir / "on_time.csv",
        pd.DataFrame([_on_time_row()]),
    )
    aircraft_csv = _write_csv(
        source_dir / "aircraft.csv",
        pd.DataFrame([_b43_row()]),
    )

    out = tmp_path / "input" / "bts_clean.csv"
    report = clean_bts_data(schedule_csv, aircraft_csv, out)

    assert out.exists()
    assert report.schedule_csv == schedule_csv
    assert report.aircraft_csv == aircraft_csv
    assert report.rows_read == 1
    assert report.rows_written == 1
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "DEP_ICAO",
        "ARR_ICAO",
        "STD_REFTZ",
        "STA_REFTZ",
        "ATD_REFTZ",
        "ATA_REFTZ",
        "AC_OPER",
        "AC_REG",
        "AC_WAKE",
    ]
    row = df.iloc[0]
    assert row["DEP_ICAO"] == "KJFK"
    assert row["ARR_ICAO"] == "KLAX"
    assert row["AC_REG"] == "N535JB"
    assert row["AC_WAKE"] == "M"
    assert row["STD_REFTZ"] == "2024-12-01 13:00:00"
    assert row["STA_REFTZ"] == "2024-12-01 19:00:00"
    assert row["ATD_REFTZ"] == "2024-12-01 13:10:00"
    assert row["ATA_REFTZ"] == "2024-12-01 19:15:00"


def test_clean_bts_data_missing_schedule_csv_has_actionable_error(tmp_path):
    aircraft_csv = _write_csv(tmp_path / "aircraft.csv", pd.DataFrame([_b43_row()]))

    with pytest.raises(FileNotFoundError, match="BTS on-time schedule CSV not found"):
        clean_bts_data(tmp_path / "missing.csv", aircraft_csv, tmp_path / "out.csv")


def test_clean_bts_data_missing_aircraft_csv_has_actionable_error(tmp_path):
    schedule_csv = _write_csv(tmp_path / "schedule.csv", pd.DataFrame([_on_time_row()]))

    with pytest.raises(
        FileNotFoundError,
        match="Schedule B-43 aircraft inventory CSV not found",
    ):
        clean_bts_data(schedule_csv, tmp_path / "missing.csv", tmp_path / "out.csv")


def test_clean_bts_data_rejects_non_csv_inputs(tmp_path):
    schedule_zip = tmp_path / "on_time.zip"
    schedule_zip.write_text("not a csv", encoding="utf-8")
    aircraft_csv = _write_csv(tmp_path / "aircraft.csv", pd.DataFrame([_b43_row()]))

    with pytest.raises(ValueError, match="must be a \\.csv file"):
        clean_bts_data(schedule_zip, aircraft_csv, tmp_path / "out.csv")


def test_clean_bts_data_validates_schedule_columns(tmp_path):
    schedule_csv = _write_csv(
        tmp_path / "schedule.csv",
        pd.DataFrame([_without(_on_time_row(), "Year")]),
    )
    aircraft_csv = _write_csv(tmp_path / "aircraft.csv", pd.DataFrame([_b43_row()]))

    with pytest.raises(
        ValueError,
        match="BTS on-time schedule CSV is missing required columns: Year",
    ):
        clean_bts_data(schedule_csv, aircraft_csv, tmp_path / "out.csv")


def test_clean_bts_data_validates_aircraft_columns(tmp_path):
    schedule_csv = _write_csv(tmp_path / "schedule.csv", pd.DataFrame([_on_time_row()]))
    aircraft_csv = _write_csv(
        tmp_path / "aircraft.csv",
        pd.DataFrame([_without(_b43_row(), "TAIL_NUMBER")]),
    )

    with pytest.raises(
        ValueError,
        match="Schedule B-43 aircraft inventory CSV is missing required columns: TAIL_NUMBER",
    ):
        clean_bts_data(schedule_csv, aircraft_csv, tmp_path / "out.csv")
