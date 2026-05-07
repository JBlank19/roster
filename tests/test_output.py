"""Tests for ROSTER output routing."""

from pathlib import Path

import pandas as pd
import pytest

from roster_generator.auxiliary.airlines import generate_airlines
from roster_generator.config import PipelineConfig
from roster_generator.output import (
    reset_log_file,
    resolve_log_file,
    roster_print,
    validate_output_mode,
)


def test_roster_print_defaults_to_terminal(capsys):
    roster_print("hello", "ROSTER")

    captured = capsys.readouterr()
    assert captured.out == "hello ROSTER\n"


def test_roster_print_writes_to_file(tmp_path):
    log_file = tmp_path / "roster.log"

    roster_print("hello", "file", output_mode="file", log_file=log_file)

    assert log_file.read_text(encoding="utf-8") == "hello file\n"


def test_non_verbose_suppresses_terminal_and_file(capsys, tmp_path):
    log_file = tmp_path / "quiet.log"

    roster_print("hidden", output_mode="non-verbose", log_file=log_file)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert not log_file.exists()


def test_invalid_output_mode_raises():
    with pytest.raises(ValueError, match="Invalid output mode"):
        validate_output_mode("chatty")


def test_reset_log_file_truncates_existing_file(tmp_path):
    log_file = tmp_path / "roster.log"
    log_file.write_text("old content\n", encoding="utf-8")

    resolved = reset_log_file(output_mode="file", log_file=log_file)

    assert resolved == log_file
    assert log_file.read_text(encoding="utf-8") == ""


def test_automatic_log_file_uses_log_dir_and_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = PipelineConfig(
        schedule_file=tmp_path / "schedule.csv",
        analysis_dir=tmp_path / "analysis",
        output_dir=tmp_path / "output",
        suffix="_demo",
        output_mode="file",
    )

    assert cfg.resolved_log_file() == Path("log") / "roster_demo.log"
    assert resolve_log_file(config=cfg) == Path("log") / "roster_demo.log"

    roster_print("automatic path", config=cfg)

    assert (tmp_path / "log" / "roster_demo.log").read_text(encoding="utf-8") == "automatic path\n"


def _write_initial_conditions(path: Path) -> None:
    pd.DataFrame(
        {
            "AC_OPER": ["IBE", "RYR"],
            "AC_REG": ["ECAAA", "ECBBB"],
            "AC_WAKE": ["M", "M"],
        }
    ).to_csv(path, index=False)


def test_generator_file_mode_routes_stage_output(tmp_path):
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "output"
    analysis_dir.mkdir()
    _write_initial_conditions(analysis_dir / "initial_conditions.csv")
    log_file = tmp_path / "log" / "stage.log"
    cfg = PipelineConfig(
        schedule_file=tmp_path / "schedule.csv",
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        output_mode="file",
        log_file=log_file,
    )

    generate_airlines(cfg)

    log_text = log_file.read_text(encoding="utf-8")
    assert "[Airlines] --- AIRLINE CATALOGUE ---" in log_text
    assert "[Airlines] --- SUCCESS ---" in log_text
    assert (output_dir / "airlines.csv").exists()


def test_generator_non_verbose_suppresses_stage_output(capsys, tmp_path):
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "output"
    analysis_dir.mkdir()
    _write_initial_conditions(analysis_dir / "initial_conditions.csv")
    cfg = PipelineConfig(
        schedule_file=tmp_path / "schedule.csv",
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        output_mode="non-verbose",
    )

    generate_airlines(cfg)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert (output_dir / "airlines.csv").exists()
