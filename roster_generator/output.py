"""Shared output routing for ROSTER status messages."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Protocol, cast

OutputMode = Literal["terminal", "file", "non-verbose"]

DEFAULT_OUTPUT_MODE: OutputMode = "terminal"
VALID_OUTPUT_MODES: tuple[OutputMode, ...] = ("terminal", "file", "non-verbose")


class OutputConfig(Protocol):
    """Minimal config surface needed by the output router."""

    output_mode: OutputMode
    log_file: Path | None
    suffix: str


def validate_output_mode(value: object) -> OutputMode:
    """Validate and normalize a ROSTER output mode."""
    text = str(value).strip().lower()
    if text not in VALID_OUTPUT_MODES:
        modes = ", ".join(VALID_OUTPUT_MODES)
        raise ValueError(f"Invalid output mode {value!r}. Expected one of: {modes}.")
    return cast(OutputMode, text)


def validate_log_file(value: object) -> Path | None:
    """Validate an optional log-file value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def resolve_log_file(
    *,
    config: OutputConfig | None = None,
    log_file: str | Path | None = None,
    suffix: str | None = None,
) -> Path:
    """Resolve the file used when routed output is in file mode."""
    explicit_log_file = log_file
    if explicit_log_file is None and config is not None:
        explicit_log_file = config.log_file
    if explicit_log_file is not None:
        return Path(explicit_log_file)

    resolved_suffix = suffix
    if resolved_suffix is None and config is not None:
        resolved_suffix = config.suffix
    return Path("log") / f"roster{resolved_suffix or ''}.log"


def _mode_from_args(
    *,
    config: OutputConfig | None,
    output_mode: OutputMode | str | None,
) -> OutputMode:
    if output_mode is not None:
        return validate_output_mode(output_mode)
    if config is not None:
        return validate_output_mode(config.output_mode)
    return DEFAULT_OUTPUT_MODE


def roster_print(
    *values: object,
    config: OutputConfig | None = None,
    output_mode: OutputMode | str | None = None,
    log_file: str | Path | None = None,
    sep: str = " ",
    end: str = "\n",
) -> None:
    """Route a status message to terminal, file, or nowhere."""
    mode = _mode_from_args(config=config, output_mode=output_mode)
    if mode == "non-verbose":
        return
    if mode == "terminal":
        print(*values, sep=sep, end=end)
        return

    path = resolve_log_file(config=config, log_file=log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = sep.join(str(value) for value in values)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.write(end)


def reset_log_file(
    *,
    config: OutputConfig | None = None,
    output_mode: OutputMode | str | None = None,
    log_file: str | Path | None = None,
    suffix: str | None = None,
) -> Path | None:
    """Truncate the active log file when output is routed to a file."""
    mode = _mode_from_args(config=config, output_mode=output_mode)
    if mode != "file":
        return None

    path = resolve_log_file(config=config, log_file=log_file, suffix=suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def add_output_arguments(parser: ArgumentParser) -> None:
    """Add standard output-routing arguments to an argparse parser."""
    parser.add_argument(
        "--output-mode",
        choices=VALID_OUTPUT_MODES,
        default=None,
        help="Route status output to terminal, file, or suppress it.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path used with --output-mode file.",
    )
