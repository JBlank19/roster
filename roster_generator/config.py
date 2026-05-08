"""Central configuration for the pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Literal, Mapping

from .output import (
    DEFAULT_OUTPUT_MODE,
    OutputMode,
    resolve_log_file,
    validate_log_file,
    validate_output_mode,
)
from .time_window import (
    DEFAULT_ACTUAL_TIMES,
    DEFAULT_REFTZ,
    DEFAULT_SAVE_COMPUTED,
    DEFAULT_WINDOW_LENGTH_HOURS,
    DEFAULT_WINDOW_START,
    validate_actual_times,
    validate_reftz,
    validate_window_length_hours,
    validate_window_start,
    window_start_to_minutes,
)

ManipulationFn = Callable[[dict[str, float], str], dict[str, float]]


@dataclass(frozen=True)
class MarkovContext:
    """Metadata exposed to user Markov manipulation callbacks."""

    table_kind: Literal["primary", "fallback"]
    airline: str
    wake: str
    prev_origin: str | None
    origin: str
    dep_hour_reftz: int
    base_probs: Mapping[str, float]
    base_counts: Mapping[str, int]


MarkovManipulationFn = Callable[
    [dict[str, float], MarkovContext],
    dict[str, float] | None,
]


def _default_manipulation(params: dict[str, float], dtype: str) -> dict[str, float]:
    """Identity manipulation: returns parameters unchanged."""
    return params


def _default_markov_manipulation(
    _params: dict[str, float],
    _context: MarkovContext,
) -> None:
    """Identity Markov manipulation: leaves transition weights unchanged."""
    return None


@dataclass
class PipelineConfig:
    """All paths and parameters every pipeline step needs.

    Parameters
    ----------
    schedule_file : Path
        Cleaned CSV (e.g. ``september2023.csv``).
    analysis_dir : Path
        Intermediate analysis outputs (markov, turnaround params, etc.).
    output_dir : Path
        Final outputs consumed by the simulation (fleet, airports, schedule, etc.).
    seed : int
        Master RNG seed.  Passed to numpy / random.
    suffix : str
        Optional file-name suffix.
    reftz : str
        Reference timezone for time-of-day/day-boundary logic.
    window_start : str
        Window start in HH:MM in reference timezone.
    window_length_hours : int
        Window length in hours (1..24).
    actual_times : bool
        Whether actual timestamp columns are required and used.
    output_mode : {"terminal", "file", "non-verbose"}
        Where ROSTER status messages are routed.
    log_file : Path | None
        Optional explicit log path used when output_mode is ``"file"``.
    save_computed : bool
        Whether to keep intermediate analysis files after the pipeline
        completes.  Set to ``False`` to delete them automatically (saves
        disk space on repeated runs).  Default is ``True``.
    """

    schedule_file: Path
    analysis_dir: Path
    output_dir: Path
    seed: int = 42
    suffix: str = ""
    reftz: str = DEFAULT_REFTZ
    window_start: str = DEFAULT_WINDOW_START
    window_length_hours: int = DEFAULT_WINDOW_LENGTH_HOURS
    actual_times: bool = DEFAULT_ACTUAL_TIMES
    output_mode: OutputMode = DEFAULT_OUTPUT_MODE
    log_file: Path | None = None
    save_computed: bool = DEFAULT_SAVE_COMPUTED
    manipulation_fn: ManipulationFn = field(default=_default_manipulation, repr=False)
    markov_manipulation_fn: MarkovManipulationFn = field(
        default=_default_markov_manipulation,
        repr=False,
    )
    window_start_mins: int = field(init=False)
    window_length_mins: int = field(init=False)

    def __post_init__(self) -> None:
        # Accept strings
        self.schedule_file = Path(self.schedule_file)
        self.analysis_dir = Path(self.analysis_dir)
        self.output_dir = Path(self.output_dir)
        self.reftz = validate_reftz(self.reftz)
        self.window_start = validate_window_start(self.window_start)
        self.window_length_hours = validate_window_length_hours(self.window_length_hours)
        self.actual_times = validate_actual_times(self.actual_times)
        self.output_mode = validate_output_mode(self.output_mode)
        self.log_file = validate_log_file(self.log_file)
        self.window_start_mins = window_start_to_minutes(self.window_start)
        self.window_length_mins = int(self.window_length_hours) * 60

    _ANALYSIS_NAMES: ClassVar[tuple[str, ...]] = (
        "initial_conditions",
        "markov",
        "scheduled_turnaround_intraday_params",
        "scheduled_turnaround_temporal_profile",
        "scheduled_flight_time",
    )

    # helpers

    def analysis_path(self, name: str) -> Path:
        """Return ``analysis_dir / <name><suffix>.csv``."""
        return self.analysis_dir / f"{name}{self.suffix}.csv"

    def output_path(self, name: str) -> Path:
        """Return ``output_dir / <name><suffix>.csv``."""
        return self.output_dir / f"{name}{self.suffix}.csv"

    def cleanup_analysis(self) -> None:
        """Delete all intermediate analysis files from analysis_dir."""
        for name in self._ANALYSIS_NAMES:
            self.analysis_path(name).unlink(missing_ok=True)

    def resolved_log_file(self) -> Path:
        """Return the log path used when ``output_mode='file'``."""
        return resolve_log_file(config=self)
