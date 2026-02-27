"""Central configuration for the pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

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
    """

    schedule_file: Path
    analysis_dir: Path
    output_dir: Path
    seed: int = 42
    suffix: str = ""

    def __post_init__(self) -> None:
        # Accept strings
        self.schedule_file = Path(self.schedule_file)
        self.analysis_dir = Path(self.analysis_dir)
        self.output_dir = Path(self.output_dir)

    # helpers

    def analysis_path(self, name: str) -> Path:
        """Return ``analysis_dir / <name><suffix>.csv``."""
        return self.analysis_dir / f"{name}{self.suffix}.csv"

    def output_path(self, name: str) -> Path:
        """Return ``output_dir / <name><suffix>.csv``."""
        return self.output_dir / f"{name}{self.suffix}.csv"
