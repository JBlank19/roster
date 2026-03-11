from dataclasses import dataclass, field
from typing import List


@dataclass
class Flight:
    """One scheduled leg in minutes-from-midnight UTC."""

    orig: str
    dest: str
    std: int  # scheduled departure UTC minutes
    sta: int  # scheduled arrival UTC minutes
    turnaround_to_next_category: str = ""
    turnaround_to_next_minutes: int = -1


@dataclass
class Aircraft:
    """Aircraft schedule state used during greedy chain generation."""

    reg: str
    operator: str
    wake: str
    initial_flight: Flight | None = None
    prior_flight: Flight | None = None
    chain: List[Flight] = field(default_factory=list)
    is_single_flight: bool = False
    is_prior_only: bool = False
