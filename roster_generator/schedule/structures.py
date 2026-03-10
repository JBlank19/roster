from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Flight:
    orig: str
    dest: str
    std: int   # Scheduled Time of Departure (minutes from midnight UTC)
    sta: int   # Scheduled Time of Arrival (minutes from midnight UTC)
    turnaround_to_next_category: str = ""
    turnaround_to_next_minutes: int = -1

@dataclass
class Aircraft:
    reg: str
    operator: str
    wake: str
    initial_flight: Optional[Flight] = None
    prior_flight: Optional[Flight] = None
    chain: List[Flight] = field(default_factory=list)
    is_single_flight: bool = False
    is_prior_only: bool = False
