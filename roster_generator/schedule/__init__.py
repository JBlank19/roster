"""Schedule Generator sub-package.

This module exposes the public API for generating the synthetic flight schedules.
"""

from .generator import ScheduleGenerator
from .capacity import CapacityTracker
from .data_manager import BIN_SIZE_MINS, END_OF_DAY_MINS, DataManager
from .structures import Aircraft, Flight
from .schedule_main import generate_schedule
from .stats import GenerationStats

__all__ = (
    "Aircraft",
    "BIN_SIZE_MINS",
    "CapacityTracker",
    "DataManager",
    "END_OF_DAY_MINS",
    "Flight",
    "GenerationStats",
    "ScheduleGenerator",
    "generate_schedule",
)
