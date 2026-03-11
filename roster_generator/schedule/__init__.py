"""Schedule Generator sub-package.

This module exposes the public API for generating the synthetic flight schedules.
"""

from ._schedule_generator import ScheduleGenerator
from ._schedule_capacity import CapacityTracker
from ._schedule_data_manager import BIN_SIZE_MINS, END_OF_DAY_MINS, DataManager
from ._schedule_structures import Aircraft, Flight
from .schedule import generate_schedule
from ._schedule_stats import GenerationStats

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
