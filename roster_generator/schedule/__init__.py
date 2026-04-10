"""Schedule Generator sub-package.

This module exposes the public API for generating the synthetic flight schedules.
"""

from ._rejection_log import CapacityRejection, EndOfDayRejection, NoDestinationRejection, RejectionLog
from ._schedule_capacity import CapacityCheckResult, CapacityTracker
from ._schedule_data_manager import BIN_SIZE_MINS, END_OF_DAY_MINS, DataManager
from ._schedule_generator import ScheduleGenerator
from ._schedule_stats import GenerationStats
from ._schedule_structures import Aircraft, Flight
from .schedule import generate_schedule

__all__ = (
    "Aircraft",
    "BIN_SIZE_MINS",
    "CapacityCheckResult",
    "CapacityRejection",
    "CapacityTracker",
    "DataManager",
    "END_OF_DAY_MINS",
    "EndOfDayRejection",
    "Flight",
    "GenerationStats",
    "NoDestinationRejection",
    "RejectionLog",
    "ScheduleGenerator",
    "generate_schedule",
)
