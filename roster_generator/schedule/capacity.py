from collections import defaultdict
from typing import Dict, Tuple

from .structures import Flight

# Configuration constants
BIN_SIZE_MINS = 5
END_OF_DAY_MINS = 1440
ROLLING_WINDOW_SIZE_BINS = 12  # 60 minutes / 5 minutes per bin
DEFAULT_CAPACITY = 999

class CapacityTracker:
    """
    Tracks and validates airport capacity constraints for flights.
    
    Supports two types of constraints:
    1. Burst Capacity: Maximum movements allowed in a single time bin (e.g., 5 mins).
    2. Rolling Capacity: Maximum movements allowed in a rolling 60-minute window.
    """

    def __init__(self, rolling_capacities: Dict[str, float], burst_capacities: Dict[str, float]):
        """
        Initialize the tracker with airport-specific capacity limits.
        
        Args:
            rolling_capacities: Map of airport code to hourly movement limit.
            burst_capacities: Map of airport code to movement limit per bin.
        """
        self.rolling_cap = rolling_capacities
        self.burst_cap = burst_capacities
        self.num_bins = END_OF_DAY_MINS // BIN_SIZE_MINS
        
        # Internal state tracking movements
        self.dep_slots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.arr_slots: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.movements_rolling: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def _get_bin_index(self, time_mins: int) -> int:
        """Map a time in minutes (0-1439) to its corresponding bin index."""
        return max(0, min(time_mins // BIN_SIZE_MINS, self.num_bins - 1))

    def _check_airport_availability(self, airport_code: str, time_mins: int, is_departure: bool) -> bool:
        """
        Helper to check if an airport can accommodate a movement at a specific time.
        
        Checks both burst (per-bin) and rolling (hourly) capacity.
        """
        # 1. Check Burst (Slot) Capacity
        time_bin = self._get_bin_index(time_mins)
        slots_tracker = self.dep_slots if is_departure else self.arr_slots
        
        burst_limit = self.burst_cap.get(airport_code, DEFAULT_CAPACITY)
        if slots_tracker[airport_code][time_bin] >= burst_limit:
            return False

        # 2. Check Rolling Window Capacity
        rolling_limit = self.rolling_cap.get(airport_code, DEFAULT_CAPACITY)
        
        # Check all rolling windows that include this time bin
        # A movement at 'time_bin' affects all 60-min windows starting at or after 'time_bin'
        # but within 60 mins of it.
        # However, the original logic checks if the CURRENT movement would violate ANY 
        # future window that includes it.
        start_bin = time_bin
        end_bin = min(self.num_bins - 1, time_bin + ROLLING_WINDOW_SIZE_BINS - 1)

        for b in range(start_bin, end_bin + 1):
            if self.movements_rolling[airport_code][b] >= rolling_limit:
                return False

        return True

    def check_availability(self, 
                           origin: str, 
                           destination: str, 
                           std_mins: int, 
                           sta_mins: int) -> bool:
        """
        Verifies if both origin and destination can accept the flight at the given times.
        
        Args:
            origin: Origin airport code.
            destination: Destination airport code.
            std_mins: Scheduled Time of Departure in minutes from start of day.
            sta_mins: Scheduled Time of Arrival in minutes from start of day.
            
        Returns:
            True if the flight can be scheduled without violating capacity, False otherwise.
        """
        # Ignore flights outside the tracking day (e.g., next day arrivals)
        if not (0 <= std_mins < END_OF_DAY_MINS and 0 <= sta_mins < END_OF_DAY_MINS):
            return True

        # Check origin departure capacity
        if not self._check_airport_availability(origin, std_mins, is_departure=True):
            return False

        # Check destination arrival capacity
        if not self._check_airport_availability(destination, sta_mins, is_departure=False):
            return False

        return True

    def _update_movement_counts(self, airport_code: str, time_mins: int, is_departure: bool):
        """Internal helper to increment movement counts for an airport at a given time."""
        time_bin = self._get_bin_index(time_mins)
        
        # Update burst slots
        slots_tracker = self.dep_slots if is_departure else self.arr_slots
        slots_tracker[airport_code][time_bin] += 1

        # Update all rolling windows affected by this movement
        start_bin = time_bin
        end_bin = min(self.num_bins - 1, time_bin + ROLLING_WINDOW_SIZE_BINS - 1)
        for b in range(start_bin, end_bin + 1):
            self.movements_rolling[airport_code][b] += 1

    def add_flight(self, flight: Flight):
        """
        Registers a flight in the tracker, updating usage counts for origin and destination.
        
        Note: This does not check capacity; it assumes check_availability was called prior.
        """
        if 0 <= flight.std < END_OF_DAY_MINS:
            self._update_movement_counts(flight.orig, flight.std, is_departure=True)

        if 0 <= flight.sta < END_OF_DAY_MINS:
            self._update_movement_counts(flight.dest, flight.sta, is_departure=False)

    def compute_violations(self) -> Tuple[int, int]:
        """
        Calculates total movements exceeding capacity across all airports and bins.
        
        Returns:
            Tuple of (total_burst_violations, total_rolling_violations).
        """
        total_burst_violations = 0
        total_rolling_violations = 0

        # Burst violations (Departures and Arrivals summed separately)
        for airport, bins in self.dep_slots.items():
            limit = self.burst_cap.get(airport, DEFAULT_CAPACITY)
            for count in bins.values():
                total_burst_violations += max(0, count - int(limit))
                
        for airport, bins in self.arr_slots.items():
            limit = self.burst_cap.get(airport, DEFAULT_CAPACITY)
            for count in bins.values():
                total_burst_violations += max(0, count - int(limit))

        # Rolling window violations
        for airport, windows in self.movements_rolling.items():
            limit = self.rolling_cap.get(airport, DEFAULT_CAPACITY)
            for count in windows.values():
                total_rolling_violations += max(0, count - int(limit))

        return total_burst_violations, total_rolling_violations
