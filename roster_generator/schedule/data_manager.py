import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import airportsdata
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

# Shared constants
BIN_SIZE_MINS = 5
END_OF_DAY_MINS = 1440
P_NEXT_BIN_SIZE_MINS = 60
MAX_INTRADAY_RESAMPLE_ATTEMPTS = 256

class DataManager:
    """
    Holds and manages all lookup tables required for the roster generation schedule.
    
    This includes loading and providing access to:
    - 2nd-order Markov chain transitions for destination selection.
    - Turnaround time distributions (intraday and next-day).
    - Route schedules (median flight times).
    - Airport capacities.
    """

    def __init__(
        self,
        rng: random.Random,
        routes_path: Path,
        airports_path: Path,
        markov_path: Path,
        turnaround_intraday_params_path: Path,
        turnaround_temporal_profile_path: Path,
    ):
        self.rng = rng
        self.routes_path = routes_path
        self.airports_path = airports_path
        self.markov_path = markov_path
        self.turnaround_intraday_params_path = turnaround_intraday_params_path
        self.turnaround_temporal_profile_path = turnaround_temporal_profile_path
        
        # Statistics tracking for debugging and validation
        self.turnaround_lookup_stats: Dict[str, int] = defaultdict(int)
        
        self._load_all()

    def _load_all(self):
        """Orchestrates the loading of all necessary datasets."""
        print("[Schedule] Loading data...")
        print(f"[Schedule]   Routes: {self.routes_path}")
        print(f"[Schedule]   Airports: {self.airports_path}")

        self._load_markov_data()
        self._load_turnaround_data()
        self._load_route_data()
        self._load_airport_data()

        print(f"[Schedule]   Markov hourly states: {len(self.markov_hourly)}")
        print(f"[Schedule]   Turnaround intraday keys: {len(self.turnaround_intraday_params)}")
        print(f"[Schedule]   Turnaround temporal profile keys: {len(self.turnaround_temporal_profiles)}")
        print(f"[Schedule]   Turnaround p_next (airline,wake) keys: {len(self.turnaround_temporal_next_prob)}")
        print(f"[Schedule]   Routes: {len(self.routes)}")

    # ---------------------------------------------------------
    # Data Loading Methods
    # ---------------------------------------------------------

    def _load_markov_data(self):
        """
        Loads 2nd-order Markov transition probabilities from CSV.
        
        Populates:
        - self.markov_hourly: Dict keyed by (operator, wake, prev_origin, origin)
        - self.markov_fallback_hourly: Dict keyed by (operator, wake, origin)
        """
        markov_df = pd.read_csv(self.markov_path)
        self.markov_hourly: Dict[tuple, Dict[int, Dict[str, int]]] = {}
        self.markov_fallback_hourly: Dict[tuple, Dict[int, Dict[str, int]]] = {}

        # Remove END states as we are continuously generating a schedule
        if "ARR_ICAO" in markov_df.columns:
            arr_codes = markov_df["ARR_ICAO"].astype(str).str.upper().str.strip()
            end_mask = arr_codes == "END"
            end_rows = int(end_mask.sum())
            if end_rows:
                markov_df = markov_df[~end_mask].copy()
                print(f"[Schedule]   Ignored {end_rows} Markov END transitions")

        # Use fast numpy arrays for iteration
        operators = markov_df["AC_OPERATOR"].to_numpy(dtype=object)
        wakes = markov_df["AC_WAKE"].to_numpy(dtype=object)
        prev_origins = markov_df["PREV_ICAO"].to_numpy(dtype=object)
        origins = markov_df["DEP_ICAO"].to_numpy(dtype=object)
        destinations = markov_df["ARR_ICAO"].to_numpy(dtype=object)
        counts = markov_df["COUNT"].astype(int).to_numpy()
        dep_hours = pd.to_numeric(markov_df["DEP_HOUR_UTC"], errors="coerce").fillna(12).astype(int).to_numpy()

        for i in range(len(operators)):
            primary_key = (operators[i], wakes[i], prev_origins[i], origins[i])
            fallback_key = (operators[i], wakes[i], origins[i])
            dep_hour = int(dep_hours[i])
            destination = destinations[i]
            count = int(counts[i])

            # Populate primary (2nd-order) Markov dictionary
            if primary_key not in self.markov_hourly:
                self.markov_hourly[primary_key] = {}
            if dep_hour not in self.markov_hourly[primary_key]:
                self.markov_hourly[primary_key][dep_hour] = {}
            self.markov_hourly[primary_key][dep_hour][destination] = count

            # Populate fallback (1st-order) Markov dictionary
            if fallback_key not in self.markov_fallback_hourly:
                self.markov_fallback_hourly[fallback_key] = {}
            if dep_hour not in self.markov_fallback_hourly[fallback_key]:
                self.markov_fallback_hourly[fallback_key][dep_hour] = {}
            
            existing_count = self.markov_fallback_hourly[fallback_key][dep_hour].get(destination, 0)
            self.markov_fallback_hourly[fallback_key][dep_hour][destination] = existing_count + count

    def _load_turnaround_data(self):
        """
        Loads turnaround time distributions and temporal profiles.
        
        Populates:
        - self.turnaround_intraday_params: lognormal parameters
        - self.turnaround_temporal_profiles: sparse transition profiles
        - self.turnaround_temporal_totals: aggregated sums
        - self.turnaround_temporal_next_prob: empirical hourly probability of waiting until next day
        """
        self.turnaround_intraday_params: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.turnaround_temporal_profiles: Dict[tuple, Tuple[Dict[int, float], Dict[int, float]]] = {}
        self.turnaround_temporal_totals: Dict[tuple, Tuple[float, float]] = {}
        self.turnaround_temporal_next_prob: Dict[Tuple[str, str], np.ndarray] = {}

        intraday_df = pd.read_csv(self.turnaround_intraday_params_path)
        temporal_df = pd.read_csv(self.turnaround_temporal_profile_path)

        # Validate Intraday CSV columns
        expected_intraday_cols = {"airline", "wake", "location", "shape"}
        if not expected_intraday_cols.issubset(intraday_df.columns):
            raise ValueError(
                f"Missing columns in intraday params: expected {sorted(expected_intraday_cols)} "
                f"got {list(intraday_df.columns)}"
            )

        # Validate Temporal CSV columns
        expected_temporal_cols = {
            "airline", "previous_origin", "origin", "wake",
            "intraday_sparse", "next_day_sparse",
            "total_intraday", "total_next_day",
        }
        if not expected_temporal_cols.issubset(temporal_df.columns):
            raise ValueError(
                f"Missing columns in temporal profile: expected {sorted(expected_temporal_cols)} "
                f"got {list(temporal_df.columns)}"
            )

        # 1. Parse Lognormal Intraday Parameters
        for row in intraday_df.itertuples(index=False):
            key = (str(row.airline).strip(), str(row.wake).strip())
            if key in self.turnaround_intraday_params:
                raise ValueError(f"Duplicate intraday turnaround key: {key}")
            
            loc = float(row.location)
            shape = float(row.shape)
            if not np.isfinite(loc) or not np.isfinite(shape) or shape <= 0:
                raise ValueError(f"Invalid intraday params for key {key}: loc={loc}, shape={shape}")
            
            self.turnaround_intraday_params[key] = (loc, shape)

        # Intermediate dictionaries for aggregating next-day probabilities by (airline, wake)
        airline_wake_intraday: Dict[Tuple[str, str], Dict[int, float]] = {}
        airline_wake_next_day: Dict[Tuple[str, str], Dict[int, float]] = {}

        # 2. Parse Temporal Profiles
        for row in temporal_df.itertuples(index=False):
            key = (
                str(row.airline).strip(),
                str(row.previous_origin).strip(),
                str(row.origin).strip(),
                str(row.wake).strip(),
            )
            if key in self.turnaround_temporal_profiles:
                raise ValueError(f"Duplicate temporal key: {key}")

            intra_sparse = self._decode_sparse_counts(row.intraday_sparse)
            next_sparse = self._decode_sparse_counts(row.next_day_sparse)

            self.turnaround_temporal_profiles[key] = (intra_sparse, next_sparse)
            self.turnaround_temporal_totals[key] = (float(row.total_intraday), float(row.total_next_day))

            # Aggregate by (airline, wake) for fallback next-day probabilities
            aw_key = (str(row.airline).strip(), str(row.wake).strip())
            if aw_key not in airline_wake_intraday:
                airline_wake_intraday[aw_key] = {}
                airline_wake_next_day[aw_key] = {}
                
            for bin_idx, count in intra_sparse.items():
                airline_wake_intraday[aw_key][bin_idx] = airline_wake_intraday[aw_key].get(bin_idx, 0.0) + count
            for bin_idx, count in next_sparse.items():
                airline_wake_next_day[aw_key][bin_idx] = airline_wake_next_day[aw_key].get(bin_idx, 0.0) + count

        # 3. Compute Hourly Next-Day Probabilities
        all_aw_keys = set(airline_wake_intraday.keys()) | set(airline_wake_next_day.keys())
        for aw_key in all_aw_keys:
            intra_counts = airline_wake_intraday.get(aw_key, {})
            next_counts = airline_wake_next_day.get(aw_key, {})
            prob_vector = self._build_temporal_next_probability_vector(intra_counts, next_counts)
            self.turnaround_temporal_next_prob[aw_key] = prob_vector

    def _load_route_data(self):
        """Loads valid connections and scheduled duration."""
        routes_df = pd.read_csv(self.routes_path)
        self.routes: Dict[tuple, int] = {}
        for r in routes_df.itertuples():
            self.routes[(r.orig_id, r.dest_id, r.airline_id, r.wake_type)] = int(r.scheduled_time)

    def _load_airport_data(self):
        """Loads airport capacities and determines timezones."""
        airports_df = pd.read_csv(self.airports_path)
        self.rolling_capacity: Dict[str, float] = {}
        self.burst_capacity: Dict[str, float] = {}

        for r in airports_df.itertuples():
            self.rolling_capacity[r.airport_id] = float(r.rolling_capacity)
            self.burst_capacity[r.airport_id] = float(r.burst_capacity)

        self._build_tz_offsets()

    # ---------------------------------------------------------
    # Helper Utilities
    # ---------------------------------------------------------

    def _decode_sparse_counts(self, payload: str) -> Dict[int, float]:
        """Decodes string representation of sparse temporal profiles (e.g. 'minute:count;minute:count')."""
        out: Dict[int, float] = {}
        if payload is None:
            return out
        if isinstance(payload, float) and np.isnan(payload):
            return out
            
        text = str(payload).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return out
            
        for token in text.split(";"):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid sparse temporal token '{token}'")
                
            minute_s, count_s = token.split(":", 1)
            minute = int(minute_s)
            
            if minute < 0 or minute >= END_OF_DAY_MINS or (minute % BIN_SIZE_MINS) != 0:
                raise ValueError(f"Invalid sparse temporal minute '{minute}'")
                
            count = float(count_s)
            if count > 0:
                out[minute // BIN_SIZE_MINS] = count
        return out

    def _build_temporal_next_probability_vector(
        self,
        intraday_counts: Dict[int, float],
        next_day_counts: Dict[int, float],
    ) -> np.ndarray:
        """Build empirical hourly array representing p(next_day) based on real aggregated counts."""
        n_bins = END_OF_DAY_MINS // P_NEXT_BIN_SIZE_MINS
        intra_hourly = np.zeros(n_bins, dtype=float)
        next_hourly = np.zeros(n_bins, dtype=float)

        # Aggregate 5-min bins into hourly bins
        for bin_idx, count in intraday_counts.items():
            minute = int(bin_idx) * BIN_SIZE_MINS
            hour_idx = minute // P_NEXT_BIN_SIZE_MINS
            if 0 <= hour_idx < n_bins and count > 0:
                intra_hourly[hour_idx] += float(count)

        for bin_idx, count in next_day_counts.items():
            minute = int(bin_idx) * BIN_SIZE_MINS
            hour_idx = minute // P_NEXT_BIN_SIZE_MINS
            if 0 <= hour_idx < n_bins and count > 0:
                next_hourly[hour_idx] += float(count)

        totals = intra_hourly + next_hourly
        n_total = float(np.sum(totals))
        if n_total <= 0:
            return np.zeros(n_bins, dtype=float)

        probs = np.zeros(n_bins, dtype=float)
        nonzero = totals > 0
        probs[nonzero] = next_hourly[nonzero] / totals[nonzero]
        return probs

    def _build_tz_offsets(self):
        """Creates a lookup for caching Time Zone offsets from UTC for capacity."""
        airports_db = airportsdata.load("ICAO")
        # Fixed reference date to avoid daylight saving fluctuation edge cases
        ref_date = datetime(2023, 9, 15)
        self.tz_offset: Dict[str, int] = {}
        
        for icao in set(self.rolling_capacity.keys()):
            try:
                tz_str = airports_db.get(icao, {}).get("tz", "UTC")
                tz = pytz.timezone(tz_str)
                self.tz_offset[icao] = int(tz.utcoffset(ref_date).total_seconds() // 3600)
            except Exception:
                self.tz_offset[icao] = 0

    def get_utc_hour(self, utc_mins: int) -> int:
        """Converts minute of day to hour of day (0-23)."""
        return (utc_mins // 60) % 24

    # ---------------------------------------------------------
    # Markov Lookup Lookups
    # ---------------------------------------------------------

    def _find_hourly_data_with_radius(
        self, data_source: dict, key: tuple, center_hour: int, max_radius: int = 2,
    ) -> Tuple[Dict[str, int], int]:
        """
        Looks up transitions for a specific hour, using expanding radius search if missing.
        As a last resort, checks for the next available chronological hour.
        
        Returns:
            Tuple containing the destination distribution dictionary and the hour it was found at.
        """
        hourly_data = data_source.get(key, {})
        if not hourly_data:
            return {}, -1

        candidates = [center_hour]
        for radius in range(1, max_radius + 1):
            # Randomize search direction
            if self.rng.random() < 0.5:
                candidates.append((center_hour - radius) % 24)
                candidates.append((center_hour + radius) % 24)
            else:
                candidates.append((center_hour + radius) % 24)
                candidates.append((center_hour - radius) % 24)

        for h in candidates:
            if h in hourly_data:
                return hourly_data[h], h

        # Fallback to nearest future hour
        available_hours = sorted(hourly_data.keys())
        future_hours = [h for h in available_hours if h >= center_hour]

        if future_hours:
            next_hour = future_hours[0]
            return hourly_data[next_hour], next_hour

        return {}, -1

    def get_destinations(
        self, op: str, wake: str, prev_origin: str, origin: str,
        dep_utc_mins: int, arr_utc_mins: int,
    ) -> Tuple[List[Tuple[str, float]], str]:
        """
        Determines possible destinations and their probabilities from a given origin.
        
        Attempts matching in this order:
        1. Exact primary match: (op, wake, prev_origin, origin) at exact hour
        2. Expanded primary match: Primary key but at a nearby hour
        3. Expanded fallback match: (op, wake, origin) at a nearby hour
        4. Return to origin: If previous origin is available, assume direct return
        
        Returns:
            Tuple containing a list of (destination, probability) and a string describing how it was found.
        """
        primary_key = (op, wake, prev_origin, origin)
        fallback_key = (op, wake, origin)
        dep_hour = self.get_utc_hour(dep_utc_mins)

        dest_counts, found_hour = self._find_hourly_data_with_radius(self.markov_hourly, primary_key, dep_hour)
        source = "none"

        if dest_counts:
            source = "primary_exact" if found_hour == dep_hour else "primary_expanded"
        else:
            dest_counts, found_hour = self._find_hourly_data_with_radius(
                self.markov_fallback_hourly, fallback_key, dep_hour,
            )
            if dest_counts:
                source = "fallback_expanded"

        if source == "none":
            if prev_origin and prev_origin != origin:
                return [(prev_origin, 1.0)], "return_to_origin"
            return [], "none"

        total_count = sum(dest_counts.values())
        results = []
        if total_count > 0:
            for dest, count in dest_counts.items():
                results.append((dest, count / total_count))

        results.sort(key=lambda x: -x[1])
        return results, source

    # ---------------------------------------------------------
    # Turnaround Logic Lookups
    # ---------------------------------------------------------

    def _resolve_turnaround_key(
        self,
        params_map: Dict[Tuple[str, str], Tuple[float, float]],
        op: str,
        wake: str,
        count_stats: bool = True,
    ) -> Tuple[Optional[Tuple[str, str]], str]:
        """Resolves lognormal dictionary lookup, reporting stats."""
        key = (str(op).strip(), str(wake).strip())
        if key in params_map:
            if count_stats:
                self.turnaround_lookup_stats["param_exact"] += 1
            return key, "exact"
        if count_stats:
            self.turnaround_lookup_stats["param_missing"] += 1
        return None, "none"

    def _resolve_temporal_next_probability(
        self, op: str, prev_origin: str, origin: str, wake: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Finds hourly probability array for skipping to the next day."""
        key = (str(op).strip(), str(wake).strip())
        if key in self.turnaround_temporal_next_prob:
            self.turnaround_lookup_stats["temporal_exact"] += 1
            return self.turnaround_temporal_next_prob[key], "exact"

        self.turnaround_lookup_stats["temporal_missing_exact"] += 1
        return None, "none"

    def _sample_lognormal_minutes(self, location: float, shape: float) -> int:
        """Samples from lognormal distribution and clamps to 5-min bins."""
        draw = self.rng.lognormvariate(float(location), float(max(shape, 1e-6)))
        ta = int(round(max(0.0, draw) / BIN_SIZE_MINS) * BIN_SIZE_MINS)
        return max(BIN_SIZE_MINS, ta)

    def _build_next_day_turnaround(self, arr_utc_mins: int) -> int:
        """Calculates exact turnaround time to delay departure until the next day."""
        min_needed = max(BIN_SIZE_MINS, END_OF_DAY_MINS - int(arr_utc_mins) + BIN_SIZE_MINS)
        return int(math.ceil(min_needed / BIN_SIZE_MINS) * BIN_SIZE_MINS)

    def get_turnaround_category(
        self, op: str, prev_origin: str, origin: str, wake: str, arr_utc_mins: int,
    ) -> str:
        """Decides conditionally based on time if an arrival should wait until next day to depart."""
        p_next_by_bin, _ = self._resolve_temporal_next_probability(op, prev_origin, origin, wake)
        if p_next_by_bin is None:
            return "missing"
            
        hour_idx = (int(arr_utc_mins) % END_OF_DAY_MINS) // P_NEXT_BIN_SIZE_MINS
        p_next = float(p_next_by_bin[hour_idx])
        return "next_day" if self.rng.random() < p_next else "intraday"

    def sample_turnaround_for_prev_origin(
        self, op: str, prev_origin: str, origin: str, wake: str, arr_utc_mins: int,
    ) -> Tuple[int, str]:
        """
        Determines the total turnaround time on the ground at the origin.
        
        Workflow:
        1. Decide if 'intraday' or 'next_day' using temporal profile probabilities.
        2. If 'next_day', force departure to delay until after midnight.
        3. If 'intraday', sample from empirical lognormal distribution. Resample boundedly 
           to ensure departure remains in the same day. 
        """
        category = self.get_turnaround_category(op, prev_origin, origin, wake, arr_utc_mins)
        if category == "missing":
            return -1, "missing"

        if category == "next_day":
            return self._build_next_day_turnaround(arr_utc_mins), "next_day"

        key, _ = self._resolve_turnaround_key(self.turnaround_intraday_params, op, wake)
        if key is None:
            return -1, "missing"

        location, shape = self.turnaround_intraday_params[key]

        max_intraday = END_OF_DAY_MINS - int(arr_utc_mins) - BIN_SIZE_MINS
        max_intraday = int(math.floor(max_intraday / BIN_SIZE_MINS) * BIN_SIZE_MINS)
        
        # Guard if the arrival is at the very end of the day bounds
        if max_intraday < BIN_SIZE_MINS:
            return self._build_next_day_turnaround(arr_utc_mins), "next_day"

        # Attempt independent resampling
        for _ in range(MAX_INTRADAY_RESAMPLE_ATTEMPTS):
            sampled = self._sample_lognormal_minutes(location, shape)
            if sampled <= max_intraday:
                return sampled, "intraday"

        # Capped fallback to keep within same day bounds
        self.turnaround_lookup_stats["intraday_resample_guard"] += 1
        return max_intraday, "intraday"

    def get_turnaround_options(
        self, op: str, prev_origin: str, origin: str, wake: str, arr_utc_mins: int,
    ) -> List[Tuple[int, float]]:
        """Placeholder matching API parity (empty options array since no fallback distribution is available)."""
        del op, prev_origin, origin, wake, arr_utc_mins
        return []

    # ---------------------------------------------------------
    # Route Validation Lookups
    # ---------------------------------------------------------

    def get_flight_time(self, origin: str, dest: str, op: str, wake: str, dep_utc_mins: int = None) -> int:
        """Looks up the precalculated median flight duration for the given route."""
        del dep_utc_mins
        return self._get_flight_time_median(origin, dest, op, wake)

    def _get_flight_time_median(self, origin: str, dest: str, op: str, wake: str) -> int:
        """Internal lookup with fallback to an 'ALL' airline code median."""
        t = self.routes.get((origin, dest, op, wake), 0)
        if t == 0:
            t = self.routes.get((origin, dest, "ALL", wake), 0)
            
        if t > 0:
            t = int(t // 5 * 5)
        return t
