import math
import random
from typing import List, Tuple

from .structures import Aircraft, Flight
from .data_manager import DataManager, BIN_SIZE_MINS, END_OF_DAY_MINS
from .capacity import CapacityTracker
from .stats import GenerationStats

class ScheduleGenerator:
    """
    Greedy forward construction with exact-key turnaround logic.
    This class is responsible for generating continuous flight schedules for individual aircraft
    by selecting appropriate destinations and calculating turnaround times based on the probabilistic models.
    """

    def __init__(self, data: DataManager, tracker: CapacityTracker, stats: GenerationStats, rng: random.Random):
        self.data = data
        self.tracker = tracker
        self.stats = stats
        self.rng = rng

    def _get_turnaround_candidates(
        self, aircraft: Aircraft, prev_origin: str, current_airport: str, arrival_time: int,
    ) -> List[Tuple[int, str, float]]:
        """
        Get all turnaround candidates to try, including scheduled, interval, and extended.
        Options are selected using Weighted Random Sampling without replacement to ensure
        the synthetic distribution tail matches reality.
        """
        ta_options = self.data.get_turnaround_options(
            aircraft.operator, prev_origin, current_airport, aircraft.wake, arrival_time,
        )

        if ta_options:
            scheduled_times = [t for t, _ in ta_options]
            min_ta = min(scheduled_times)
            max_ta = max(scheduled_times)
        else:
            min_ta = BIN_SIZE_MINS
            max_ta = END_OF_DAY_MINS - int(max(0, arrival_time)) + BIN_SIZE_MINS
            max_ta = max(BIN_SIZE_MINS, min(max_ta, END_OF_DAY_MINS - BIN_SIZE_MINS))
            ta_options = []

        candidates = []
        seen: set[int] = set()

        # 1. Stochastic selection proportional to probability
        weighted_items = []
        for ta_time, prob in ta_options:
            r = self.rng.random()
            if r == 0:
                r = 1e-10
            weight = prob if prob > 0 else 1e-10
            score = pow(r, 1.0 / weight)
            weighted_items.append((ta_time, prob, score))

        weighted_items.sort(key=lambda x: -x[2])

        # 2. Add scheduled times (Fallback 1)
        for ta_time, prob, _ in weighted_items:
            if ta_time not in seen:
                candidates.append((ta_time, "scheduled", prob))
                seen.add(ta_time)

        # 3. Interval search within min/max bounds (Fallback 2)
        for ta_time in range(min_ta, max_ta + 1, 5):
            if ta_time not in seen:
                candidates.append((ta_time, "interval", 0.0))
                seen.add(ta_time)

        # 4. Extended search beyond max scheduled time (Fallback 3)
        for ta_time in range(max_ta + 5, END_OF_DAY_MINS, 5):
            if ta_time not in seen:
                candidates.append((ta_time, "extended", 0.0))
                seen.add(ta_time)

        return candidates

    def seed_initial_flights(self, aircraft: Aircraft) -> bool:
        """
        Register initial/prior flights into the chain and capacity tracker.

        Returns True if greedy extension should follow, False if the aircraft
        is complete (prior-only, single-flight, or has no initial flight).
        """
        aircraft.chain.clear()

        # Handle aircraft with only an incomplete prior flight
        if aircraft.initial_flight is None:
            if aircraft.prior_flight:
                prior = Flight(
                    orig=aircraft.prior_flight.orig,
                    dest=aircraft.prior_flight.dest,
                    std=aircraft.prior_flight.std,
                    sta=aircraft.prior_flight.sta,
                )
                aircraft.chain.append(prior)
                self.tracker.add_flight(prior)
                self.stats.prior_flight_pasted += 1
                self.stats.prior_only_pasted += 1
            return False

        # Add prior flight if it exists
        if aircraft.prior_flight:
            prior = Flight(
                orig=aircraft.prior_flight.orig,
                dest=aircraft.prior_flight.dest,
                std=aircraft.prior_flight.std,
                sta=aircraft.prior_flight.sta,
            )
            aircraft.chain.append(prior)
            self.tracker.add_flight(prior)
            self.stats.prior_flight_pasted += 1

        # Add the first flight of the day
        first_flight = Flight(
            orig=aircraft.initial_flight.orig,
            dest=aircraft.initial_flight.dest,
            std=aircraft.initial_flight.std,
            sta=aircraft.initial_flight.sta,
        )
        aircraft.chain.append(first_flight)
        self.tracker.add_flight(first_flight)

        # If aircraft only performs one flight, do not extend
        if aircraft.is_single_flight:
            self.stats.single_flight_passthrough += 1
            if aircraft.prior_flight:
                self.stats.single_flight_with_prior += 1
            return False

        return True

    def generate_greedy_chain(self, aircraft: Aircraft) -> bool:
        """
        Extend an already-seeded chain using greedy forward construction.
        Continues appending flights until the end of the day or no valid options remain.
        """
        if not aircraft.chain:
            return False

        last_flight = aircraft.chain[-1]
        current_airport = last_flight.dest
        prev_origin = last_flight.orig
        arrival_time = last_flight.sta

        # Loop until the day ends
        while arrival_time < END_OF_DAY_MINS:
            current_anchor_flight = aircraft.chain[-1] if aircraft.chain else None

            # --- 1. Sample Turnaround Time ---
            turnaround_time, ta_category = self.data.sample_turnaround_for_prev_origin(
                op=aircraft.operator,
                prev_origin=prev_origin,
                origin=current_airport,
                wake=aircraft.wake,
                arr_utc_mins=arrival_time,
            )

            # Failure: Could not sample a valid turnaround time
            if turnaround_time < 0:
                self.stats.no_destinations += 1
                if len(aircraft.chain) == 1:
                    self.stats.single_flight_total += 1
                    self.stats.single_flight_no_destinations += 1
                markov_key = (aircraft.operator, aircraft.wake, prev_origin, current_airport)
                hourly_data = self.data.markov_hourly.get(markov_key, {})
                available_hours = sorted(set(hourly_data.keys()))
                self.stats.add_example_no_destinations(
                    f"  Aircraft {aircraft.reg} ({aircraft.operator}, {aircraft.wake}):\\n"
                    f"    At {current_airport} from {prev_origin}, arrival={arrival_time} mins\\n"
                    f"    No turnaround params -- key: {(aircraft.operator, prev_origin, current_airport, aircraft.wake)}\\n"
                    f"    Markov available hours: {available_hours}"
                )
                break

            # Calculate next Scheduled Time of Departure (STD), rounded up to nearest 5 minutes
            scheduled_departure_time = int(math.ceil((arrival_time + turnaround_time) / 5.0) * 5)

            # Stop if turnaround extends to the next day or runs past END_OF_DAY_MINS
            if ta_category == "next_day" or scheduled_departure_time >= END_OF_DAY_MINS:
                if current_anchor_flight is not None:
                    current_anchor_flight.turnaround_to_next_category = "next_day"
                    current_anchor_flight.turnaround_to_next_minutes = int(turnaround_time)
                self.stats.end_of_day += 1
                if len(aircraft.chain) == 1:
                    self.stats.single_flight_end_of_day += 1
                    self.stats.single_flight_total += 1
                    term_hour = (scheduled_departure_time // 60) % 24
                    self.stats.single_flight_termination_hours[term_hour] += 1
                break

            # --- 2. Select Next Destination ---
            destinations, source_type = self.data.get_destinations(
                aircraft.operator, aircraft.wake, prev_origin, current_airport,
                dep_utc_mins=scheduled_departure_time, arr_utc_mins=arrival_time,
            )

            flight_added = False
            termination_reason = None

            if not destinations:
                termination_reason = "no_dest"
            else:
                # Shuffle destinations based on their selection weights
                reordered = []
                remaining = list(destinations)
                while remaining:
                    weights = [prob for _, prob in remaining]
                    idx = self.rng.choices(range(len(remaining)), weights=weights, k=1)[0]
                    reordered.append(remaining.pop(idx))

                # Try to schedule a flight to a destination until one works
                for dest, _ in reordered:
                    flight_time = self.data.get_flight_time(
                        current_airport, dest, aircraft.operator, aircraft.wake, dep_utc_mins=scheduled_departure_time,
                    )
                    if flight_time <= 0:
                        continue

                    scheduled_arrival_time = scheduled_departure_time + flight_time

                    # Check airport capacity constraints if arriving today
                    if scheduled_arrival_time < END_OF_DAY_MINS:
                        if not self.tracker.check_availability(current_airport, dest, scheduled_departure_time, scheduled_arrival_time):
                            termination_reason = "capacity"
                            continue

                    # --- 3. Append Flight to Chain ---
                    if current_anchor_flight is not None:
                        current_anchor_flight.turnaround_to_next_category = "intraday"
                        current_anchor_flight.turnaround_to_next_minutes = int(turnaround_time)

                    flight = Flight(orig=current_airport, dest=dest, std=scheduled_departure_time, sta=scheduled_arrival_time)
                    aircraft.chain.append(flight)
                    
                    if scheduled_arrival_time < END_OF_DAY_MINS:
                        self.tracker.add_flight(flight)

                    # Update statistics based on how the destination was found
                    if source_type == "primary_exact":
                        self.stats.dest_found_primary_exact += 1
                    elif source_type == "primary_expanded":
                        self.stats.dest_found_primary_expanded += 1
                    elif source_type == "fallback_expanded":
                        self.stats.dest_found_fallback_expanded += 1
                    elif source_type == "return_to_origin":
                        self.stats.dest_found_return_to_origin += 1

                    # Update state for the next iteration
                    prev_origin = current_airport
                    current_airport = dest
                    arrival_time = scheduled_arrival_time

                    if scheduled_arrival_time >= END_OF_DAY_MINS:
                        self.stats.end_of_day += 1

                    flight_added = True
                    break

            # --- 4. Handle Failure to Add Next Flight ---
            if not flight_added:
                if termination_reason == "capacity":
                    self.stats.capacity_exhausted += 1
                    if len(aircraft.chain) == 1:
                        self.stats.single_flight_total += 1
                        self.stats.single_flight_capacity_exhausted += 1
                else:
                    self.stats.no_destinations += 1
                    if len(aircraft.chain) == 1:
                        self.stats.single_flight_total += 1
                        self.stats.single_flight_no_destinations += 1

                    markov_key = (aircraft.operator, aircraft.wake, prev_origin, current_airport)
                    hourly_data = self.data.markov_hourly.get(markov_key, {})
                    available_hours = sorted(set(hourly_data.keys()))
                    self.stats.add_example_no_destinations(
                        f"  Aircraft {aircraft.reg} ({aircraft.operator}, {aircraft.wake}):\\n"
                        f"    At {current_airport} from {prev_origin}, arrival={arrival_time} mins\\n"
                        f"    Markov key: {markov_key}\\n"
                        f"    Available departure hours: {available_hours}"
                    )
                break

        return len(aircraft.chain) > 0
