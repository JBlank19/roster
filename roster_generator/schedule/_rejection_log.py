"""Detailed per-event rejection log for schedule generation diagnostics.

Usage from a tutorial script::

    stats = roster_generator.generate_schedule(config)
    log = stats.rejection_log

    # Human-readable reports
    print(log.summary())
    print(log.airport_summary("EGLL"))

    # Programmatic access
    cap_by_airport = log.capacity_by_airport()   # {"EGLL": 91, "LFPG": 43, ...}
    egll_cap = log.total_capacity_at("EGLL")

    # Full DataFrame for custom analysis
    df = log.to_dataframe()
    df[df["airport"] == "EGLL"].groupby(["rejection_type", "blocked_constraint"]).size()

Notes
-----
Capacity rejections are recorded for **every** individual capacity-check failure,
including cases where the chain builder subsequently found an alternative
destination in the same iteration.  This maximises diagnostic coverage; the
``summary()`` output labels these clearly so counts are not confused with the
``capacity_exhausted`` counter in :class:`GenerationStats` (which only
increments when *all* candidate destinations were blocked).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Rejection event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityRejection:
    """A single capacity-check failure during destination selection.

    Attributes
    ----------
    aircraft_reg:
        Aircraft registration (tail number).
    operator:
        ICAO airline code of the operating carrier.
    wake:
        ICAO wake-turbulence category (L / M / H).
    origin:
        Departure airport ICAO code for this attempted leg.
    destination:
        Candidate destination ICAO code for this attempted leg.
    blocked_airport:
        The specific airport whose capacity limit was exceeded
        (may be *origin* for a departure constraint or *destination*
        for an arrival constraint).
    blocked_movement:
        ``"departure"`` if the origin departure slot was full;
        ``"arrival"`` if the destination arrival slot was full.
    blocked_constraint:
        ``"burst"`` if the per-5-minute-bin limit was exceeded;
        ``"rolling"`` if the 60-minute rolling-window limit was exceeded.
    std_mins:
        Scheduled time of departure in minutes from window start.
    sta_mins:
        Scheduled time of arrival in minutes from window start.
    """

    aircraft_reg: str
    operator: str
    wake: str
    origin: str
    destination: str
    blocked_airport: str
    blocked_movement: str
    blocked_constraint: str
    std_mins: int
    sta_mins: int


@dataclass(frozen=True)
class NoDestinationRejection:
    """A chain-termination event caused by missing Markov or turnaround data.

    Attributes
    ----------
    aircraft_reg:
        Aircraft registration.
    operator:
        ICAO airline code.
    wake:
        ICAO wake-turbulence category.
    airport:
        Airport where the aircraft was stranded (no continuation found).
    prev_origin:
        Airport from which the aircraft arrived at *airport*.
    arrival_mins:
        Arrival time at *airport* in minutes from window start.
    reason:
        ``"no_markov_data"`` — no Markov destinations available for the
        (operator, wake, prev_origin, airport, hour) key;
        ``"missing_turnaround"`` — turnaround sampling failed because
        lognormal parameters were absent for this airport/operator/wake.
    available_hours:
        Sorted departure hours for which Markov data *does* exist under
        this key (empty if the key is entirely absent).
    """

    aircraft_reg: str
    operator: str
    wake: str
    airport: str
    prev_origin: str
    arrival_mins: int
    reason: str
    available_hours: tuple


@dataclass(frozen=True)
class EndOfDayRejection:
    """A chain-termination event caused by the computed departure falling beyond
    the scheduling window.

    Attributes
    ----------
    aircraft_reg:
        Aircraft registration.
    operator:
        ICAO airline code.
    wake:
        ICAO wake-turbulence category.
    airport:
        Airport where the chain ended (the aircraft's final resting location).
    std_mins:
        The computed next departure time that exceeded the window length
        (i.e. why the chain could not continue).
    chain_length:
        Number of flights in the chain at termination (including the
        seeded initial/prior flights).
    """

    aircraft_reg: str
    operator: str
    wake: str
    airport: str
    std_mins: int
    chain_length: int


# ---------------------------------------------------------------------------
# RejectionLog
# ---------------------------------------------------------------------------


class RejectionLog:
    """Collect and query detailed rejection events produced during schedule generation.

    All three rejection categories are recorded in insertion order and
    exposed through query helpers that group, filter, and summarise events.
    """

    def __init__(self) -> None:
        self._capacity: List[CapacityRejection] = []
        self._no_destination: List[NoDestinationRejection] = []
        self._end_of_day: List[EndOfDayRejection] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_capacity(
        self,
        *,
        aircraft_reg: str,
        operator: str,
        wake: str,
        origin: str,
        destination: str,
        blocked_airport: str,
        blocked_movement: str,
        blocked_constraint: str,
        std_mins: int,
        sta_mins: int,
    ) -> None:
        """Append a capacity-check failure event."""
        self._capacity.append(
            CapacityRejection(
                aircraft_reg=aircraft_reg,
                operator=operator,
                wake=wake,
                origin=origin,
                destination=destination,
                blocked_airport=blocked_airport,
                blocked_movement=blocked_movement,
                blocked_constraint=blocked_constraint,
                std_mins=std_mins,
                sta_mins=sta_mins,
            )
        )

    def record_no_destination(
        self,
        *,
        aircraft_reg: str,
        operator: str,
        wake: str,
        airport: str,
        prev_origin: str,
        arrival_mins: int,
        reason: str,
        available_hours: Sequence[int],
    ) -> None:
        """Append a no-destination (Markov or turnaround data missing) termination event."""
        self._no_destination.append(
            NoDestinationRejection(
                aircraft_reg=aircraft_reg,
                operator=operator,
                wake=wake,
                airport=airport,
                prev_origin=prev_origin,
                arrival_mins=arrival_mins,
                reason=reason,
                available_hours=tuple(available_hours),
            )
        )

    def record_end_of_day(
        self,
        *,
        aircraft_reg: str,
        operator: str,
        wake: str,
        airport: str,
        std_mins: int,
        chain_length: int,
    ) -> None:
        """Append an end-of-day termination event."""
        self._end_of_day.append(
            EndOfDayRejection(
                aircraft_reg=aircraft_reg,
                operator=operator,
                wake=wake,
                airport=airport,
                std_mins=std_mins,
                chain_length=chain_length,
            )
        )

    # ------------------------------------------------------------------
    # Read-only access to raw event lists
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> List[CapacityRejection]:
        """All capacity-check failure events (read-only view)."""
        return self._capacity

    @property
    def no_destination(self) -> List[NoDestinationRejection]:
        """All no-destination termination events (read-only view)."""
        return self._no_destination

    @property
    def end_of_day(self) -> List[EndOfDayRejection]:
        """All end-of-day termination events (read-only view)."""
        return self._end_of_day

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def total_capacity_at(self, airport: str) -> int:
        """Total capacity-check failures where *airport* was the blocked airport."""
        return sum(1 for r in self._capacity if r.blocked_airport == airport)

    def total_no_destination_at(self, airport: str) -> int:
        """Total no-destination terminations where the aircraft was stranded at *airport*."""
        return sum(1 for r in self._no_destination if r.airport == airport)

    def total_end_of_day_at(self, airport: str) -> int:
        """Total end-of-day terminations where the chain ended at *airport*."""
        return sum(1 for r in self._end_of_day if r.airport == airport)

    def capacity_by_airport(self) -> Dict[str, int]:
        """Capacity-check failures grouped by blocked airport, sorted descending."""
        counts: Dict[str, int] = defaultdict(int)
        for r in self._capacity:
            counts[r.blocked_airport] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def no_destination_by_airport(self) -> Dict[str, int]:
        """No-destination terminations grouped by stranded airport, sorted descending."""
        counts: Dict[str, int] = defaultdict(int)
        for r in self._no_destination:
            counts[r.airport] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def end_of_day_by_airport(self) -> Dict[str, int]:
        """End-of-day terminations grouped by final airport, sorted descending."""
        counts: Dict[str, int] = defaultdict(int)
        for r in self._end_of_day:
            counts[r.airport] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------
    # Formatted reports
    # ------------------------------------------------------------------

    def airport_summary(self, airport: str) -> str:
        """Return a detailed multi-section breakdown for a single airport.

        Covers all three rejection categories with sub-breakdowns by
        constraint type, movement direction, operator, wake, and hour band.
        """
        lines: List[str] = [
            f"=== Rejection Summary for {airport} ===",
            "",
        ]

        # --- Capacity ---
        cap_events = [r for r in self._capacity if r.blocked_airport == airport]
        lines.append(f"--- Capacity Rejections: {len(cap_events)} ---")
        if cap_events:
            by_movement: Dict[str, int] = defaultdict(int)
            by_constraint: Dict[str, int] = defaultdict(int)
            by_operator: Dict[str, int] = defaultdict(int)
            by_wake: Dict[str, int] = defaultdict(int)
            by_hour: Dict[int, int] = defaultdict(int)
            for r in cap_events:
                by_movement[r.blocked_movement] += 1
                by_constraint[r.blocked_constraint] += 1
                by_operator[r.operator] += 1
                by_wake[r.wake] += 1
                hour = r.std_mins // 60
                by_hour[hour] += 1

            lines.append("  By movement type:")
            for mv, cnt in sorted(by_movement.items()):
                lines.append(f"    {mv}: {cnt}")

            lines.append("  By constraint type:")
            for ct, cnt in sorted(by_constraint.items()):
                lines.append(f"    {ct}: {cnt}")

            lines.append("  By operator (top 10):")
            for op, cnt in sorted(by_operator.items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"    {op}: {cnt}")

            lines.append("  By wake category:")
            for wk, cnt in sorted(by_wake.items()):
                lines.append(f"    {wk}: {cnt}")

            lines.append("  By departure hour (STD):")
            for hr in sorted(by_hour):
                lines.append(f"    {hr:02d}h: {by_hour[hr]}")
        lines.append("")

        # --- No Destination ---
        nd_events = [r for r in self._no_destination if r.airport == airport]
        lines.append(f"--- No-Destination Terminations: {len(nd_events)} ---")
        if nd_events:
            by_reason: Dict[str, int] = defaultdict(int)
            by_operator: Dict[str, int] = defaultdict(int)
            by_wake: Dict[str, int] = defaultdict(int)
            by_hour: Dict[int, int] = defaultdict(int)
            for r in nd_events:
                by_reason[r.reason] += 1
                by_operator[r.operator] += 1
                by_wake[r.wake] += 1
                by_hour[r.arrival_mins // 60] += 1

            lines.append("  By reason:")
            for rs, cnt in sorted(by_reason.items()):
                lines.append(f"    {rs}: {cnt}")

            lines.append("  By operator (top 10):")
            for op, cnt in sorted(by_operator.items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"    {op}: {cnt}")

            lines.append("  By wake category:")
            for wk, cnt in sorted(by_wake.items()):
                lines.append(f"    {wk}: {cnt}")

            lines.append("  By arrival hour:")
            for hr in sorted(by_hour):
                lines.append(f"    {hr:02d}h: {by_hour[hr]}")
        lines.append("")

        # --- End of Day ---
        eod_events = [r for r in self._end_of_day if r.airport == airport]
        lines.append(f"--- End-of-Day Terminations: {len(eod_events)} ---")
        if eod_events:
            by_operator: Dict[str, int] = defaultdict(int)
            by_wake: Dict[str, int] = defaultdict(int)
            by_hour: Dict[int, int] = defaultdict(int)
            chain_lengths = [r.chain_length for r in eod_events]
            for r in eod_events:
                by_operator[r.operator] += 1
                by_wake[r.wake] += 1
                by_hour[r.std_mins // 60] += 1

            lines.append(f"  Chain length range: {min(chain_lengths)}–{max(chain_lengths)}")

            lines.append("  By operator (top 10):")
            for op, cnt in sorted(by_operator.items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"    {op}: {cnt}")

            lines.append("  By wake category:")
            for wk, cnt in sorted(by_wake.items()):
                lines.append(f"    {wk}: {cnt}")

            lines.append("  By overflow departure hour (STD):")
            for hr in sorted(by_hour):
                lines.append(f"    {hr:02d}h: {by_hour[hr]}")

        return "\n".join(lines)

    def summary(self, top_n: int = 10) -> str:
        """Return a full multi-section rejection summary.

        Parameters
        ----------
        top_n:
            Number of airports to show in the per-airport ranking tables.
        """
        lines: List[str] = [
            "=== Rejection Log Summary ===",
            "",
        ]

        # --- Capacity ---
        lines.append(f"--- Capacity Check Failures: {len(self._capacity)} total ---")
        lines.append(
            "  (Includes all individual check failures; a chain may try multiple"
        )
        lines.append(
            "   destinations before succeeding — those failures are included here.)"
        )
        if self._capacity:
            by_mv: Dict[str, int] = defaultdict(int)
            by_ct: Dict[str, int] = defaultdict(int)
            for r in self._capacity:
                by_mv[r.blocked_movement] += 1
                by_ct[r.blocked_constraint] += 1
            lines.append("  By movement type:")
            for mv, cnt in sorted(by_mv.items()):
                lines.append(f"    {mv}: {cnt}")
            lines.append("  By constraint type:")
            for ct, cnt in sorted(by_ct.items()):
                lines.append(f"    {ct}: {cnt}")
            lines.append(f"  Top {top_n} blocked airports:")
            for ap, cnt in list(self.capacity_by_airport().items())[:top_n]:
                lines.append(f"    {ap}: {cnt}")
        lines.append("")

        # --- No Destination ---
        lines.append(f"--- No-Destination Terminations: {len(self._no_destination)} total ---")
        if self._no_destination:
            by_reason: Dict[str, int] = defaultdict(int)
            for r in self._no_destination:
                by_reason[r.reason] += 1
            lines.append("  By reason:")
            for rs, cnt in sorted(by_reason.items()):
                lines.append(f"    {rs}: {cnt}")
            lines.append(f"  Top {top_n} stranded airports:")
            for ap, cnt in list(self.no_destination_by_airport().items())[:top_n]:
                lines.append(f"    {ap}: {cnt}")
        lines.append("")

        # --- End of Day ---
        lines.append(f"--- End-of-Day Terminations: {len(self._end_of_day)} total ---")
        if self._end_of_day:
            by_hour: Dict[int, int] = defaultdict(int)
            for r in self._end_of_day:
                by_hour[r.std_mins // 60] += 1
            lines.append("  By overflow departure hour:")
            for hr in sorted(by_hour):
                lines.append(f"    {hr:02d}h: {by_hour[hr]}")
            lines.append(f"  Top {top_n} final airports:")
            for ap, cnt in list(self.end_of_day_by_airport().items())[:top_n]:
                lines.append(f"    {ap}: {cnt}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return all rejection events as a single tidy ``pd.DataFrame``.

        Each row represents one event.  The ``rejection_type`` column takes
        values ``"capacity"``, ``"no_destination"``, or ``"end_of_day"``.
        Columns that do not apply to a given type are filled with ``None``.

        Columns
        -------
        rejection_type, aircraft_reg, operator, wake,
        airport, origin, destination,
        blocked_airport, blocked_movement, blocked_constraint,
        std_mins, sta_mins,
        prev_origin, arrival_mins, reason, available_hours,
        chain_length
        """
        rows: List[dict] = []

        for r in self._capacity:
            rows.append({
                "rejection_type": "capacity",
                "aircraft_reg": r.aircraft_reg,
                "operator": r.operator,
                "wake": r.wake,
                "airport": r.blocked_airport,
                "origin": r.origin,
                "destination": r.destination,
                "blocked_airport": r.blocked_airport,
                "blocked_movement": r.blocked_movement,
                "blocked_constraint": r.blocked_constraint,
                "std_mins": r.std_mins,
                "sta_mins": r.sta_mins,
                "prev_origin": None,
                "arrival_mins": None,
                "reason": None,
                "available_hours": None,
                "chain_length": None,
            })

        for r in self._no_destination:
            rows.append({
                "rejection_type": "no_destination",
                "aircraft_reg": r.aircraft_reg,
                "operator": r.operator,
                "wake": r.wake,
                "airport": r.airport,
                "origin": None,
                "destination": None,
                "blocked_airport": None,
                "blocked_movement": None,
                "blocked_constraint": None,
                "std_mins": None,
                "sta_mins": None,
                "prev_origin": r.prev_origin,
                "arrival_mins": r.arrival_mins,
                "reason": r.reason,
                "available_hours": r.available_hours if r.available_hours else None,
                "chain_length": None,
            })

        for r in self._end_of_day:
            rows.append({
                "rejection_type": "end_of_day",
                "aircraft_reg": r.aircraft_reg,
                "operator": r.operator,
                "wake": r.wake,
                "airport": r.airport,
                "origin": None,
                "destination": None,
                "blocked_airport": None,
                "blocked_movement": None,
                "blocked_constraint": None,
                "std_mins": r.std_mins,
                "sta_mins": None,
                "prev_origin": None,
                "arrival_mins": None,
                "reason": None,
                "available_hours": None,
                "chain_length": r.chain_length,
            })

        if not rows:
            return pd.DataFrame(columns=[
                "rejection_type", "aircraft_reg", "operator", "wake",
                "airport", "origin", "destination",
                "blocked_airport", "blocked_movement", "blocked_constraint",
                "std_mins", "sta_mins",
                "prev_origin", "arrival_mins", "reason", "available_hours",
                "chain_length",
            ])

        return pd.DataFrame(rows)
