from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from ._rejection_log import RejectionLog


@dataclass
class GenerationStats:
    """Track schedule-generation statistics and diagnostics."""

    total_aircraft: int = 0
    successful_chains: int = 0
    total_flights: int = 0

    end_of_day: int = 0
    no_destinations: int = 0
    capacity_exhausted: int = 0

    single_flight_passthrough: int = 0
    single_flight_with_prior: int = 0
    prior_flight_pasted: int = 0
    prior_only_pasted: int = 0

    dest_found_primary_exact: int = 0
    dest_found_primary_expanded: int = 0
    dest_found_fallback_expanded: int = 0
    dest_found_return_to_origin: int = 0

    examples_no_destinations: List[str] = field(default_factory=list)
    examples_capacity_exhausted: List[str] = field(default_factory=list)

    rejection_log: RejectionLog = field(default_factory=RejectionLog)

    single_flight_total: int = 0
    single_flight_end_of_day: int = 0
    single_flight_no_destinations: int = 0
    single_flight_capacity_exhausted: int = 0
    single_flight_termination_hours: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add_example_no_destinations(self, msg: str) -> None:
        """Store at most two no-destination examples."""
        if len(self.examples_no_destinations) < 2:
            self.examples_no_destinations.append(msg)

    def add_example_capacity_exhausted(self, msg: str) -> None:
        """Store at most two capacity-exhausted examples."""
        if len(self.examples_capacity_exhausted) < 2:
            self.examples_capacity_exhausted.append(msg)

    def _termination_lines(self) -> List[str]:
        return [
            "--- Chain Termination ---",
            f"  Single-flight passthrough (real data): {self.single_flight_passthrough}",
            f"    With overnight arrival: {self.single_flight_with_prior}",
            f"  Prior flights pasted (all aircraft): {self.prior_flight_pasted}",
            f"  Prior-only aircraft pasted: {self.prior_only_pasted}",
            f"  End of day: {self.end_of_day}",
            f"  No Markov data: {self.no_destinations}",
            f"  Capacity exhausted: {self.capacity_exhausted}",
            "",
        ]

    def _fallback_lines(self) -> List[str]:
        return [
            "--- Fallback Usage ---",
            f"  Primary Exact: {self.dest_found_primary_exact}",
            f"  Primary Expanded: {self.dest_found_primary_expanded}",
            f"  Fallback Expanded: {self.dest_found_fallback_expanded}",
            f"  Return to Origin: {self.dest_found_return_to_origin}",
            "",
        ]

    def _single_flight_lines(self) -> List[str]:
        lines = [
            "--- Single Flight Chains Analysis ---",
            f"  Total Single Flights: {self.single_flight_total}",
            "  Reasons:",
            f"    End of Day (Turnaround > Midnight): {self.single_flight_end_of_day}",
            f"    No Markov Data: {self.single_flight_no_destinations}",
            f"    Capacity Exhausted: {self.single_flight_capacity_exhausted}",
            "",
            "  Termination Hour Distribution (End of Day):",
        ]
        lines.extend(
            f"    {hour:02d}h: {count}"
            for hour, count in sorted(self.single_flight_termination_hours.items())
            if count > 0
        )
        return lines

    def summary(self) -> str:
        """Return a printable multi-section generation summary."""
        lines = [
            "=== Generation Statistics ===",
            f"Aircraft: {self.successful_chains}/{self.total_aircraft} successful",
            f"Total flights: {self.total_flights}",
            "",
        ]
        lines.extend(self._termination_lines())
        lines.extend(self._fallback_lines())
        lines.extend(self._single_flight_lines())

        if self.examples_no_destinations:
            lines.append("")
            lines.append("--- Examples: No Destinations ---")
            lines.extend(self.examples_no_destinations)

        if self.examples_capacity_exhausted:
            lines.append("")
            lines.append("--- Examples: Capacity Exhausted ---")
            lines.extend(self.examples_capacity_exhausted)

        return "\n".join(lines)
