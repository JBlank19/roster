from .auxiliary.airlines import generate_airlines
from .auxiliary.airports import generate_airports
from .config import MarkovContext, PipelineConfig
from .data_cleaning.clean_data import clean as clean_data
from .auxiliary.fleet import generate_fleet
from .distribution_analysis.markov import generate_markov
from .auxiliary.routes import generate_routes
from .distribution_analysis.scheduled_flight_time import analyze_flight_time_distribution
from .schedule import generate_schedule
from .schedule import GenerationStats, RejectionLog
from .distribution_analysis.scheduled_turnaround import analyze_turnaround_distribution


def clean_bts_data(*args, **kwargs):
    """Clean BTS on-time data into the normalized ROSTER schedule schema."""
    from .data_cleaning.clean_bts import clean_bts_data as _clean_bts_data

    return _clean_bts_data(*args, **kwargs)
