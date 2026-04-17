from .clean_data import clean as clean_data

__all__ = ["clean_bts_data", "clean_data"]


def clean_bts_data(*args, **kwargs):
    """Clean BTS on-time data into the normalized ROSTER schedule schema."""
    from .clean_bts import clean_bts_data as _clean_bts_data

    return _clean_bts_data(*args, **kwargs)
