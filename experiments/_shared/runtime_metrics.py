"""Portable peak resident-memory diagnostic for simulation receipts."""
import sys

import psutil


def peak_rss_bytes():
    """Peak process RSS in bytes; fall back to current RSS when unavailable."""
    info = psutil.Process().memory_info()
    peak_windows = getattr(info, "peak_wset", None)
    if peak_windows is not None:
        return int(peak_windows)
    try:
        import resource
        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(maximum if sys.platform == "darwin" else maximum * 1024)
    except (ImportError, AttributeError):
        return int(info.rss)
