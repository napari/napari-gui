"""PerfEvent class.
"""
from typing import Optional


class PerfEvent:
    """One perf event represents a span of time.

    Parameters
    ----------
    name : str
        The name of this event like "draw".
    start_ns : int
        Start time in nanoseconds.
    end_ns : int
        End time in nanoseconds.
    category :str
        Comma separated categories such has "render,update".
    **kwargs : dict
        Additional keyword arguments for the "args" field of the event.
    Notes
    -----
    The time stamps are from perf_counter_ns() and do not indicate time of
    day. The origin is arbitrary, but subtracting two counters results in
    a span of wall clock time.
    """

    def __init__(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        category: Optional[str] = None,
        **kwargs,
    ):
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.category = category
        self.args = kwargs

    @property
    def start_us(self):
        return self.start_ns / 1e3

    @property
    def start_ms(self):
        return self.start_ns / 1e6

    @property
    def duration_ns(self):
        return self.end_ns - self.start_ns

    @property
    def duration_us(self):
        return self.duration_ns / 1e3

    @property
    def duration_ms(self):
        return self.duration_ns / 1e6
