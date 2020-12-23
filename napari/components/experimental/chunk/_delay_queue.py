"""DelayQueue class.

Delays a configurable amount of time before submitting a request.
"""
import logging
import threading
import time
from typing import Callable, List, NamedTuple, Optional

from ....utils.perf import add_counter_event
from ._request import ChunkRequest

LOGGER = logging.getLogger("napari.loader")


class QueueEntry(NamedTuple):
    """The request we are doing to submit and when to submit it.

    Parameters
    ----------
    request : ChunkRequest
        The request to submit.
    submit_time : float
        The time to submit the request in time.time() seconds.
    """

    request: ChunkRequest
    submit_time: float


class DelayQueue(threading.Thread):
    """A threaded queue that delays request submission.

    The DelayQueue exists so we can avoid spamming the ChunkLoader loader
    pools with requests for chunks that are potentially going to be out of
    view before they are loaded.

    For example, when rapidly scrolling through slices, it would be
    pointless to submit requests for every new slice we pass through.
    Instead by using a small delay, we hold back submitting our requests
    until the user has settled on a specific slice.

    Similarly with the Octree, when panning and zooming rapidly we might
    choose to delay so that we don't waste time loading chunks that
    will quickly be out of view.

    With the Octree however we do want to show something as you pan and
    zoom around. For this reason ChunkLoader can have multiple loader pools
    each with different delays Typically we we delay the "ideal level"
    chunks the most, but we load coarser levels sooner. So we show the user
    something quickly, but we only load the full set of ideal chunks when
    the camera movement has settled down.

    Parameters
    ----------
    delay_queue_ms : float
        Delay the request for this many milliseconds.
    submit_func
        Call this function to submit the request.

    Attributes
    ----------
    delay_seconds : float
        Delay each request by this many seconds.
    _submit_func : Callable[[ChunkRequest], None]
        Call this function to submit the request.
    _entries : List[QueueEntry]
        The entries in the queue.
    _lock : threading.Lock
        Lock access to the self.entires queue.
    _event : threading.Event
        Event we signal to wake up the worker.
    """

    def __init__(
        self,
        delay_queue_ms: float,
        submit_func: Callable[[ChunkRequest], None],
    ):
        super().__init__(daemon=True)
        self.delay_seconds: float = (delay_queue_ms / 1000)
        self._submit_func = submit_func

        self._entries: List[QueueEntry] = []
        self._lock = threading.Lock()
        self._event = threading.Event()

        self.start()

    def add(self, request) -> None:
        """Insert the request into the queue.

        Parameters
        ----------
        request : ChunkRequest
            Insert this request into the queue.
        """
        if self.delay_seconds == 0:
            self._submit_func(request)  # Submit with no delay.
            return

        LOGGER.info("DelayQueue.add: %s", request.location)

        # Create entry with the time to submit it.
        submit_time = time.time() + self.delay_seconds
        entry = QueueEntry(request, submit_time)

        with self._lock:
            self._entries.append(entry)
            num_entries = len(self._entries)

        add_counter_event("delay_queue", entries=num_entries)

        if num_entries == 1:
            self._event.set()  # The list was empty so wake up the worker.

    def cancel_requests(
        self, should_cancel: Callable[[ChunkRequest], bool]
    ) -> List[ChunkRequest]:
        """Cancel pending requests based on the given filter.

        Parameters
        ----------
        should_cancel : Callable[[ChunkRequest], bool]
            Cancel the request if this returns True.

        Return
        ------
        List[ChunkRequests]
            The requests that were cancelled, if any.
        """
        keep = []
        cancel = []
        with self._lock:
            for entry in self._entries:
                if should_cancel(entry.request):
                    cancel.append(entry.request)
                else:
                    keep.append(entry)
            self._entries = keep

        return cancel

    def submit(self, entry: QueueEntry, now: float) -> bool:
        """Submit and return True if entry is ready to be submitted.

        Parameters
        ----------
        entry : QueueEntry
            The entry to potentially submit.
        now : float
            Current time in seconds.

        Return
        ------
        bool
            True if the entry was submitted.
        """
        # If entry is due to be submitted.
        if entry.submit_time < now:
            LOGGER.info("DelayQueue.submit: %s", entry.request.location)
            self._submit_func(entry.request)
            return True  # We submitted this request.
        return False

    def run(self):
        """The DelayQueue thread's main method.

        Submit all due entires, then sleep or wait on self._event
        for new entries.
        """
        while True:
            now = time.time()

            with self._lock:
                seconds = self._submit_due_entries(now)
                num_entries = len(self._entries)

            add_counter_event("delay_queue", entries=num_entries)

            if seconds is None:
                # There were no entries left, so wait until there is one.
                self._event.wait()
                self._event.clear()
            else:
                # Sleep until the next entry is due. This will tend to
                # oversleep by a few milliseconds, but close enough for our
                # purposes. Once we wake up we'll submit all due entries.
                # So we won't miss any.
                time.sleep(seconds)

    def _submit_due_entries(self, now: float) -> Optional[float]:
        """Submit all due entries, oldest to newest.

        Parameters
        ----------
        now : float
            Current time in seconds.

        Returns
        -------
        Optional[float]
            Seconds until next entry is due, or None if no next entry.
        """
        while self._entries:
            # Submit the oldest entry if it's due.
            if self.submit(self._entries[0], now):
                self._entries.pop(0)  # Remove the one we just submitted.
            else:
                # Oldest entry is not due, return time until it is.
                return self._entries[0].submit_time - now

        return None  # There are no more entries.

    def flush(self):
        """Submit all entries right now."""
        with self._lock:
            for entry in self._entries:
                self._submit_func(entry.request)
            self._entries = []
