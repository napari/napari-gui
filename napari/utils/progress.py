from typing import Iterable, Optional

from tqdm import tqdm

from napari.utils.events.event import EmitterGroup, Event

from ..utils.events.containers import EventedSet
from ..utils.translations import trans


class progress(tqdm):
    """This class inherits from tqdm and provides an interface for
    progress bars in the napari viewer. Progress bars can be created
    directly by wrapping an iterable or by providing a total number
    of expected updates.

    See tqdm.tqdm API for valid args and kwargs:
    https://tqdm.github.io/docs/tqdm/

    Also, any keyword arguments to the :class:`ProgressBar` `QWidget`
    are also accepted and will be passed to the ``ProgressBar``.

    Examples
    --------

    >>> def long_running(steps=10, delay=0.1):
    ...     for i in progress(range(steps)):
    ...         sleep(delay)

    it can also be used as a context manager:

    >>> def long_running(steps=10, repeats=4, delay=0.1):
    ...     with progress(range(steps)) as pbr:
    ...         for i in pbr:
    ...             sleep(delay)

    or equivalently, using the `progrange` shorthand
    ...     with progrange(steps) as pbr:
    ...         for i in pbr:
    ...             sleep(delay)

    For manual updates:

    >>> def manual_updates(total):
    ...     pbr = progress(total=total)
    ...     sleep(10)
    ...     pbr.set_description("Step 1 Complete")
    ...     pbr.update(1)
    ...     # must call pbr.close() when using outside for loop
    ...     # or context manager
    ...     pbr.close()

    """

    monitor_interval = 0  # set to 0 to disable the thread
    all_progress = EventedSet()

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        nest_under: Optional['progress'] = None,
        *args,
        **kwargs,
    ) -> None:
        self.events = EmitterGroup(
            value=Event, description=Event, overflow=Event, eta=Event
        )
        self.nest_under = nest_under
        super().__init__(iterable, desc, total, *args, **kwargs)

        if not self.desc:
            self.set_description(trans._("progress"))
        progress.all_progress.add(self)

    def display(self, msg: str = None, pos: int = None) -> None:
        """Update the display."""
        if not self.gui:
            super().display(msg, pos)
        # TODO: This could break if user is formatting their own terminal tqdm
        if self.total != 0:
            etas = str(self).split('|')[-1]

        self.events.eta(value=etas)
        self.events.value(value=self.n)

    def increment_with_overflow(self):
        """Update if not exceeding total, else set indeterminate range."""
        if self.n == self.total:
            self.total = 0
            self.events.overflow()
        else:
            self.update(1)

    def set_description(self, desc):
        """Update progress bar description"""
        super().set_description(desc, refresh=True)
        self.events.description(value=desc)

    def close(self):
        """Closes and deletes the progress object"""
        if self.disable:
            return
        progress.all_progress.remove(self)
        super().close()


def progrange(*args, **kwargs):
    """Shorthand for `progress(range(*args), **kwargs)`.

    Adds tqdm based progress bar to napari viewer, if it
    exists, and returns the wrapped range object.

    Returns
    -------
    progress
        wrapped range object
    """
    return progress(range(*args), **kwargs)
