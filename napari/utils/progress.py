from collections.abc import Callable, Generator, Iterable, Iterator
from itertools import takewhile
from typing import Optional

from tqdm import tqdm

from napari.utils.events.containers import EventedSet
from napari.utils.events.event import EmitterGroup, Event
from napari.utils.translations import trans

__all__ = ['cancelable_progress', 'progrange', 'progress']


class progress(tqdm):
    """This class inherits from tqdm and provides an interface for
    progress bars in the napari viewer. Progress bars can be created
    directly by wrapping an iterable or by providing a total number
    of expected updates.

    While this interface is primarily designed to be displayed in
    the viewer, it can also be used without a viewer open, in which
    case it behaves identically to tqdm and produces the progress
    bar in the terminal.

    See tqdm.tqdm API for valid args and kwargs:
    https://tqdm.github.io/docs/tqdm/

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

    .. code-block:: python

        with progrange(steps) as pbr:
            for i in pbr:
                sleep(delay)

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
    # to give us a way to hook into the creation and update of progress objects
    # without progress knowing anything about a Viewer, we track all instances in
    # this evented *class* attribute, accessed through `progress._all_instances`
    # this allows the ActivityDialog to find out about new progress objects and
    # hook up GUI progress bars to its update events
    _all_instances: EventedSet['progress'] = EventedSet()

    def __init__(
        self,
        iterable: Iterable | None = None,
        desc: str | None = None,
        total: int | None = None,
        nest_under: Optional['progress'] = None,
        *args,
        **kwargs,
    ) -> None:
        self.events = EmitterGroup(
            value=Event,
            description=Event,
            overflow=Event,
            eta=Event,
            total=Event,
        )
        self.nest_under = nest_under
        self.is_init = True
        super().__init__(iterable, desc, total, *args, **kwargs)

        # if the progress bar is set to disable the 'desc' member is not set by the
        # tqdm super constructor, so we set it to a dummy value to avoid errors thrown below
        if self.disable:
            self.desc = ''
        if not self.desc:
            self.set_description(trans._('progress'))
        progress._all_instances.add(self)
        self.is_init = False

    def __repr__(self) -> str:
        return self.desc

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, total):
        self._total = total
        self.events.total(value=self.total)

    def display(self, msg: str | None = None, pos: int | None = None) -> None:
        """Update the display and emit eta event."""
        # just plain tqdm if we don't have gui
        if not self.gui and not self.is_init:
            super().display(msg, pos)
            return
        # TODO: This could break if user is formatting their own terminal tqdm
        etas = str(self).split('|')[-1] if self.total != 0 else ''
        self.events.eta(value=etas)

    def update(self, n=1):
        """Update progress value by n and emit value event"""
        super().update(n)
        self.events.value(value=self.n)

    def increment_with_overflow(self):
        """Update if not exceeding total, else set indeterminate range."""
        if self.n == self.total:
            self.total = 0
            self.events.overflow()
        else:
            self.update(1)

    def set_description(self, desc):
        """Update progress description and emit description event."""
        super().set_description(desc, refresh=True)
        self.events.description(value=desc)

    def close(self):
        """Close progress object and emit event."""
        if self.disable:
            return
        progress._all_instances.remove(self)
        super().close()


def progrange(*args, **kwargs):
    """Shorthand for ``progress(range(*args), **kwargs)``.

    Adds tqdm based progress bar to napari viewer, if it
    exists, and returns the wrapped range object.

    Returns
    -------
    progress
        wrapped range object

    """
    return progress(range(*args), **kwargs)


class cancelable_progress(progress):
    """This class inherits from progress, providing the additional
    ability to cancel expensive executions. When progress is
    canceled by the user in the napari UI, two things will happen:

    Firstly, the is_canceled attribute will become True, and the
    for loop will terminate after the current iteration, regardless
    of whether or not the iterator had more items.

    Secondly, cancel_callback will be called, allowing the computation
    to close resources, repair state, etc.

    See napari.utils.progress and tqdm.tqdm API for valid args and kwargs:
    https://tqdm.github.io/docs/tqdm/

    Examples
    --------

    >>> def long_running(steps=10, delay=0.1):
    ...     def on_cancel():
    ...         print("Canceled operation")
    ...     for i in cancelable_progress(range(steps), cancel_callback=on_cancel):
    ...         sleep(delay)
    """

    def __init__(
        self,
        iterable: Iterable | None = None,
        desc: str | None = None,
        total: int | None = None,
        nest_under: Optional['progress'] = None,
        cancel_callback: Callable | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.cancel_callback = cancel_callback
        self.is_canceled = False

        super().__init__(iterable, desc, total, nest_under, *args, **kwargs)

    def __iter__(self) -> Iterator:
        itr = super().__iter__()

        def is_canceled(_):
            if self.is_canceled:
                # If we've canceled, run the callback and then notify takewhile
                if self.cancel_callback:
                    self.cancel_callback()
                # Perform additional cleanup for generators
                if isinstance(self.iterable, Generator):
                    self.iterable.close()
                return False
                # Otherwise, continue
            return True

        return takewhile(is_canceled, itr)

    def cancel(self):
        """Cancels the execution of the underlying computation.
        Note that the current iteration will be allowed to complete, however
        future iterations will not be run.
        """
        self.is_canceled = True
