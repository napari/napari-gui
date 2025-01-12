import inspect
import warnings
from collections.abc import Callable, Sequence
from functools import partial, wraps
from types import FunctionType, GeneratorType
from typing import (
    TypeVar,
)

from superqt.utils import _qthreading

from napari.utils.progress import progress
from napari.utils.translations import trans

__all__ = [
    'FunctionWorker',
    'GeneratorWorker',
    'create_worker',
    'register_threadworker_processors',
    'thread_worker',
]

wait_for_workers_to_quit = _qthreading.WorkerBase.await_workers


class _NotifyingMixin:
    def __init__(self: _qthreading.WorkerBase, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.errored.connect(self._relay_error)
        self.warned.connect(self._relay_warning)

    def _relay_error(self, exc: Exception):
        from napari.utils.notifications import notification_manager

        notification_manager.receive_error(type(exc), exc, exc.__traceback__)

    def _relay_warning(self, show_warn_args: tuple):
        from napari.utils.notifications import notification_manager

        notification_manager.receive_warning(*show_warn_args)


_Y = TypeVar('_Y')
_S = TypeVar('_S')
_R = TypeVar('_R')


class FunctionWorker(_qthreading.FunctionWorker[_R], _NotifyingMixin): ...


class GeneratorWorker(
    _qthreading.GeneratorWorker[_Y, _S, _R], _NotifyingMixin
): ...


# these are re-implemented from superqt just to provide progress


def create_worker(
    func: FunctionType | GeneratorType,
    *args,
    _start_thread: bool | None = None,
    _connect: dict[str, Callable | Sequence[Callable]] | None = None,
    _progress: bool | dict[str, int | bool | str] | None = None,
    _worker_class: type[GeneratorWorker] | type[FunctionWorker] | None = None,
    _ignore_errors: bool = False,
    **kwargs,
) -> FunctionWorker | GeneratorWorker:
    """Convenience function to start a function in another thread.

    By default, uses :class:`Worker`, but a custom ``WorkerBase`` subclass may
    be provided.  If so, it must be a subclass of :class:`Worker`, which
    defines a standard set of signals and a run method.

    Parameters
    ----------
    func : Callable
        The function to call in another thread.
    _start_thread : bool, optional
        Whether to immediaetly start the thread.  If False, the returned worker
        must be manually started with ``worker.start()``. by default it will be
        ``False`` if the ``_connect`` argument is ``None``, otherwise ``True``.
    _connect : Dict[str, Union[Callable, Sequence]], optional
        A mapping of ``"signal_name"`` -> ``callable`` or list of ``callable``:
        callback functions to connect to the various signals offered by the
        worker class. by default None
    _progress : Union[bool, Dict[str, Union[int, bool, str]]], optional
        Can be True, to provide indeterminate progress bar, or dictionary.
        If dict, requires mapping of 'total' to number of expected yields.
        If total is not provided, progress bar will be indeterminate. Will connect
        progress bar update to yields and display this progress in the viewer.
        Can also take a mapping of 'desc' to the progress bar description.
        Progress bar will become indeterminate when number of yields exceeds 'total'.
        By default None.
    _worker_class : Type[WorkerBase], optional
        The :class`WorkerBase` to instantiate, by default
        :class:`FunctionWorker` will be used if ``func`` is a regular function,
        and :class:`GeneratorWorker` will be used if it is a generator.
    _ignore_errors : bool, optional
        If ``False`` (the default), errors raised in the other thread will be
        reraised in the main thread (makes debugging significantly easier).
    *args
        will be passed to ``func``
    **kwargs
        will be passed to ``func``

    Returns
    -------
    worker : WorkerBase
        An instantiated worker.  If ``_start_thread`` was ``False``, the worker
        will have a `.start()` method that can be used to start the thread.

    Raises
    ------
    TypeError
        If a worker_class is provided that is not a subclass of WorkerBase.
    TypeError
        If _connect is provided and is not a dict of ``{str: callable}``
    TypeError
        If _progress is provided and function is not a generator

    Examples
    --------
    .. code-block:: python

        def long_function(duration):
            import time
            time.sleep(duration)

        worker = create_worker(long_function, 10)

    """
    # provide our own classes with the notification mixins
    if not _worker_class:
        if inspect.isgeneratorfunction(func):
            _worker_class = GeneratorWorker
        else:
            _worker_class = FunctionWorker

    worker = _qthreading.create_worker(
        func,
        *args,
        _start_thread=False,
        _connect=_connect,
        _worker_class=_worker_class,
        _ignore_errors=_ignore_errors,
        **kwargs,
    )

    # either True or a non-empty dictionary
    if _progress:
        if isinstance(_progress, bool):
            _progress = {}

        desc = _progress.get('desc', None)
        total = int(_progress.get('total', 0))
        if isinstance(worker, FunctionWorker) and total != 0:
            warnings.warn(
                trans._(
                    '_progress total != 0 but worker is FunctionWorker and will not yield. Returning indeterminate progress bar...',
                    deferred=True,
                ),
                RuntimeWarning,
            )
            total = 0

        with progress._all_instances.events.changed.blocker():
            pbar = progress(total=total, desc=desc)

        worker.started.connect(
            partial(
                lambda prog: progress._all_instances.events.changed(
                    added={prog}, removed={}
                ),
                pbar,
            )
        )
        worker.finished.connect(pbar.close)
        if total != 0 and isinstance(worker, GeneratorWorker):
            worker.yielded.connect(pbar.increment_with_overflow)

        worker.pbar = pbar

    if _start_thread is None:
        _start_thread = _connect is not None

    if _start_thread:
        worker.start()
    return worker


def thread_worker(
    function: Callable | None = None,
    start_thread: bool | None = None,
    connect: dict[str, Callable | Sequence[Callable]] | None = None,
    progress: bool | dict[str, int | bool | str] | None = None,
    worker_class: type[FunctionWorker] | type[GeneratorWorker] | None = None,
    ignore_errors: bool = False,
):
    """Decorator that runs a function in a separate thread when called.

    When called, the decorated function returns a :class:`WorkerBase`.  See
    :func:`create_worker` for additional keyword arguments that can be used
    when calling the function.

    The returned worker will have these signals:

    - *started*: emitted when the work is started
    - *finished*: emitted when the work is finished
    - *returned*: emitted with return value
    - *errored*: emitted with error object on Exception

    It will also have a ``worker.start()`` method that can be used to start
    execution of the function in another thread. (useful if you need to connect
    callbacks to signals prior to execution)

    If the decorated function is a generator, the returned worker will also
    provide these signals:

    - *yielded*: emitted with yielded values
    - *paused*: emitted when a running job has successfully paused
    - *resumed*: emitted when a paused job has successfully resumed
    - *aborted*: emitted when a running job is successfully aborted

    And these methods:

    - *quit*: ask the thread to quit
    - *toggle_paused*: toggle the running state of the thread.
    - *send*: send a value into the generator.  (This requires that your
        decorator function uses the ``value = yield`` syntax)

    Parameters
    ----------
    function : callable
        Function to call in another thread.  For communication between threads
        may be a generator function.
    start_thread : bool, optional
        Whether to immediaetly start the thread.  If False, the returned worker
        must be manually started with ``worker.start()``. by default it will be
        ``False`` if the ``_connect`` argument is ``None``, otherwise ``True``.
    connect : Dict[str, Union[Callable, Sequence]], optional
        A mapping of ``"signal_name"`` -> ``callable`` or list of ``callable``:
        callback functions to connect to the various signals offered by the
        worker class. by default None
    progress : Union[bool, Dict[str, Union[int, bool, str]]], optional
        Can be True, to provide indeterminate progress bar, or dictionary.
        If dict, requires mapping of 'total' to number of expected yields.
        If total is not provided, progress bar will be indeterminate. Will connect
        progress bar update to yields and display this progress in the viewer.
        Can also take a mapping of 'desc' to the progress bar description.
        Progress bar will become indeterminate when number of yields exceeds 'total'.
        By default None. Must be used in conjunction with a generator function.
    worker_class : Type[WorkerBase], optional
        The :class`WorkerBase` to instantiate, by default
        :class:`FunctionWorker` will be used if ``func`` is a regular function,
        and :class:`GeneratorWorker` will be used if it is a generator.
    ignore_errors : bool, optional
        If ``False`` (the default), errors raised in the other thread will be
        reraised in the main thread (makes debugging significantly easier).

    Returns
    -------
    callable
        function that creates a worker, puts it in a new thread and returns
        the worker instance.

    Examples
    --------
    .. code-block:: python

        @thread_worker
        def long_function(start, end):
            # do work, periodically yielding
            i = start
            while i <= end:
                time.sleep(0.1)
                yield i

            # do teardown
            return 'anything'

        # call the function to start running in another thread.
        worker = long_function()
        # connect signals here if desired... or they may be added using the
        # `connect` argument in the `@thread_worker` decorator... in which
        # case the worker will start immediately when long_function() is called
        worker.start()
    """

    def _inner(func):
        @wraps(func)
        def worker_function(*args, **kwargs):
            # decorator kwargs can be overridden at call time by using the
            # underscore-prefixed version of the kwarg.
            kwargs['_start_thread'] = kwargs.get('_start_thread', start_thread)
            kwargs['_connect'] = kwargs.get('_connect', connect)
            kwargs['_progress'] = kwargs.get('_progress', progress)
            kwargs['_worker_class'] = kwargs.get('_worker_class', worker_class)
            kwargs['_ignore_errors'] = kwargs.get(
                '_ignore_errors', ignore_errors
            )
            return create_worker(
                func,
                *args,
                **kwargs,
            )

        return worker_function

    return _inner if function is None else _inner(function)


_new_worker_qthread = _qthreading.new_worker_qthread


def _add_worker_data(worker: FunctionWorker, return_type, source=None):
    from napari._qt._qapp_model.injection._qprocessors import (
        _add_layer_data_to_viewer,
    )

    cb = _add_layer_data_to_viewer
    worker.signals.returned.connect(
        partial(cb, return_type=return_type, source=source)
    )


def _add_worker_data_from_tuple(
    worker: FunctionWorker, return_type, source=None
):
    from napari._qt._qapp_model.injection._qprocessors import (
        _add_layer_data_tuples_to_viewer,
    )

    cb = _add_layer_data_tuples_to_viewer
    worker.signals.returned.connect(
        partial(cb, return_type=return_type, source=source)
    )


def register_threadworker_processors():
    from functools import partial

    import magicgui

    from napari import layers, types
    from napari._app_model import get_app_model
    from napari.types import LayerDataTuple
    from napari.utils import _magicgui as _mgui

    app = get_app_model()

    for _type in (LayerDataTuple, list[LayerDataTuple]):
        t = FunctionWorker[_type]
        magicgui.register_type(t, return_callback=_mgui.add_worker_data)
        app.injection_store.register(
            processors={t: _add_worker_data_from_tuple}
        )
    for layer_name in layers.NAMES:
        _type = getattr(types, f'{layer_name.title()}Data')
        t = FunctionWorker[_type]
        magicgui.register_type(
            t,
            return_callback=partial(_mgui.add_worker_data, _from_tuple=False),
        )
        app.injection_store.register(processors={t: _add_worker_data})
