"""Handles the logic of asynchronously slicing of multiple layers.

See the NAP for more details: https://napari.org/dev/naps/4-async-slicing.html
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterable
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from napari.layers import Layer
from napari.settings import get_settings
from napari.utils.events.event import EmitterGroup, Event

if TYPE_CHECKING:
    from napari.components import Dims

logger = logging.getLogger('napari.components._layer_slicer')


# Layers that can be asynchronously sliced must be able to make
# a slice request that can be called and will produce a slice
# response. The request and response types are coupled but will
# vary per layer type, which means that the values of the dictionary
# result of ``_slice_layers`` cannot be fixed to a single type.


class _SliceRequest(Protocol):
    id: int

    def __call__(self) -> Any: ...


@runtime_checkable
class _AsyncSliceable(Protocol):
    """The methods needed for async slicing to be supported on a layer.

    These methods are private to avoid inflating the public API of
    layers while async slicing is being developed.
    """

    def _make_slice_request(self, dims: Dims) -> _SliceRequest:
        """Makes a callable slice request that returns a response.

        This method should run quickly, as it is expected to run on the main thread.
        Slower parts of slicing should be moved into the callable request, which can
        be run off the main thread.
        The request should capture any state it needs from a layer to generate
        the response and should not modify any of that state. In combination with
        other design choices, this allows us to avoid using locks while slicing.
        """

    def _update_slice_response(self, response: Any) -> None:
        """Passes through a completed slice response.

        This method should run on the main thread and is mostly needed to update
        slice state on layers.
        """

    def _set_unloaded_slice_id(self, slice_id: int) -> None:
        """Sets the ID associated with the latest slice request.

        This is needed to support ``Layer.loaded`` in async slicing.
        This could be done at the end of ``_make_slice_request``, but was separated
        to avoid mutations in that method and to clarify responsibilities.
        """


class _LayerSlicer:
    """
    High level class to control the creation of a slice (via a slice request),
    submit it (synchronously or asynchronously) to a thread pool, and emit the
    results when complete.

    Events
    ------
    ready
        emitted after slicing is done with a dict value that maps from layer
        to slice response. Note that this may be emitted on the main or
        a non-main thread. If usage of this event relies on something happening
        on the main thread, actions should be taken to ensure that the callback
        is also executed on the main thread (e.g. by decorating the callback
        with `@ensure_main_thread`).
    """

    def __init__(self) -> None:
        """
        Attributes
        ----------
        _executor : concurrent.futures.ThreadPoolExecutor
            manager for the slicing threading
        _force_sync: bool
            if true, forces slicing to execute synchronously
        _layers_to_task : dict of tuples of layer weakrefs to futures
            task storage for cancellation logic
        _lock_layers_to_task : threading.RLock
            lock to guard against changes to `_layers_to_task` when finding,
            adding, or removing tasks.
        """
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._force_sync = not get_settings().experimental.async_
        self._layers_to_task: dict[
            tuple[weakref.ReferenceType[Layer], ...], Future
        ] = {}
        self._lock_layers_to_task = RLock()

    @contextmanager
    def force_sync(self):
        """Context manager to temporarily force slicing to be synchronous.

        This should only be used from the main thread.

        >>> layer_slicer = _LayerSlicer()
        >>> layer = Image(...)  # an async-ready layer
        >>> with layer_slice.force_sync():
        >>>     layer_slicer.submit(layers=[layer], dims=Dims())
        """
        prev = self._force_sync
        self._force_sync = True
        try:
            yield None
        finally:
            self._force_sync = prev

    def wait_until_idle(self, timeout: float | None = None) -> None:
        """Wait for all slicing tasks to complete before returning.

        Attributes
        ----------
        timeout: float or None
            (Optional) time in seconds to wait before raising TimeoutError. If set as None,
            there is no limit to the wait time. Defaults to None

        Raises
        ------
        TimeoutError: when the timeout limit has been exceeded and the task is
            not yet complete
        """
        futures = self._layers_to_task.values()
        _, not_done_futures = wait(futures, timeout=timeout)

        if len(not_done_futures) > 0:
            raise TimeoutError(
                f'Slicing {len(not_done_futures)} tasks did not complete within timeout ({timeout}s).'
            )

    def submit(
        self,
        *,
        layers: Iterable[Layer],
        dims: Dims,
        force: bool = False,
    ) -> Future[dict] | None:
        """Slices the given layers with the given dims.

        Submitting multiple layers at one generates multiple requests, but only ONE task.

        This will attempt to cancel all pending slicing tasks that can be entirely
        replaced the new ones. If multiple layers are sliced, any task that contains
        only one of those layers can safely be cancelled. If a single layer is sliced,
        it will wait for any existing tasks that include that layer AND another layer,
        In other words, it will only cancel if the new task will replace the
        slices of all the layers in the pending task.

        This should only be called from the main thread.

        Parameters
        ----------
        layers : iterable of layers
            The layers to slice.
        dims : Dims
            The dimensions values associated with the view to be sliced.
        force : bool
            True if slicing should be forced to occur, even when some cache thinks
            it already has a valid slice ready. False otherwise.

        Returns
        -------
        future of dict or none
            A future with a result that maps from a layer to an async layer
            slice response. Or none if no async slicing tasks were submitted.
        """
        logger.debug(
            '_LayerSlicer.submit: layers=%s, dims=%s, force=%s',
            layers,
            dims,
            force,
        )
        if existing_task := self._find_existing_task(layers):
            logger.debug('Cancelling task %s', id(existing_task))
            existing_task.cancel()

        # Not all layer types will initially be asynchronously sliceable.
        # The following logic gives us a way to handle those in the short
        # term as we develop, and also in the long term if there are cases
        # when we want to perform sync slicing anyway.
        requests: dict[weakref.ref, _SliceRequest] = {}
        sync_layers = []
        for layer in layers:
            # Slicing of non-visible layers is handled differently by sync
            # and async slicing. For async, we do not make request since a
            # later change to visibility triggers slicing. For sync, we want
            # to set the slice input with `Layer._slice_dims` but don't want
            # to fetch data yet (only if/when it becomes visible in the future).
            # Further development should allow us to remove this special case
            # by making the sync and async slicing code paths almost identical.
            if (
                isinstance(layer, _AsyncSliceable)
                and not self._force_sync
                and layer.visible
            ):
                logger.debug('Making async slice request for %s', layer)
                request = layer._make_slice_request(dims)
                weak_layer = weakref.ref(layer)
                requests[weak_layer] = request
                layer._set_unloaded_slice_id(request.id)
            else:
                logger.debug('Sync slicing for %s', layer)
                sync_layers.append(layer)

        # First maybe submit an async slicing task to start it ASAP.
        task = None
        if len(requests) > 0:
            logger.debug('Submitting task %s', id(task))
            task = self._executor.submit(self._slice_layers, requests)
            # Store task before adding done callback to ensure there is always
            # a task to remove in the done callback.
            with self._lock_layers_to_task:
                self._layers_to_task[tuple(requests)] = task
            task.add_done_callback(self._on_slice_done)

        # Then execute sync slicing tasks to run concurrent with async ones.
        for layer in sync_layers:
            layer._slice_dims(
                dims=dims,
                force=force,
            )

        return task

    def shutdown(self) -> None:
        """Shuts this down, preventing any new slice tasks from being submitted.

        This waits for any running tasks to finish, cancels any pending tasks,
        and disconnects any observers from this LayerSlicer's events.
        This should only be called from the main thread.
        """
        logger.debug('_LayerSlicer.shutdown')
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.events.disconnect()
        self.events.ready.disconnect()

    def _slice_layers(self, requests: dict) -> dict:
        """
        Iterates through a dictionary of request objects and call the slice
        on each individual layer. Can be called from the main or slicing thread.

        Attributes
        ----------
        requests: dict[Layer, SliceRequest]
            Dictionary of request objects to be used for constructing the slice

        Returns
        -------
        dict[Layer, SliceResponse]: which contains the results of the slice
        """
        logger.debug('_LayerSlicer._slice_layers: %s', requests)
        result = {layer: request() for layer, request in requests.items()}
        self.events.ready(value=result)
        return result

    def _on_slice_done(self, task: Future[dict]) -> None:
        """
        This is the "done_callback" which is added to each task.
        Can be called from the main or slicing thread.
        """
        logger.debug('_LayerSlicer._on_slice_done: %s', id(task))
        if not self._try_to_remove_task(task):
            logger.debug('Task not found: %s', id(task))

        if task.cancelled():
            logger.debug('Cancelled task: %s', id(task))
            return

    def _try_to_remove_task(self, task: Future[dict]) -> bool:
        """
        Attempt to remove task, return false if task not found, return true
        if task is found and removed from layers_to_task dict.

        This function provides a lock to ensure that the layers_to_task dict
        is unmodified during this process.
        """
        with self._lock_layers_to_task:
            for k_layers, v_task in self._layers_to_task.items():
                if v_task == task:
                    del self._layers_to_task[k_layers]
                    return True
        return False

    def _find_existing_task(
        self, layers: Iterable[Layer]
    ) -> Future[dict] | None:
        """Find the task associated with a list of layers. Returns the first
        task found for which the layers of the task are a subset of the input
        layers.

        This function provides a lock to ensure that the layers_to_task dict
        is unmodified during this process.
        """
        with self._lock_layers_to_task:
            layer_set = set(layers)
            for weak_task_layers, task in self._layers_to_task.items():
                task_layers = {w() for w in weak_task_layers} - {None}
                if task_layers.issubset(layer_set):
                    logger.debug('Found existing task for %s', task_layers)
                    return task
        return None
