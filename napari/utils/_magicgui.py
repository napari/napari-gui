"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

Because of headless tests the tests for this module are
in napari/_tests/test_magicgui.py
"""

from __future__ import annotations

import weakref
from functools import cache, partial
from typing import TYPE_CHECKING, Any, get_args

from napari.utils._proxies import PublicOnlyProxy

if TYPE_CHECKING:
    from concurrent.futures import Future

    from magicgui.widgets import FunctionGui
    from magicgui.widgets._bases import CategoricalWidget

    from napari._qt.qthreading import FunctionWorker
    from napari.layers import Layer
    from napari.viewer import Viewer


def add_layer_data_to_viewer(gui: FunctionGui, result: Any, return_type: type):
    """Show a magicgui result in the viewer.

    This function will be called when a magicgui-decorated function has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in ``result`` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call. For this function, this should be
        *just* the data part of the corresponding layer type.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the result as a viewer Image.

    >>> @magicgui
    ... def make_layer() -> napari.types.ImageData:
    ...     return np.random.rand(256, 256)

    """
    from napari._qt._qapp_model.injection._qprocessors import (
        _add_layer_data_to_viewer,
    )

    if result is not None and (viewer := find_viewer_ancestor(gui)):
        _add_layer_data_to_viewer(
            result,
            return_type=return_type,
            viewer=viewer,
            layer_name=gui.result_name,
            source={'widget': gui},
        )


def add_layer_data_tuples_to_viewer(gui, result, return_type):
    """Show a magicgui result in the viewer.

    This function will be called when a magicgui-decorated function has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in ``result`` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call. For this function, this should be
        *just* the data part of the corresponding layer type.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the result to the viewer

    >>> @magicgui
    ... def make_layer() -> napari.types.LayerDataTuple:
    ...     return (np.ones((10,10)), {'name': 'hi'})

    >>> @magicgui
    ... def make_layer() -> List[napari.types.LayerDataTuple]:
    ...     return [(np.ones((10,10)), {'name': 'hi'})]

    """
    from napari._qt._qapp_model.injection._qprocessors import (
        _add_layer_data_tuples_to_viewer,
    )

    if viewer := find_viewer_ancestor(gui):
        _add_layer_data_tuples_to_viewer(
            result, viewer=viewer, source={'widget': gui}
        )


def add_worker_data(
    widget, worker: FunctionWorker, return_type, _from_tuple=True
):
    """Handle a thread_worker object returned from a magicgui widget.

    This allows someone annotate their magicgui with a return type of
    `FunctionWorker[...]`, create a napari thread worker (e.g. with the
    ``@thread_worker`` decorator), then simply return the worker.  We will hook up
    the `returned` signal to the machinery to add the result of the
    long-running function to the viewer.

    Parameters
    ----------
    widget : MagicGui
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    worker : WorkerBase
        An instance of `napari._qt.qthreading.WorkerBase`, on worker.returned,
        the result will be added to the viewer.
    return_type : type
        The return annotation that was used in the decorated function.
    _from_tuple : bool, optional
        (only for internal use). True if the worker returns `LayerDataTuple`,
        False if it returns one of the `LayerData` types.

    Examples
    --------

    .. code-block:: python

        @magicgui
        def my_widget(...) -> FunctionWorker[ImageData]:

            @thread_worker
            def do_something_slowly(...) -> ImageData:
                ...

            return do_something_slowly(...)

    """

    cb = (
        add_layer_data_tuples_to_viewer
        if _from_tuple
        else add_layer_data_to_viewer
    )
    _return_type = get_args(return_type)[0]
    worker.signals.returned.connect(
        partial(cb, widget, return_type=_return_type)
    )


def add_future_data(gui, future: Future, return_type, _from_tuple=True):
    """Process a Future object from a magicgui widget.

    This function will be called when a magicgui-decorated function has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in ``result`` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    gui : FunctionGui
        The instantiated magicgui widget.  May or may not be docked in a
        dock widget.
    future : Future
        An instance of `concurrent.futures.Future` (or any third-party) object
        with the same interface, that provides `add_done_callback` and `result`
        methods.  When the future is `done()`, the `result()` will be added
        to the viewer.
    return_type : type
        The return annotation that was used in the decorated function.
    _from_tuple : bool, optional
        (only for internal use). True if the future returns `LayerDataTuple`,
        False if it returns one of the `LayerData` types.
    """
    from napari._qt._qapp_model.injection._qprocessors import _add_future_data

    if viewer := find_viewer_ancestor(gui):
        _add_future_data(
            future,
            return_type=get_args(return_type)[0],
            _from_tuple=_from_tuple,
            viewer=viewer,
            source={'widget': gui},
        )


def find_viewer_ancestor(widget) -> Viewer | None:
    """Return the closest parent Viewer of ``widget``.

    Priority is given to `Viewer` ancestors of ``widget``.
    `napari.current_viewer()` is called for Widgets without a
    Viewer ancestor.

    Parameters
    ----------
    widget : QWidget
        A widget

    Returns
    -------
    viewer : napari.Viewer or None
        Viewer ancestor if it exists, else `napari.current_viewer()`
    """
    from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget

    # magicgui v0.2.0 widgets are no longer QWidget subclasses, but the native
    # widget is available at widget.native
    if hasattr(widget, 'native') and hasattr(widget.native, 'parent'):
        parent = widget.native.parent()
    else:
        parent = widget.parent()
    from napari.viewer import current_viewer

    while parent:
        if hasattr(parent, '_qt_viewer'):  # QMainWindow
            return parent._qt_viewer.viewer
        if isinstance(parent, QtViewerDockWidget):  # DockWidget
            qt_viewer = parent._ref_qt_viewer()
            if qt_viewer is not None:
                return qt_viewer.viewer
            return current_viewer()
        parent = parent.parent()
    return current_viewer()


def proxy_viewer_ancestor(widget) -> PublicOnlyProxy[Viewer] | None:
    if viewer := find_viewer_ancestor(widget):
        return PublicOnlyProxy(viewer)
    return None


def get_layers(gui: CategoricalWidget) -> list[Layer]:
    """Retrieve layers matching gui.annotation, from the Viewer the gui is in.

    Parameters
    ----------
    gui : magicgui.widgets.Widget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.

    Returns
    -------
    tuple
        Tuple of layers of type ``gui.annotation``

    Examples
    --------
    This allows the user to do this, and get a dropdown box in their GUI
    that shows the available image layers.

    >>> @magicgui
    ... def get_layer_mean(layer: napari.layers.Image) -> float:
    ...     return layer.data.mean()

    """
    if viewer := find_viewer_ancestor(gui.native):
        return [x for x in viewer.layers if isinstance(x, gui.annotation)]
    return []


def get_layers_data(gui: CategoricalWidget) -> list[tuple[str, Any]]:
    """Retrieve layers matching gui.annotation, from the Viewer the gui is in.

    As opposed to `get_layers`, this function returns just `layer.data` rather
    than the full layer object.

    Parameters
    ----------
    gui : magicgui.widgets.Widget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.

    Returns
    -------
    tuple
        Tuple of layer.data from layers of type ``gui.annotation``

    Examples
    --------
    This allows the user to do this, and get a dropdown box in their GUI
    that shows the available image layers, but just get the data from the image
    as function input

    >>> @magicgui
    ... def get_layer_mean(data: napari.types.ImageData) -> float:
    ...     return data.mean()

    """
    from napari import layers

    if not (viewer := find_viewer_ancestor(gui.native)):
        return ()

    layer_type_name = gui.annotation.__name__.replace('Data', '').title()
    layer_type = getattr(layers, layer_type_name)
    choices = []
    for layer in [x for x in viewer.layers if isinstance(x, layer_type)]:
        choice_key = f'{layer.name} (data)'
        choices.append((choice_key, layer.data))
        layer.events.data.connect(_make_choice_data_setter(gui, choice_key))

    return choices


@cache
def _make_choice_data_setter(gui: CategoricalWidget, choice_name: str):
    """Return a function that sets the ``data`` for ``choice_name`` in ``gui``.

    Note, using lru_cache here so that the **same** function object is returned
    if you call this twice for the same widget/choice_name combination. This is
    so that when we connect it above in `layer.events.data.connect()`, it will
    only get connected once (because ``.connect()`` will not add a specific callback
    more than once)
    """
    gui_ref = weakref.ref(gui)

    def setter(event):
        _gui = gui_ref()
        if _gui is not None:
            _gui.set_choice(choice_name, event.value)

    return setter


def add_layer_to_viewer(gui, result: Any, return_type: type[Layer]) -> None:
    """Show a magicgui result in the viewer.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the resulting layer to the viewer.

    >>> @magicgui
    ... def make_layer() -> napari.layers.Image:
    ...     return napari.layers.Image(np.random.rand(64, 64))

    """
    add_layers_to_viewer(gui, [result], list[return_type])


def add_layers_to_viewer(
    gui: FunctionGui[Any], result: Any, return_type: type[list[Layer]]
) -> None:
    """Show a magicgui result in the viewer.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the resulting layer to the viewer.

    >>> @magicgui
    ... def make_layer() -> List[napari.layers.Layer]:
    ...     return napari.layers.Image(np.random.rand(64, 64))

    """
    from napari._qt._qapp_model.injection._qprocessors import (
        _add_layer_to_viewer,
    )

    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

    for item in result:
        if item is not None:
            _add_layer_to_viewer(item, viewer=viewer, source={'widget': gui})
