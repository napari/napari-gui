# mypy: disable-error-code=empty-body
"""
All napari hook specifications for pluggable functionality are defined here.

A *hook specification* is a function signature (with documentation) that
declares an API that plugin developers must adhere to when providing hook
implementations.  *Hook implementations* provided by plugins (and internally by
napari) will then be invoked in various places throughout the code base.

When implementing a hook specification, pay particular attention to the number
and types of the arguments in the specification signature, as well as the
expected return type.

To allow for hook specifications to evolve over the lifetime of napari,
hook implementations may accept *fewer* arguments than defined in the
specification. (This allows for extending existing hook arguments without
breaking existing implementations). However, implementations must not require
*more* arguments than defined in the spec.

For more general background on the plugin hook calling mechanism, see the
`napari-plugin-manager documentation
<https://napari-plugin-engine.readthedocs.io/en/latest/>`_.

.. NOTE::
    Hook specifications are a feature borrowed from `pluggy
    <https://pluggy.readthedocs.io/en/latest/#specs>`_. In the `pluggy
    documentation <https://pluggy.readthedocs.io/en/latest/>`_, hook
    specification marker instances are named ``hookspec`` by convention, and
    hook implementation marker instances are named ``hookimpl``.  The
    convention in napari is to name them more explicitly:
    ``napari_hook_specification`` and ``napari_hook_implementation``,
    respectively.
"""

# These hook specifications also serve as the API reference for plugin
# developers, so comprehensive documentation with complete type annotations is
# imperative!
from __future__ import annotations

from types import FunctionType
from typing import Any

from napari_plugin_engine import napari_hook_specification

from napari.types import (
    AugmentedWidget,
    ReaderFunction,
    SampleData,
    SampleDict,
    WriterFunction,
)

# -------------------------------------------------------------------------- #
#                                 IO Hooks                                   #
# -------------------------------------------------------------------------- #


@napari_hook_specification(historic=True)
def napari_provide_sample_data() -> dict[str, SampleData | SampleDict]:
    """Provide sample data.

    Plugins may implement this hook to provide sample data for use in napari.
    Sample data is accessible in the `File > Open Sample` menu, or
    programmatically, with :meth:`napari.Viewer.open_sample`.

    Plugins implementing this hook specification must return a ``dict``, where
    each key is a `sample_key` (the string that will appear in the
    `Open Sample` menu), and the value is either a string, or
    a callable that returns an iterable of ``LayerData`` tuples, where each
    tuple is a 1-, 2-, or 3-tuple of ``(data,)``, ``(data, meta)``, or ``(data,
    meta, layer_type)`` (thus, an individual sample-loader may provide multiple
    layers).  If the value is a string, it will be opened with
    :meth:`napari.Viewer.open`.

    Examples
    --------
    Here's a minimal example of a plugin that provides three samples:

        1. random data from numpy
        2. a random image pulled from the internet
        3. random data from numpy, provided as a dict with the keys:
            * 'display_name': a string that will show in the menu (by default,
                the `sample_key` will be shown)
            * 'data': a string or callable, as in 1/2.

    .. code-block:: python

        import numpy as np
        from napari_plugin_engine import napari_hook_implementation

        def _generate_random_data(shape=(512, 512)):
            data = np.random.rand(*shape)
            return [(data, {'name': 'random data'})]

        @napari_hook_implementation
        def napari_provide_sample_data():
            return {
                'random data': _generate_random_data,
                'random image': 'https://picsum.photos/1024',
                'sample_key': {
                    'display_name': 'Some Random Data (512 x 512)'
                    'data': _generate_random_data,
                }
            }

    Returns
    -------
    Dict[ str, Union[str, Callable[..., Iterable[LayerData]]] ]
        A mapping of `sample_key` to `data_loader`
    """


@napari_hook_specification(firstresult=True)
def napari_get_reader(path: str | list[str]) -> ReaderFunction | None:
    """Return a function capable of loading ``path`` into napari, or ``None``.

    This is the primary "**reader plugin**" function.  It accepts a path or
    list of paths, and returns a list of data to be added to the ``Viewer``.
    The function may return ``[(None, )]`` to indicate that the file was read
    successfully, but did not contain any data.

    The main place this hook is used is in :func:`Viewer.open()
    <napari.components.viewer_model.ViewerModel.open>`, via the
    :func:`~napari.plugins.io.read_data_with_plugins` function.

    It will also be called on ``File -> Open...`` or when a user drops a file
    or folder onto the viewer. This function must execute **quickly**, and
    should return ``None`` if the filepath is of an unrecognized format for
    this reader plugin.  If ``path`` is determined to be recognized format,
    this function should return a *new* function that accepts the same filepath
    (or list of paths), and returns a list of ``LayerData`` tuples, where each
    tuple is a 1-, 2-, or 3-tuple of ``(data,)``, ``(data, meta)``, or ``(data,
    meta, layer_type)``.

    ``napari`` will then use each tuple in the returned list to generate a new
    layer in the viewer using the :func:`Viewer._add_layer_from_data()
    <napari.components.viewer_model.ViewerModel._add_layer_from_data>`
    method.  The first, (optional) second, and (optional) third items in each
    tuple in the returned layer_data list, therefore correspond to the
    ``data``, ``meta``, and ``layer_type`` arguments of the
    :func:`Viewer._add_layer_from_data()
    <napari.components.viewer_model.ViewerModel._add_layer_from_data>`
    method, respectively.

    .. important::

       ``path`` may be either a ``str`` or a ``list`` of ``str``.  If a
       ``list``, then each path in the list can be assumed to be one part of a
       larger multi-dimensional stack (for instance: a list of 2D image files
       that should be stacked along a third axis). Implementations should do
       their own checking for ``list`` or ``str``, and handle each case as
       desired.

    Parameters
    ----------
    path : str or list of str
        Path to file, directory, or resource (like a URL), or a list of paths.

    Returns
    -------
    Callable or None
        A function that accepts the path, and returns a list of ``layer_data``,
        where ``layer_data`` is one of ``(data,)``, ``(data, meta)``, or
        ``(data, meta, layer_type)``.
        If unable to read the path, must return ``None`` (not ``False``!).
    """


@napari_hook_specification(firstresult=True)
def napari_get_writer(
    path: str, layer_types: list[str]
) -> WriterFunction | None:
    """Return function capable of writing napari layer data to ``path``.

    This function will be called whenever the user attempts to save multiple
    layers (e.g. via ``File -> Save Layers``, or
    :func:`~napari.plugins.io.save_layers`).
    This function must execute **quickly**, and should return ``None`` if
    ``path`` has an unrecognized extension for the reader plugin or the list of
    layer types are incompatible with what the plugin can write. If ``path`` is
    a recognized format, this function should return a *function* that accepts
    the same ``path``, and a list of tuples containing the data for each layer
    being saved in the form of ``(Layer.data, Layer._get_state(),
    Layer._type_string)``. The writer function should return a list of strings
    (the actual filepath(s) that were written).

    .. important::

        It is up to plugins to inspect and obey any extension in ``path``
        (and return ``None`` if it is an unsupported extension).

    An example function signature for a ``WriterFunction`` that might be
    returned by this hook specification is as follows:

    .. code-block:: python

        def writer_function(
            path: str, layer_data: List[Tuple[Any, Dict, str]]
        ) -> List[str]:
            ...

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).  Any extensions in
        the path should be examined and obeyed.  (i.e. if the plugin is
        incapable of returning a requested extension, it should return
        ``None``).
    layer_types : list of str
        List of layer types (e.g. "image", "labels") that will be provided to
        the writer function.

    Returns
    -------
    Callable or None
        A function that accepts the path, a list of layer_data (where
        layer_data is ``(data, meta, layer_type)``). If unable to write to the
        path or write the layer_data, must return ``None`` (not ``False``).
    """


@napari_hook_specification(firstresult=True)
def napari_write_image(path: str, data: Any, meta: dict) -> str | None:
    """Write image data and metadata into a path.

    It is the responsibility of the implementation to check any extension on
    ``path`` and return ``None`` if it is an unsupported extension.  If
    ``path`` has no extension, implementations may append their preferred
    extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If meta['rgb'] is ``True`` then the
        data should be interpreted as RGB or RGBA. If meta['multiscale'] is
        True, then the data should be interpreted as a multiscale image.
    meta : dict
        Image metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """


@napari_hook_specification(firstresult=True)
def napari_write_labels(path: str, data: Any, meta: dict) -> str | None:
    """Write labels data and metadata into a path.

    It is the responsibility of the implementation to check any extension on
    ``path`` and return ``None`` if it is an unsupported extension.  If
    ``path`` has no extension, implementations may append their preferred
    extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Integer valued label data. Can be N dimensional. Every pixel contains
        an integer ID corresponding to the region it belongs to. The label 0 is
        rendered as transparent. If a list and arrays are decreasing in shape
        then the data is from a multiscale image.
    meta : dict
        Labels metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """


@napari_hook_specification(firstresult=True)
def napari_write_points(path: str, data: Any, meta: dict) -> str | None:
    """Write points data and metadata into a path.

    It is the responsibility of the implementation to check any extension on
    ``path`` and return ``None`` if it is an unsupported extension.  If
    ``path`` has no extension, implementations may append their preferred
    extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array (N, D)
        Coordinates for N points in D dimensions.
    meta : dict
        Points metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """


@napari_hook_specification(firstresult=True)
def napari_write_shapes(path: str, data: Any, meta: dict) -> str | None:
    """Write shapes data and metadata into a path.

    It is the responsibility of the implementation to check any extension on
    ``path`` and return ``None`` if it is an unsupported extension.  If
    ``path`` has no extension, implementations may append their preferred
    extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : list
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions.
    meta : dict
        Shapes metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """


@napari_hook_specification(firstresult=True)
def napari_write_surface(path: str, data: Any, meta: dict) -> str | None:
    """Write surface data and metadata into a path.

    It is the responsibility of the implementation to check any extension on
    ``path`` and return ``None`` if it is an unsupported extension.  If
    ``path`` has no extension, implementations may append their preferred
    extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles. The second is an (M, 3) array of int of indices
        of the mesh triangles. The third element is the (K0, ..., KL, N)
        array of values used to color vertices where the additional L
        dimensions are used to color the same mesh with different values.
    meta : dict
        Surface metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """


@napari_hook_specification(firstresult=True)
def napari_write_vectors(path: str, data: Any, meta: dict) -> str | None:
    """Write vectors data and metadata into a path.

    It is the responsibility of the implementation to check any extension on
    ``path`` and return ``None`` if it is an unsupported extension.  If
    ``path`` has no extension, implementations may append their preferred
    extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : (N, 2, D) array
        The start point and projections of N vectors in D dimensions.
    meta : dict
        Vectors metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """


# -------------------------------------------------------------------------- #
#                                 GUI Hooks                                  #
# -------------------------------------------------------------------------- #


@napari_hook_specification(historic=True)
def napari_experimental_provide_function() -> (
    FunctionType | list[FunctionType]
):
    """Provide function(s) that can be passed to magicgui.

    This hook specification is marked as experimental as the API or how the
    returned value is handled may change here more frequently then the
    rest of the codebase.

    Returns
    -------
    function(s) : FunctionType or list of FunctionType
        Implementations should provide either a single function, or a list of
        functions. Note that this does not preclude specifying multiple
        separate implementations in the same module or class.
        The functions should have Python type annotations so that
        `magicgui <https://napari.org/magicgui>`_ can generate a widget from
        them.

    Examples
    --------
    >>> from napari.types import ImageData, LayerDataTuple
    >>>
    >>> def my_function(image : ImageData) -> LayerDataTuple:
    >>>     # process the image
    >>>     result = -image
    >>>     # return it + some layer properties
    >>>     return result, {'colormap':'turbo'}
    >>>
    >>> @napari_hook_implementation
    >>> def napari_experimental_provide_function():
    >>>     return my_function
    """


@napari_hook_specification(historic=True)
def napari_experimental_provide_dock_widget() -> (
    AugmentedWidget | list[AugmentedWidget]
):
    """Provide functions that return widgets to be docked in the viewer.

    This hook specification is marked as experimental as the API or how the
    returned value is handled may change here more frequently then the
    rest of the codebase.

    Returns
    -------
    result : callable or tuple or list of callables or list of tuples
        A "callable" in this context is a class or function that, when
        called, returns an instance of either a
        :class:`~qtpy.QtWidgets.QWidget` or a
        :class:`~magicgui.widgets.FunctionGui`.

        Implementations of this hook specification must return a callable, or a
        tuple of ``(callable, dict)``, where the dict contains keyword
        arguments for :meth:`napari.qt.Window.add_dock_widget`. (note, however,
        that ``shortcut=`` keyword is not yet supported).

        Implementations may also return a list, in which each item must be a
        callable or ``(callable, dict)`` tuple. Note that this does not
        preclude specifying multiple separate implementations in the same module
        or class.

    Examples
    --------
    An example with a QtWidget:

    >>> from qtpy.QtWidgets import QWidget
    >>> from napari_plugin_engine import napari_hook_implementation
    >>>
    >>> class MyWidget(QWidget):
    ...     def __init__(self, napari_viewer):
    ...         self.viewer = napari_viewer
    ...         super().__init__()
    ...
    ...         # initialize layout
    ...         layout = QGridLayout()
    ...
    ...         # add a button
    ...         btn = QPushButton('Click me!', self)
    ...         def trigger():
    ...             print("napari has", len(napari_viewer.layers), "layers")
    ...         btn.clicked.connect(trigger)
    ...         layout.addWidget(btn)
    ...
    ...         # activate layout
    ...         self.setLayout(layout)
    >>>
    >>> @napari_hook_implementation
    >>> def napari_experimental_provide_dock_widget():
    ...     return MyWidget

    An example using magicgui:

    >>> from magicgui import magic_factory
    >>> from napari_plugin_engine import napari_hook_implementation
    >>>
    >>> @magic_factory(auto_call=True, threshold={'max': 2 ** 16})
    >>> def threshold(
    ...     data: 'napari.types.ImageData', threshold: int
    ... ) -> 'napari.types.LabelsData':
    ...     return (data > threshold).astype(int)
    >>>
    >>> @napari_hook_implementation
    >>> def napari_experimental_provide_dock_widget():
    ...     return threshold
    """


@napari_hook_specification(historic=True)
def napari_experimental_provide_theme() -> (
    dict[str, dict[str, str | tuple | list]]
):
    """Provide GUI with a set of colors used through napari. This hook allows you to
    provide additional color schemes so you can accomplish your desired styling.

    Themes are provided as `dict` with several required fields and correctly formatted
    color values. Colors can be specified using color names (e.g. ``white``), hex color
    (e.g. ``#ff5733``), rgb color in 0-255 range (e.g. ``rgb(255, 0, 127)`` or as
    3- or 4-element tuples or lists (e.g. ``(255, 0, 127)``. The `Theme` model will
    automatically handle the conversion.

    See :class:`~napari.utils.theme.Theme` for more detail of what are the required keys.

    Returns
    -------
    themes : Dict[str, Dict[str, Union[str, Tuple, List]]
        Sequence of dictionaries containing new color schemes to be used by napari.
        You can replace existing themes by using the same names.

    Examples
    --------
    >>> def get_new_theme() -> Dict[str, Dict[str, Union[str, Tuple, List]]:
    ...     # specify theme(s) that should be added to napari
    ...     themes = {
    ...         "super_dark": {
    ...             "name": "super_dark",
    ...             "background": "rgb(12, 12, 12)",
    ...             "foreground": "rgb(65, 72, 81)",
    ...             "primary": "rgb(90, 98, 108)",
    ...             "secondary": "rgb(134, 142, 147)",
    ...             "highlight": "rgb(106, 115, 128)",
    ...             "text": "rgb(240, 241, 242)",
    ...             "icon": "rgb(209, 210, 212)",
    ...             "warning": "rgb(153, 18, 31)",
    ...             "current": "rgb(0, 122, 204)",
    ...             "syntax_style": "native",
    ...             "console": "rgb(0, 0, 0)",
    ...             "canvas": "black",
    ...         }
    ...     }
    ...     return themes
    >>>
    >>> @napari_hook_implementation
    >>> def napari_experimental_provide_theme():
    ...     return get_new_theme()

    """
