import os
import sys
from collections import abc
from contextlib import suppress
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import DTypeLike

from napari import Viewer
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.utils.color import ColorArray
from napari.utils.events.event import WarningEmitter

skip_on_win_ci = pytest.mark.skipif(
    sys.platform.startswith('win') and os.getenv('CI', '0') != '0',
    reason='Screenshot tests are not supported on windows CI.',
)

skip_on_mac_ci = pytest.mark.skipif(
    sys.platform.startswith('darwin') and os.getenv('CI', '0') != '0',
    reason='Unsupported test on macOS CI.',
)

skip_local_popups = pytest.mark.skipif(
    not os.getenv('CI') and os.getenv('NAPARI_POPUP_TESTS', '0') == '0',
    reason='Tests requiring GUI windows are skipped locally by default.'
    ' Set NAPARI_POPUP_TESTS=1 environment variable to enable.',
)

skip_local_focus = pytest.mark.skipif(
    not os.getenv('CI') and os.getenv('NAPARI_FOCUS_TESTS', '0') == '0',
    reason='Tests requiring GUI windows focus are skipped locally by default.'
    ' Set NAPARI_FOCUS_TESTS=1 environment variable to enable.',
)

"""
The default timeout duration in seconds when waiting on tasks running in non-main threads.
The value was chosen to be consistent with `QtBot.waitSignal` and `QtBot.waitUntil`.
"""
DEFAULT_TIMEOUT_SECS: float = 5


"""
Used as pytest params for testing layer add and view functionality (Layer class, data, ndim)
"""
layer_test_data = [
    (Image, np.random.random((10, 15)), 2),
    (Image, np.random.random((10, 15, 20)), 3),
    (Image, np.random.random((5, 10, 15, 20)), 4),
    (Image, [np.random.random(s) for s in [(40, 20), (20, 10), (10, 5)]], 2),
    (Image, np.array([[1.5, np.nan], [np.inf, 2.2]]), 2),
    (Labels, np.random.randint(20, size=(10, 15)), 2),
    (Labels, np.zeros((10, 10), dtype=bool), 2),
    (Labels, np.random.randint(20, size=(6, 10, 15)), 3),
    (
        Labels,
        [np.random.randint(20, size=s) for s in [(40, 20), (20, 10), (10, 5)]],
        2,
    ),
    (Points, 20 * np.random.random((10, 2)), 2),
    (Points, 20 * np.random.random((10, 3)), 3),
    (Vectors, 20 * np.random.random((10, 2, 2)), 2),
    (Shapes, 20 * np.random.random((10, 4, 2)).astype(np.float32), 2),
    (
        Surface,
        (
            20 * np.random.random((10, 3)),
            np.random.randint(10, size=(6, 3)),
            np.random.random(10),
        ),
        3,
    ),
    (
        Tracks,
        np.column_stack(
            (np.ones(20), np.arange(20), 20 * np.random.random((20, 2)))
        ),
        3,
    ),
    (
        Tracks,
        np.column_stack(
            (np.ones(20), np.arange(20), 20 * np.random.random((20, 3)))
        ),
        4,
    ),
]

with suppress(ModuleNotFoundError):
    import tensorstore as ts

    m = ts.array(np.random.random((10, 15)))
    p = [ts.array(np.random.random(s)) for s in [(40, 20), (20, 10), (10, 5)]]
    layer_test_data.extend([(Image, m, 2), (Image, p, 2)])

classes = [Labels, Points, Vectors, Shapes, Surface, Tracks, Image]
names = [cls.__name__.lower() for cls in classes]
layer2addmethod = {
    cls: getattr(Viewer, 'add_' + name)
    for cls, name in zip(classes, names, strict=False)
}


# examples of valid tuples that might be passed to viewer._add_layer_from_data
good_layer_data = [
    (np.random.random((10, 10)),),
    (np.random.random((10, 10, 3)), {'rgb': True}),
    (np.random.randint(20, size=(10, 15)), {'opacity': 0.9}, 'labels'),
    (np.random.random((10, 2)) * 20, {'face_color': 'blue'}, 'points'),
    (np.random.random((10, 2, 2)) * 20, {}, 'vectors'),
    (np.random.random((10, 4, 2)) * 20, {'opacity': 1}, 'shapes'),
    (
        (
            np.random.random((10, 3)),
            np.random.randint(10, size=(6, 3)),
            np.random.random(10),
        ),
        {'name': 'some surface'},
        'surface',
    ),
]


class LockableData:
    """A wrapper for napari layer data that blocks read-access with a lock.

    This is useful when testing async slicing with real napari layers because
    it allows us to control when slicing tasks complete.
    """

    def __init__(self, data: LayerDataProtocol) -> None:
        self.data = data
        self.lock = RLock()

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        # LayerDataProtocol does not have ndim, but this should be equivalent.
        return len(self.data.shape)

    def __getitem__(
        self, key: Index | tuple[Index, ...] | LayerDataProtocol
    ) -> LayerDataProtocol:
        with self.lock:
            return self.data[key]

    def __len__(self) -> int:
        return len(self.data)


def add_layer_by_type(viewer, layer_type, data, visible=True):
    """
    Convenience method that maps a LayerType to its add_layer method.

    Parameters
    ----------
    layer_type : LayerTypes
        Layer type to add
    data
        The layer data to view
    """
    return layer2addmethod[layer_type](viewer, data, visible=visible)


def are_objects_equal(object1, object2):
    """
    compare two (collections of) arrays or other objects for equality. Ignores nan.
    """
    if isinstance(object1, abc.Sequence):
        items = zip(object1, object2, strict=False)
    elif isinstance(object1, dict):
        items = [(value, object2[key]) for key, value in object1.items()]
    else:
        items = [(object1, object2)]

    # equal_nan does not exist in array_equal in old numpy
    try:
        return np.all(
            [np.array_equal(a1, a2, equal_nan=True) for a1, a2 in items]
        )
    except TypeError:
        # np.array_equal fails for arrays of type `object` (e.g: strings)
        return np.all([a1 == a2 for a1, a2 in items])


def check_viewer_functioning(viewer, view=None, data=None, ndim=2):
    viewer.dims.ndisplay = 2
    # if multiscale or composite data (surface), check one by one
    assert are_objects_equal(viewer.layers[0].data, data)
    assert len(viewer.layers) == 1
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == ndim
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == ndim - 2

    # Switch to 3D rendering mode and back to 2D rendering mode
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3

    # Flip dims order displayed
    dims_order = list(range(ndim))
    viewer.dims.order = dims_order
    assert viewer.dims.order == tuple(dims_order)

    # Flip dims order including non-displayed
    dims_order[0], dims_order[-1] = dims_order[-1], dims_order[0]
    viewer.dims.order = dims_order
    assert viewer.dims.order == tuple(dims_order)

    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2


def check_view_transform_consistency(layer, viewer, transf_dict):
    """Check layer transforms have been applied to the view.

    Note this check only works for non-multiscale data.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer model.
    viewer : napari.Viewer
        Viewer, including Qt elements
    transf_dict : dict
        Dictionary of transform properties with keys referring to the name of
        the transform property (i.e. `scale`, `translate`) and the value
        corresponding to the array of property values
    """
    if layer.multiscale:
        return

    # Get an handle on visual layer:
    vis_lyr = viewer.window._qt_viewer.canvas.layer_to_visual[layer]
    # Visual layer attributes should match expected from viewer dims:
    for transf_name, transf in transf_dict.items():
        disp_dims = list(viewer.dims.displayed)  # dimensions displayed in 2D
        # values of visual layer
        vis_vals = getattr(vis_lyr, transf_name)[1::-1]

        np.testing.assert_almost_equal(vis_vals, transf[disp_dims])


def check_layer_world_data_extent(layer, extent, scale, translate):
    """Test extents after applying transforms.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer to be tested.
    extent : array, shape (2, D)
        Extent of data in layer.
    scale : array, shape (D,)
        Scale to be applied to layer.
    translate : array, shape (D,)
        Translation to be applied to layer.
    """
    np.testing.assert_almost_equal(layer.extent.data, extent)
    np.testing.assert_almost_equal(layer.extent.world, extent)

    # Apply scale transformation
    layer.scale = scale
    scaled_world_extent = np.multiply(extent, scale)
    np.testing.assert_almost_equal(layer.extent.data, extent)
    np.testing.assert_almost_equal(layer.extent.world, scaled_world_extent)

    # Apply translation transformation
    layer.translate = translate
    translated_world_extent = np.add(scaled_world_extent, translate)
    np.testing.assert_almost_equal(layer.extent.data, extent)
    np.testing.assert_almost_equal(layer.extent.world, translated_world_extent)


def assert_layer_state_equal(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Asserts that an layer state dictionary is equal to an expected one.

    This is useful because some members of state may array-like whereas others
    maybe dataframe-like, which need to be checked for equality differently.
    """
    assert actual.keys() == expected.keys()
    for name in actual:
        actual_value = actual[name]
        expected_value = expected[name]
        if isinstance(actual_value, pd.DataFrame):
            pd.testing.assert_frame_equal(actual_value, expected_value)
        else:
            np.testing.assert_equal(actual_value, expected_value)


def assert_colors_equal(actual, expected):
    """Asserts that a sequence of colors is equal to an expected one.

    This converts elements in the given sequences from color values
    recognized by ``transform_color`` to the canonical RGBA array form.

    Examples
    --------
    >>> assert_colors_equal([[1, 0, 0, 1], [0, 0, 1, 1]], ['red', 'blue'])

    >>> assert_colors_equal([[1, 0, 0, 1], [0, 0, 1, 1]], ['red', 'green'])
    Traceback (most recent call last):
    AssertionError:
    ...
    """
    actual_array = ColorArray.validate(actual)
    expected_array = ColorArray.validate(expected)
    np.testing.assert_array_equal(actual_array, expected_array)


def count_warning_events(callbacks) -> int:
    """
    Counts the number of WarningEmitter in the callback list.
    Useful to filter out deprecated events' callbacks.
    """
    return len(
        list(filter(lambda x: isinstance(x, WarningEmitter), callbacks))
    )
