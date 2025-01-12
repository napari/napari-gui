from __future__ import annotations

import contextlib
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Optional
from weakref import ref

from app_model.expressions import ContextKey

from napari._app_model.context._context_keys import ContextNamespace
from napari.utils._dtype import normalize_dtype
from napari.utils.translations import trans

if TYPE_CHECKING:
    from weakref import ReferenceType

    from numpy.typing import DTypeLike

    from napari.components.layerlist import LayerList
    from napari.layers import Layer
    from napari.utils.events import Selection

    LayerSel = Selection[Layer]


def _len(layers: LayerSel | LayerList) -> int:
    return len(layers)


class LayerListContextKeys(ContextNamespace['Layer']):
    """These are the available context keys relating to a LayerList.

    Consists of a default value, a description, and a function to retrieve the
    current value from `layers`.
    """

    num_layers = ContextKey(
        0,
        trans._('Number of layers.'),
        _len,
    )


def _all_linked(s: LayerSel) -> bool:
    from napari.layers.utils._link_layers import layer_is_linked

    return bool(s and all(layer_is_linked(x) for x in s))


def _n_unselected_links(s: LayerSel) -> int:
    from napari.layers.utils._link_layers import get_linked_layers

    return len(get_linked_layers(*s) - s)


def _is_rgb(s: LayerSel) -> bool:
    return getattr(s.active, 'rgb', False)


def _only_img(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'image' for x in s))


def _n_selected_imgs(s: LayerSel) -> int:
    return sum(x._type_string == 'image' for x in s)


def _only_labels(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'labels' for x in s))


def _n_selected_labels(s: LayerSel) -> int:
    return sum(x._type_string == 'labels' for x in s)


def _only_points(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'points' for x in s))


def _n_selected_points(s: LayerSel) -> int:
    return sum(x._type_string == 'points' for x in s)


def _only_shapes(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'shapes' for x in s))


def _n_selected_shapes(s: LayerSel) -> int:
    return sum(x._type_string == 'shapes' for x in s)


def _only_surface(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'surface' for x in s))


def _n_selected_surfaces(s: LayerSel) -> int:
    return sum(x._type_string == 'surface' for x in s)


def _only_vectors(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'vectors' for x in s))


def _n_selected_vectors(s: LayerSel) -> int:
    return sum(x._type_string == 'vectors' for x in s)


def _only_tracks(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == 'tracks' for x in s))


def _n_selected_tracks(s: LayerSel) -> int:
    return sum(x._type_string == 'tracks' for x in s)


def _active_type(s: LayerSel) -> str | None:
    return s.active._type_string if s.active else None


def _active_ndim(s: LayerSel) -> int | None:
    return getattr(s.active.data, 'ndim', None) if s.active else None


def _active_shape(s: LayerSel) -> tuple[int, ...] | None:
    return getattr(s.active.data, 'shape', None) if s.active else None


def _same_shape(s: LayerSel) -> bool:
    """Return true when all given layers have the same shape.

    Notes
    -----
    The cast to tuple() is needed because some array libraries, specifically
    Apple's mlx [1]_, return a list, which is not hashable and thus causes the
    set (``{}``) to fail.

    The Data APIs Array spec specifies that ``.shape`` should be a tuple, or,
    if a custom type, it should be an immutable type [2]_, so in time, the cast
    to tuple could be removed, once all major libraries support the spec.

    References
    ----------
    .. [1] https://github.com/ml-explore/mlx
    .. [2] https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.shape.html
    """
    return len({tuple(getattr(x.data, 'shape', ())) for x in s}) == 1


def _active_dtype(s: LayerSel) -> DTypeLike:
    dtype = None
    if s.active:
        with contextlib.suppress(AttributeError):
            dtype = normalize_dtype(s.active.data.dtype).__name__
    return dtype


def _same_type(s: LayerSel) -> bool:
    return len({x._type_string for x in s}) == 1


def _active_is_image_3d(s: LayerSel) -> bool:
    _activ_ndim = _active_ndim(s)
    return (
        _active_type(s) == 'image'
        and _activ_ndim is not None
        and (_activ_ndim > 3 or ((_activ_ndim) > 2 and not _is_rgb(s)))
    )


def _shapes_selection_check(s: ReferenceType[LayerSel]) -> bool:
    s_ = s()
    if s_ is None:
        return False
    return any(x._type_string == 'shapes' and not len(x.data) for x in s_)


def _empty_shapes_layer_selected(s: LayerSel) -> Callable[[], bool]:
    check_fun = partial(_shapes_selection_check, ref(s))
    return check_fun


class LayerListSelectionContextKeys(ContextNamespace['LayerSel']):
    """Available context keys relating to the selection in a LayerList.

    Consists of a default value, a description, and a function to retrieve the
    current value from `layers.selection`.
    """

    num_selected_layers = ContextKey(
        0,
        trans._('Number of currently selected layers.'),
        _len,
    )
    num_selected_layers_linked = ContextKey(
        False,
        trans._('True when all selected layers are linked.'),
        _all_linked,
    )
    num_unselected_linked_layers = ContextKey(
        0,
        trans._('Number of unselected layers linked to selected layer(s).'),
        _n_unselected_links,
    )
    active_layer_is_rgb = ContextKey(
        False,
        trans._('True when the active layer is RGB.'),
        _is_rgb,
    )
    active_layer_type = ContextKey['LayerSel', Optional[str]](
        None,
        trans._(
            'Lowercase name of active layer type, or None of none active.'
        ),
        _active_type,
    )
    # TODO: try to reduce these `num_selected_x_layers` to a single set of strings
    # or something... however, this would require that our context expressions
    # support Sets, tuples, lists, etc...  which they currently do not.
    num_selected_image_layers = ContextKey(
        0,
        trans._('Number of selected image layers.'),
        _n_selected_imgs,
    )
    num_selected_labels_layers = ContextKey(
        0,
        trans._('Number of selected labels layers.'),
        _n_selected_labels,
    )
    num_selected_points_layers = ContextKey(
        0,
        trans._('Number of selected points layers.'),
        _n_selected_points,
    )
    num_selected_shapes_layers = ContextKey(
        0,
        trans._('Number of selected shapes layers.'),
        _n_selected_shapes,
    )
    num_selected_surface_layers = ContextKey(
        0,
        trans._('Number of selected surface layers.'),
        _n_selected_surfaces,
    )
    num_selected_vectors_layers = ContextKey(
        0,
        trans._('Number of selected vectors layers.'),
        _n_selected_vectors,
    )
    num_selected_tracks_layers = ContextKey(
        0,
        trans._('Number of selected tracks layers.'),
        _n_selected_tracks,
    )
    active_layer_ndim = ContextKey['LayerSel', Optional[int]](
        None,
        trans._(
            'Number of dimensions in the active layer, or `None` if nothing is active.'
        ),
        _active_ndim,
    )
    active_layer_shape = ContextKey['LayerSel', Optional[tuple[int, ...]]](
        (),
        trans._('Shape of the active layer, or `None` if nothing is active.'),
        _active_shape,
    )
    active_layer_is_image_3d = ContextKey(
        False,
        trans._('True when the active layer is a 3D image.'),
        _active_is_image_3d,
    )
    active_layer_dtype = ContextKey(
        None,
        trans._('Dtype of the active layer, or `None` if nothing is active.'),
        _active_dtype,
    )
    all_selected_layers_same_shape = ContextKey(
        False,
        trans._('True when all selected layers have the same shape.'),
        _same_shape,
    )
    all_selected_layers_same_type = ContextKey(
        False,
        trans._('True when all selected layers are of the same type.'),
        _same_type,
    )
    all_selected_layers_labels = ContextKey(
        False,
        trans._('True when all selected layers are labels.'),
        _only_labels,
    )
    selected_empty_shapes_layer = ContextKey(
        False,
        trans._('True when there is a shapes layer without data selected.'),
        _empty_shapes_layer_selected,
    )
