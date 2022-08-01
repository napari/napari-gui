"""This module contains actions (functions) that operate on layers.

Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""
from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Mapping,
    Sequence,
    Union,
    cast,
)

import numpy as np
from typing_extensions import TypedDict

from ..utils._injection import inject_napari_dependencies
from ..utils.context._layerlist_context import LayerListContextKeys as LLCK
from ..utils.translations import trans
from .base.base import Layer
from .utils._link_layers import get_linked_layers

if TYPE_CHECKING:
    from ..components import LayerList
    from ..utils.context._expressions import Expr


@inject_napari_dependencies
def _duplicate_layer(ll: LayerList):
    from copy import deepcopy

    for lay in list(ll.selection):
        data, state, type_str = lay.as_layer_data_tuple()
        state["name"] = trans._('{name} copy', name=lay.name)
        new = Layer.create(deepcopy(data), state, type_str)
        ll.insert(ll.index(lay) + 1, new)


@inject_napari_dependencies
def _split_stack(ll: LayerList, axis: int = 0):
    from ..utils import stack_utils

    layer = ll.selection.active
    if not layer:
        return
    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    ll.remove(layer)
    ll.extend(images)
    ll.selection = set(images)  # type: ignore


@inject_napari_dependencies
def _project(ll: LayerList, axis: int = 0, mode='max'):
    layer = ll.selection.active
    if not layer:
        return
    if layer._type_string != 'image':
        raise NotImplementedError(
            trans._(
                "Projections are only implemented for images", deferred=True
            )
        )

    # this is not the desired behavior for coordinate-based layers
    # but the action is currently only enabled for 'image_active and ndim > 2'
    # before opening up to other layer types, this line should be updated.
    data = (getattr(np, mode)(layer.data, axis=axis, keepdims=False),)
    layer = cast('Image', layer)  # noqa: F821
    # get the meta data of the layer, but without transforms
    meta = {
        key: layer._get_base_state()[key]
        for key in layer._get_base_state()
        if key not in ('scale', 'translate', 'rotate', 'shear', 'affine')
    }
    meta.update(
        {
            'name': f'{layer} {mode}-proj',
            'colormap': layer.colormap.name,
            'rendering': layer.rendering,
        }
    )
    new = Layer.create(data, meta, layer._type_string)
    # add transforms from original layer, but drop the axis of the projection
    new._transforms = layer._transforms.set_slice(
        [ax for ax in range(0, layer.ndim) if ax != axis]
    )
    ll.append(new)


@inject_napari_dependencies
def _convert_dtype(ll: LayerList, mode='int64'):
    layer = ll.selection.active
    if not layer:
        return
    if layer._type_string != 'labels':
        raise NotImplementedError(
            trans._(
                "Data type conversion only implemented for labels",
                deferred=True,
            )
        )

    target_dtype = np.dtype(mode)
    if (
        np.min(layer.data) < np.iinfo(target_dtype).min
        or np.max(layer.data) > np.iinfo(target_dtype).max
    ):
        raise AssertionError(
            trans._(
                "Labeling contains values outside of the target data type range.",
                deferred=True,
            )
        )
    else:
        layer.data = layer.data.astype(np.dtype(mode))


def _convert(ll: LayerList, type_: str):
    from ..layers import Shapes

    for lay in list(ll.selection):
        idx = ll.index(lay)
        ll.pop(idx)
        if isinstance(lay, Shapes) and type_ == 'labels':
            data = lay.to_labels()
        else:
            data = lay.data.astype(int) if type_ == 'labels' else lay.data
        new_layer = Layer.create(data, lay._get_base_state(), type_)
        ll.insert(idx, new_layer)


@inject_napari_dependencies
def _convert_to_labels(ll: LayerList):
    return _convert(ll, 'labels')


@inject_napari_dependencies
def _convert_to_image(ll: LayerList):
    return _convert(ll, 'image')


@inject_napari_dependencies
def _merge_stack(ll: LayerList, rgb=False):
    from ..utils import stack_utils

    # force selection to follow LayerList ordering
    selection = [layer for layer in ll if layer in ll.selection]
    for layer in selection:
        ll.remove(layer)
    if rgb:
        new = stack_utils.merge_rgb(selection)
    else:
        new = stack_utils.images_to_stack(selection)
    ll.append(new)


@inject_napari_dependencies
def _toggle_visibility(ll: LayerList):
    for lay in ll.selection:
        lay.visible = not lay.visible


@inject_napari_dependencies
def _select_linked_layers(ll: LayerList):
    ll.selection.update(get_linked_layers(*ll.selection))


class _MenuItem(TypedDict):
    """An object that encapsulates an Item in a QtActionContextMenu.

    Parameters
    ----------
    description : str
        The words that appear in the menu
    enable_when : str
        An expression that evaluates to a boolean (in namespace of some
        context) and controls whether the menu item is enabled.
    show_when : str
        An expression that evaluates to a boolean (in namespace of some
        context) and controls whether the menu item is visible.
    """

    description: str
    enable_when: Union[bool, Expr]
    show_when: Union[bool, Expr]


class ContextAction(_MenuItem):
    """An object that encapsulates a QAction in a QtActionContextMenu.

    Parameters
    ----------
    action : callable
        A function that may be called if the item is selected in the menu
    """

    action: Callable


class SubMenu(_MenuItem):
    action_group: Mapping[str, ContextAction]


MenuItem = Dict[str, Union[ContextAction, SubMenu]]


# Each item in LAYER_ACTIONS will be added to the `QtActionContextMenu` created
# in _qt.containers._layer_delegate.LayerDelegate (i.e. they are options in the
# menu when you right-click on a layer in the layerlist.)
#
# variable names used in the `enable_when` and `show_when` expressions must be
# keys in the napari.components.layerlist.CONTEXT_KEYS dict.  If you need a new
# context paramameter, add a key key:value pair to the CONTEXT_KEYS dict.
#
# `action` must be a callable that accepts a single argument, an instance of
# `LayerList`.
#
# Please don't abuse "show_when".  For best UI, the menu should be roughly the
# same length all the time (just with various grayed out options).  `show_when`
# works best when there two adjacent actions with opposite `show_when`
# expressions.  See, e.g., 'link_selected_layers' and 'unlink_selected_layers'

# To add a separator, add any key with a value of _SEPARATOR


def _projdict(key) -> ContextAction:
    return {
        'description': key,
        'action': partial(_project, mode=key),
        'enable_when': (
            (LLCK.active_layer_type == "image") & LLCK.active_layer_ndim > 2
        ),
        'show_when': True,
    }


def _labeltypedict(key) -> ContextAction:
    return {
        'description': key,
        'action': partial(_convert_dtype, mode=key),
        'enable_when': (
            (LLCK.num_selected_labels_layers == LLCK.num_selected_layers)
            & (LLCK.active_layer_dtype != key)
        ),
        'show_when': True,
    }


_LAYER_ACTIONS: Sequence[MenuItem] = [
    {
        'napari:duplicate_layer': {
            'description': trans._('Duplicate Layer'),
            'action': _duplicate_layer,
            'enable_when': True,
            'show_when': True,
        },
        'napari:convert_to_labels': {
            'description': trans._('Convert to Labels'),
            'action': _convert_to_labels,
            'enable_when': (
                (
                    (LLCK.num_selected_image_layers >= 1)
                    | (LLCK.num_selected_shapes_layers >= 1)
                )
                & LLCK.all_selected_layers_same_type
            ),
            'show_when': True,
        },
        'napari:convert_to_image': {
            'description': trans._('Convert to Image'),
            'action': _convert_to_image,
            'enable_when': (
                (LLCK.num_selected_labels_layers >= 1)
                & LLCK.all_selected_layers_same_type
            ),
            'show_when': True,
        },
        'napari:toggle_visibility': {
            'description': trans._('Toggle visibility'),
            'action': _toggle_visibility,
            'enable_when': True,
            'show_when': True,
        },
    },
    # (each new dict creates a seperated section in the menu)
    {
        'napari:group:convert_type': {
            'description': trans._('Convert datatype'),
            'enable_when': (
                (LLCK.num_selected_labels_layers >= 1)
                & LLCK.all_selected_layers_same_type
            ),
            'show_when': True,
            'action_group': {
                'napari:to_int8': _labeltypedict('int8'),
                'napari:to_int16': _labeltypedict('int16'),
                'napari:to_int32': _labeltypedict('int32'),
                'napari:to_int64': _labeltypedict('int64'),
                'napari:to_uint8': _labeltypedict('uint8'),
                'napari:to_uint16': _labeltypedict('uint16'),
                'napari:to_uint32': _labeltypedict('uint32'),
                'napari:to_uint64': _labeltypedict('uint64'),
            },
        }
    },
    {
        'napari:group:projections': {
            'description': trans._('Make Projection'),
            'enable_when': (
                (LLCK.active_layer_type == "image") & LLCK.active_layer_ndim
                > 2
            ),
            'show_when': True,
            'action_group': {
                'napari:max_projection': _projdict('max'),
                'napari:min_projection': _projdict('min'),
                'napari:std_projection': _projdict('std'),
                'napari:sum_projection': _projdict('sum'),
                'napari:mean_projection': _projdict('mean'),
                'napari:median_projection': _projdict('median'),
            },
        }
    },
    {
        'napari:split_stack': {
            'description': trans._('Split Stack'),
            'action': _split_stack,
            'enable_when': LLCK.active_layer_type == "image",
            'show_when': ~LLCK.active_layer_is_rgb,
        },
        'napari:split_rgb': {
            'description': trans._('Split RGB'),
            'action': _split_stack,
            'enable_when': LLCK.active_layer_is_rgb,
            'show_when': LLCK.active_layer_is_rgb,
        },
        'napari:merge_stack': {
            'description': trans._('Merge to Stack'),
            'action': _merge_stack,
            'enable_when': (
                (LLCK.num_selected_layers > 1)
                & (LLCK.num_selected_image_layers == LLCK.num_selected_layers)
                & LLCK.all_selected_layers_same_shape
            ),
            'show_when': True,
        },
    },
    {
        'napari:link_selected_layers': {
            'description': trans._('Link Layers'),
            'action': lambda ll: ll.link_layers(ll.selection),
            'enable_when': (
                (LLCK.num_selected_layers > 1)
                & ~LLCK.num_selected_layers_linked
            ),
            'show_when': ~LLCK.num_selected_layers_linked,
        },
        'napari:unlink_selected_layers': {
            'description': trans._('Unlink Layers'),
            'action': lambda ll: ll.unlink_layers(ll.selection),
            'enable_when': LLCK.num_selected_layers_linked,
            'show_when': LLCK.num_selected_layers_linked,
        },
        'napari:select_linked_layers': {
            'description': trans._('Select Linked Layers'),
            'action': _select_linked_layers,
            'enable_when': LLCK.num_unselected_linked_layers,
            'show_when': True,
        },
    },
]
