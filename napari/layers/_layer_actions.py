"""This module contains actions (functions) that operate on layers.
Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from napari import layers
from napari.layers import Image, Labels, Layer
from napari.layers._source import layer_source
from napari.layers.utils import stack_utils
from napari.layers.utils._link_layers import get_linked_layers
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components import LayerList


def _duplicate_layer(ll: LayerList, *, name: str = '') -> None:
    from copy import deepcopy

    for lay in list(ll.selection):
        data, state, type_str = lay.as_layer_data_tuple()
        state['name'] = trans._('{name} copy', name=lay.name)
        with layer_source(parent=lay):
            new = Layer.create(deepcopy(data), state, type_str)
        ll.insert(ll.index(lay) + 1, new)


def _split_stack(ll: LayerList, axis: int = 0) -> None:
    layer = ll.selection.active
    if not isinstance(layer, Image):
        return
    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    ll.remove(layer)
    ll.extend(images)
    ll.selection = set(images)  # type: ignore


def _split_rgb(ll: LayerList) -> None:
    return _split_stack(ll)


def _convert(ll: LayerList, type_: str) -> None:
    from napari.layers import Shapes

    for lay in list(ll.selection):
        idx = ll.index(lay)
        if isinstance(lay, Shapes) and type_ == 'labels':
            data = lay.to_labels()
            idx += 1
        elif (
            not np.issubdtype(lay.data.dtype, np.integer) and type_ == 'labels'
        ):
            data = lay.data.astype(int)
            idx += 1
        else:
            data = lay.data
            # int image layer to labels is fully reversible
            ll.pop(idx)
        # projection mode may not be compatible with new type,
        # we're ok with dropping it in that case
        layer_type = getattr(layers, type_.title())
        state = lay._get_base_state()
        try:
            layer_type._projectionclass(state['projection_mode'].value)
        except ValueError:
            state['projection_mode'] = 'none'
            warnings.warn(
                trans._(
                    'projection mode "{mode}" is not compatible with {type_} layers. Falling back to "none".',
                    mode=state['projection_mode'],
                    type_=type_.title(),
                    deferred=True,
                ),
                category=UserWarning,
                stacklevel=1,
            )
        new_layer = Layer.create(data, state, type_)
        ll.insert(idx, new_layer)


# TODO: currently, we have to create a thin _convert_to_x wrapper around _convert
# here for the purpose of type hinting (which partial doesn't do) ...
# so that inject_dependencies works correctly.
# however, we could conceivably add an `args` option to register_action
# that would allow us to pass additional arguments, like a partial.
def _convert_to_labels(ll: LayerList) -> None:
    return _convert(ll, 'labels')


def _convert_to_image(ll: LayerList) -> None:
    return _convert(ll, 'image')


def _merge_stack(ll: LayerList, rgb: bool = False) -> None:
    # force selection to follow LayerList ordering
    imgs = cast(list[Image], [layer for layer in ll if layer in ll.selection])
    merged = (
        stack_utils.merge_rgb(imgs)
        if rgb
        else stack_utils.images_to_stack(imgs)
    )
    for layer in imgs:
        ll.remove(layer)
    ll.append(merged)


def _toggle_visibility(ll: LayerList) -> None:
    current_visibility_state = []
    for layer in ll.selection:
        current_visibility_state.append(layer.visible)

    for visibility, layer in zip(
        current_visibility_state, ll.selection, strict=False
    ):
        if layer.visible == visibility:
            layer.visible = not visibility


def _show_selected(ll: LayerList) -> None:
    for lay in ll.selection:
        lay.visible = True


def _hide_selected(ll: LayerList) -> None:
    for lay in ll.selection:
        lay.visible = False


def _show_unselected(ll: LayerList) -> None:
    for lay in ll:
        if lay not in ll.selection:
            lay.visible = True


def _hide_unselected(ll: LayerList) -> None:
    for lay in ll:
        if lay not in ll.selection:
            lay.visible = False


def _link_selected_layers(ll: LayerList) -> None:
    ll.link_layers(ll.selection)


def _unlink_selected_layers(ll: LayerList) -> None:
    ll.unlink_layers(ll.selection)


def _select_linked_layers(ll: LayerList) -> None:
    linked_layers_in_list = [
        x for x in get_linked_layers(*ll.selection) if x in ll
    ]
    ll.selection.update(linked_layers_in_list)


def _convert_dtype(ll: LayerList, mode: npt.DTypeLike = 'int64') -> None:
    if not (layer := ll.selection.active):
        return

    if not isinstance(layer, Labels):
        raise NotImplementedError(
            trans._(
                'Data type conversion only implemented for labels',
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
                'Labeling contains values outside of the target data type range.',
                deferred=True,
            )
        )

    layer.data = layer.data.astype(np.dtype(mode))


def _project(ll: LayerList, axis: int = 0, mode: str = 'max') -> None:
    layer = ll.selection.active
    if not layer:
        return
    if not isinstance(layer, Image):
        raise NotImplementedError(
            trans._(
                'Projections are only implemented for images', deferred=True
            )
        )

    # this is not the desired behavior for coordinate-based layers
    # but the action is currently only enabled for 'image_active and ndim > 2'
    # before opening up to other layer types, this line should be updated.
    data = (getattr(np, mode)(layer.data, axis=axis, keepdims=False),)

    # Get the meta-data of the layer, but without transforms,
    # the transforms are updated bellow as projection of transforms
    # requires a bit more work than just copying them
    # (e.g., the axis of the projection should be removed).
    # It is done in `set_slice` method of `TransformChain`
    meta = {
        key: layer._get_base_state()[key]
        for key in layer._get_base_state()
        if key
        not in (
            'scale',
            'translate',
            'rotate',
            'shear',
            'affine',
            'axis_labels',
            'units',
        )
    }
    meta.update(  # sourcery skip
        {
            'name': f'{layer} {mode}-proj',
            'colormap': layer.colormap.name,
            'rendering': layer.rendering,
        }
    )
    new = Layer.create(data, meta, layer._type_string)
    # add transforms from original layer, but drop the axis of the projection
    new._transforms = layer._transforms.set_slice(
        [ax for ax in range(layer.ndim) if ax != axis]
    )

    ll.append(new)
