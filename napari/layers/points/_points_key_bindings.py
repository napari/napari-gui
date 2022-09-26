from __future__ import annotations

import numpy as np

from napari.utils.notifications import show_info

from ...layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from ...utils.translations import trans
from ._points_constants import Mode
from .points import Points


def register_points_action(description: str, repeatable: bool = False):
    return register_layer_action(Points, description, repeatable)


def register_points_mode_action(description):
    return register_layer_attr_action(Points, description, 'mode')


@Points.bind_key('Space')
def hold_to_pan_zoom(layer: Points):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        prev_selected = layer.selected_data.copy()
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode
        layer.selected_data = prev_selected
        layer._set_highlight()


@register_points_mode_action(trans._('Add points'))
def activate_points_add_mode(layer: Points):
    layer.mode = Mode.ADD


@register_points_mode_action(trans._('Select points'))
def activate_points_select_mode(layer: Points):
    layer.mode = Mode.SELECT


@register_points_mode_action(trans._('Pan/zoom'))
def activate_points_pan_zoom_mode(layer: Points):
    layer.mode = Mode.PAN_ZOOM


points_fun_to_mode = [
    (activate_points_add_mode, Mode.ADD),
    (activate_points_select_mode, Mode.SELECT),
    (activate_points_pan_zoom_mode, Mode.PAN_ZOOM),
]


@Points.bind_key('Control-C')
def copy(layer: Points):
    """Copy any selected points."""
    layer._copy_data()


@Points.bind_key('Control-V')
def paste(layer: Points):
    """Paste any copied points."""
    layer._paste_data()


def select_data(layer: Points, all_slices=True, shown=False):
    # If all points are already selected, deselect all points
    if (
        len(layer.selected_data) == len(layer.data)
        and not shown
        and all_slices
    ):
        layer.selected_data = set()
        show_info(trans._("Cleared all selections.", deferred=True))
        return

    description = trans._(
        "Different options:\nA - Select all shown points in this slice\nControl-A - Select all points in this slice\nShift-A - Select all shown points in the layer\nControl-Shift-A - Select all points in the layer",
        deferred=True,
    )
    all_points_on_slice = set(layer._slice_data(layer._slice_indices)[0])
    if shown and all_slices:
        new_selected = set(np.where(layer.shown)[0])
        insert = trans._("shown ", deferred=True)
    elif shown:
        new_selected = set(layer._indices_view[: len(layer._view_data)])
        insert = trans._("shown ", deferred=True)
    elif all_slices:
        new_selected = set(range(layer.data.shape[0]))
        insert = ""
    else:
        new_selected = all_points_on_slice
        insert = ""

    # If all visible points are already selected, deselect the visible points
    if new_selected & layer.selected_data == new_selected:
        layer.selected_data = layer.selected_data - new_selected
        select_str = trans._("Deselected", deferred=True)
    else:
        layer.selected_data = layer.selected_data | new_selected
        select_str = trans._("Selected", deferred=True)

    layer._set_highlight()
    show_info(
        trans._(
            "{select_str} all {insert}points in this slice.\n{n_slice}/{n_t_slice} selected in the slice\n{n_selected}/{n_total} selected in the layer\n{description}",
            n_selected=len(layer.selected_data),
            n_total=len(layer.data),
            n_slice=len(all_points_on_slice)
            - len(all_points_on_slice - layer.selected_data),
            n_t_slice=len(all_points_on_slice),
            insert=insert,
            description=description,
            select_str=select_str,
            deferred=True,
        )
    )


@register_points_action(
    trans._("Select all points in the current view slice."),
)
def select_all_in_slice(layer: Points):
    select_data(layer, all_slices=False, shown=False)


@register_points_action(
    trans._("Select all shown points in the current view slice."),
)
def select_all_shown_in_slice(layer: Points):
    select_data(layer, all_slices=False, shown=True)


@register_points_action(
    trans._("Select all points in the layer."),
)
def select_all_data(layer: Points):
    select_data(layer, all_slices=True, shown=False)


@register_points_action(
    trans._("Select all shown points in the layer."),
)
def select_all_shown_data(layer: Points):
    select_data(layer, all_slices=True, shown=True)


@register_points_action(trans._('Delete selected points'))
def delete_selected_points(layer: Points):
    """Delete all selected points."""
    layer.remove_selected()
