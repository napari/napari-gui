import numpy as np

from ...layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from ...utils.translations import trans
from ._labels_constants import Mode
from ._labels_utils import find_next_label
from .labels import Labels


@Labels.bind_key('Space')
def hold_to_pan_zoom(layer: Labels):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode


def register_label_action(description):
    return register_layer_action(Labels, description)


def register_label_mode_action(description):
    return register_layer_attr_action(Labels, description, 'mode')


@register_label_mode_action(trans._("Activate the paint brush"))
def activate_paint_mode(layer: Labels):
    layer.mode = Mode.PAINT


@register_label_mode_action(trans._("Activate the fill bucket"))
def activate_fill_mode(layer: Labels):
    layer.mode = Mode.FILL


@register_label_mode_action(trans._('Pan/zoom mode'))
def activate_label_pan_zoom_mode(layer: Labels):
    layer.mode = Mode.PAN_ZOOM


@register_label_mode_action(trans._('Pick mode'))
def activate_label_picker_mode(layer: Labels):
    """Activate the label picker."""
    layer.mode = Mode.PICK


@register_label_mode_action(trans._("Activate the label eraser"))
def activate_label_erase_mode(layer: Labels):
    layer.mode = Mode.ERASE


@register_label_mode_action(trans._("Activate fill contour mode"))
def activate_fill_contour_mode(layer: Labels):
    """activate fill contour mode"""
    layer.mode = Mode.FILL_CONTOUR


labels_fun_to_mode = [
    (activate_label_erase_mode, Mode.ERASE),
    (activate_paint_mode, Mode.PAINT),
    (activate_fill_mode, Mode.FILL),
    (activate_label_picker_mode, Mode.PICK),
    (activate_label_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_fill_contour_mode, Mode.FILL_CONTOUR),
]


@register_label_action(
    trans._(
        "Set the currently selected label to the largest used label plus one."
    ),
)
def new_label(layer: Labels):
    """Set the currently selected label to the largest used label plus one."""
    layer.selected_label = np.max(layer.data) + 1


@register_label_action(
    trans._("Decrease the currently selected label by one."),
)
def decrease_label_id(layer: Labels):
    layer.selected_label -= 1


@register_label_action(
    trans._("Increase the currently selected label by one."),
)
def increase_label_id(layer: Labels):
    layer.selected_label += 1


@register_label_action(
    trans._("Delete currently selected label."),
)
def delete_current_label(layer: Labels):
    curdata = layer.data.copy()
    todelete = curdata == layer.selected_label
    curdata[todelete] = 0
    layer.data = curdata
    layer.selected_label = find_next_label(layer)


@Labels.bind_key('Control-Z')
def undo(layer: Labels):
    """Undo the last paint or fill action since the view slice has changed."""
    layer.undo()


@Labels.bind_key('Control-Shift-Z')
def redo(layer: Labels):
    """Redo any previously undone actions."""
    layer.redo()
