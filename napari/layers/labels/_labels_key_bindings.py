import numpy as np
from app_model.types import KeyCode, KeyMod

from napari.layers.labels._labels_constants import Mode
from napari.layers.labels.labels import Labels
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.utils.translations import trans

MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 40


def register_label_action(description: str, repeatable: bool = False):
    return register_layer_action(Labels, description, repeatable)


def register_label_mode_action(description):
    return register_layer_attr_action(Labels, description, 'mode')


@register_label_mode_action(trans._('Transform'))
def activate_labels_transform_mode(layer: Labels):
    layer.mode = Mode.TRANSFORM


@register_label_mode_action(trans._('Pan/zoom'))
def activate_labels_pan_zoom_mode(layer: Labels):
    layer.mode = Mode.PAN_ZOOM


@register_label_mode_action(trans._("Activate the paint brush"))
def activate_labels_paint_mode(layer: Labels):
    layer.mode = Mode.PAINT


@register_label_mode_action(trans._("Activate the fill bucket"))
def activate_labels_fill_mode(layer: Labels):
    layer.mode = Mode.FILL


@register_label_mode_action(trans._('Pick mode'))
def activate_labels_picker_mode(layer: Labels):
    """Activate the label picker."""
    layer.mode = Mode.PICK


@register_label_mode_action(trans._("Activate the label eraser"))
def activate_labels_erase_mode(layer: Labels):
    layer.mode = Mode.ERASE


labels_fun_to_mode = [
    (activate_labels_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_labels_transform_mode, Mode.TRANSFORM),
    (activate_labels_erase_mode, Mode.ERASE),
    (activate_labels_paint_mode, Mode.PAINT),
    (activate_labels_fill_mode, Mode.FILL),
    (activate_labels_picker_mode, Mode.PICK),
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
    trans._("Swap between the selected label and the background label."),
)
def swap_selected_and_background_labels(layer: Labels):
    """Swap between the selected label and the background label."""
    layer.swap_selected_and_background_labels()


@register_label_action(
    trans._("Select the preceding label."),
)
def decrease_label_id(layer: Labels):
    if layer.predefined_labels is None:
        layer.selected_label -= 1
    else:
        labels = sorted(layer.predefined_labels.keys())
        prev_index = labels.index(layer.selected_label) - 1
        layer.selected_label = labels[max(prev_index, 0)]


@register_label_action(
    trans._("Select the next label."),
)
def increase_label_id(layer: Labels):
    if layer.predefined_labels is None:
        layer.selected_label += 1
    else:
        labels = sorted(layer.predefined_labels.keys())
        next_index = labels.index(layer.selected_label) + 1
        layer.selected_label = labels[min(next_index, len(labels) - 1)]


@register_label_action(
    trans._("Decrease the paint brush size by one."),
    repeatable=True,
)
def decrease_brush_size(layer: Labels):
    """Decrease the brush size"""
    if (
        layer.brush_size > MIN_BRUSH_SIZE
    ):  # here we should probably add a non-hard-coded
        # reference to the limit values of brush size?
        layer.brush_size -= 1


@register_label_action(
    trans._("Increase the paint brush size by one."),
    repeatable=True,
)
def increase_brush_size(layer: Labels):
    """Increase the brush size"""
    if (
        layer.brush_size < MAX_BRUSH_SIZE
    ):  # here we should probably add a non-hard-coded
        # reference to the limit values of brush size?
        layer.brush_size += 1


@register_layer_attr_action(
    Labels, trans._("Toggle preserve labels"), "preserve_labels"
)
def toggle_preserve_labels(layer: Labels):
    layer.preserve_labels = not layer.preserve_labels


@Labels.bind_key(KeyMod.CtrlCmd | KeyCode.KeyZ, overwrite=True)
def undo(layer: Labels):
    """Undo the last paint or fill action since the view slice has changed."""
    layer.undo()


@Labels.bind_key(KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ, overwrite=True)
def redo(layer: Labels):
    """Redo any previously undone actions."""
    layer.redo()
