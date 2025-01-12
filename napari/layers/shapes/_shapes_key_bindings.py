from collections.abc import Callable, Generator

import numpy as np
from app_model.types import KeyCode

from napari.layers.shapes._shapes_constants import Box, Mode
from napari.layers.shapes._shapes_mouse_bindings import (
    _move_active_element_under_cursor,
)
from napari.layers.shapes.shapes import Shapes
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.utils.translations import trans


@Shapes.bind_key(KeyCode.Shift, overwrite=True)
def hold_to_lock_aspect_ratio(layer: Shapes) -> Generator[None, None, None]:
    """Hold to lock aspect ratio when resizing a shape."""
    # on key press
    layer._fixed_aspect = True
    box = layer._selected_box
    if box is not None:
        size = box[Box.BOTTOM_RIGHT] - box[Box.TOP_LEFT]
        if not np.any(size == np.zeros(2)):
            layer._aspect_ratio = abs(size[1] / size[0])
        else:
            layer._aspect_ratio = 1
    else:
        layer._aspect_ratio = 1
    if layer._is_moving:
        assert layer._moving_coordinates is not None, layer
        _move_active_element_under_cursor(layer, layer._moving_coordinates)

    yield

    # on key release
    layer._fixed_aspect = False


def register_shapes_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Shapes, description, repeatable)


def register_shapes_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Shapes, description, 'mode')


@register_shapes_mode_action(trans._('Transform'))
def activate_shapes_transform_mode(layer: Shapes) -> None:
    layer.mode = Mode.TRANSFORM


@register_shapes_mode_action(trans._('Pan/zoom'))
def activate_shapes_pan_zoom_mode(layer: Shapes) -> None:
    layer.mode = Mode.PAN_ZOOM


@register_shapes_mode_action(trans._('Add rectangles'))
def activate_add_rectangle_mode(layer: Shapes) -> None:
    """Activate add rectangle tool."""
    layer.mode = Mode.ADD_RECTANGLE


@register_shapes_mode_action(trans._('Add ellipses'))
def activate_add_ellipse_mode(layer: Shapes) -> None:
    """Activate add ellipse tool."""
    layer.mode = Mode.ADD_ELLIPSE


@register_shapes_mode_action(trans._('Add lines'))
def activate_add_line_mode(layer: Shapes) -> None:
    """Activate add line tool."""
    layer.mode = Mode.ADD_LINE


@register_shapes_mode_action(trans._('Add path'))
def activate_add_path_mode(layer: Shapes) -> None:
    """Activate add path tool."""
    layer.mode = Mode.ADD_PATH


@register_shapes_mode_action(trans._('Add polygons'))
def activate_add_polygon_mode(layer: Shapes) -> None:
    """Activate add polygon tool."""
    layer.mode = Mode.ADD_POLYGON


@register_shapes_mode_action(trans._('Add polygons lasso'))
def activate_add_polygon_lasso_mode(layer: Shapes) -> None:
    """Activate add polygon tool."""
    layer.mode = Mode.ADD_POLYGON_LASSO


@register_shapes_mode_action(trans._('Select vertices'))
def activate_direct_mode(layer: Shapes) -> None:
    """Activate vertex selection tool."""
    layer.mode = Mode.DIRECT


@register_shapes_mode_action(trans._('Select shapes'))
def activate_select_mode(layer: Shapes) -> None:
    """Activate shape selection tool."""
    layer.mode = Mode.SELECT


@register_shapes_mode_action(trans._('Insert vertex'))
def activate_vertex_insert_mode(layer: Shapes) -> None:
    """Activate vertex insertion tool."""
    layer.mode = Mode.VERTEX_INSERT


@register_shapes_mode_action(trans._('Remove vertex'))
def activate_vertex_remove_mode(layer: Shapes) -> None:
    """Activate vertex deletion tool."""
    layer.mode = Mode.VERTEX_REMOVE


shapes_fun_to_mode = [
    (activate_shapes_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_shapes_transform_mode, Mode.TRANSFORM),
    (activate_add_rectangle_mode, Mode.ADD_RECTANGLE),
    (activate_add_ellipse_mode, Mode.ADD_ELLIPSE),
    (activate_add_line_mode, Mode.ADD_LINE),
    (activate_add_path_mode, Mode.ADD_PATH),
    (activate_add_polygon_mode, Mode.ADD_POLYGON),
    (activate_add_polygon_lasso_mode, Mode.ADD_POLYGON_LASSO),
    (activate_direct_mode, Mode.DIRECT),
    (activate_select_mode, Mode.SELECT),
    (activate_vertex_insert_mode, Mode.VERTEX_INSERT),
    (activate_vertex_remove_mode, Mode.VERTEX_REMOVE),
]


@register_shapes_action(trans._('Copy any selected shapes'))
def copy_selected_shapes(layer: Shapes) -> None:
    """Copy any selected shapes."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._copy_data()


@register_shapes_action(trans._('Paste any copied shapes'))
def paste_shape(layer: Shapes) -> None:
    """Paste any copied shapes."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._paste_data()


@register_shapes_action(trans._('Select all shapes in the current view slice'))
def select_all_shapes(layer: Shapes) -> None:
    """Select all shapes in the current view slice."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer.selected_data = set(np.nonzero(layer._data_view._displayed)[0])
        layer._set_highlight()


@register_shapes_action(trans._('Delete any selected shapes'))
def delete_selected_shapes(layer: Shapes) -> None:
    """."""

    if not layer._is_creating:
        layer.remove_selected()


@register_shapes_action(trans._('Move to front'))
def move_shapes_selection_to_front(layer: Shapes) -> None:
    layer.move_to_front()


@register_shapes_action(trans._('Move to back'))
def move_shapes_selection_to_back(layer: Shapes) -> None:
    layer.move_to_back()


@register_shapes_action(
    trans._(
        'Finish any drawing, for example when using the path or polygon tool'
    ),
)
def finish_drawing_shape(layer: Shapes) -> None:
    """Finish any drawing, for example when using the path or polygon tool."""
    layer._finish_drawing()
