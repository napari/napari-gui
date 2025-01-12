from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shapes_constants import Box, Mode
from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.layers.shapes._shapes_utils import point_to_lines
from napari.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy.typing as npt
    from vispy.app.canvas import MouseEvent

    from napari.layers.shapes.shapes import Shapes


def highlight(layer: Shapes, event: MouseEvent) -> None:
    """Render highlights of shapes.

    Highlight hovered shapes, including boundaries, vertices, interaction boxes, and drag
    selection box when appropriate.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event. Though not used here it is passed as argument by the
        shapes layer mouse move callbacks.

    Returns
    -------
    None
    """
    layer._set_highlight()


def select(layer: Shapes, event: MouseEvent) -> Generator[None, None, None]:
    """Select shapes or vertices either in select or direct select mode.

    Once selected shapes can be moved or resized, and vertices can be moved
    depending on the mode. Holding shift when resizing a shape will preserve
    the aspect ratio.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    shift = 'Shift' in event.modifiers
    # on press
    value = layer.get_value(event.position, world=True)
    layer._moving_value = copy(value)
    shape_under_cursor, vertex_under_cursor = value
    if vertex_under_cursor is None:
        if shift and shape_under_cursor is not None:
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            else:
                if len(layer.selected_data):
                    # one or more shapes already selected
                    layer.selected_data.add(shape_under_cursor)
                else:
                    # first shape being selected
                    layer.selected_data = {shape_under_cursor}
        elif shape_under_cursor is not None:
            if shape_under_cursor not in layer.selected_data:
                layer.selected_data = {shape_under_cursor}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    # we don't update the thumbnail unless a shape has been moved
    update_thumbnail = False

    # Set _drag_start value here to prevent an offset when mouse_move happens
    # https://github.com/napari/napari/pull/4999
    _set_drag_start(layer, layer.world_to_data(event.position))
    yield

    # on move
    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        # ToDo: Need to pass moving_coordinates to allow fixed aspect ratio
        # keybinding to work, this should be dropped
        layer._moving_coordinates = coordinates
        # Drag any selected shapes
        if len(layer.selected_data) == 0:
            _drag_selection_box(layer, coordinates)
        else:
            _move_active_element_under_cursor(layer, coordinates)

        # if a shape is being moved, update the thumbnail
        if layer._is_moving:
            update_thumbnail = True
        yield

    # only emit data once dragging has finished
    if layer._is_moving:
        vertex_indices = tuple(
            tuple(
                vertex_index
                for vertex_index, coord in enumerate(layer.data[i])
            )
            for i in layer.selected_data
        )
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=tuple(layer.selected_data),
            vertex_indices=vertex_indices,
        )

    # on release
    shift = 'Shift' in event.modifiers
    if not layer._is_moving and not layer._is_selecting and not shift:
        if shape_under_cursor is not None:
            layer.selected_data = {shape_under_cursor}
        else:
            layer.selected_data = set()
    elif layer._is_selecting:
        layer.selected_data = layer._data_view.shapes_in_box(layer._drag_box)
        layer._is_selecting = False
        layer._set_highlight()

    layer._is_moving = False
    layer._drag_start = None
    layer._drag_box = None
    layer._fixed_vertex = None
    layer._moving_value = (None, None)
    layer._set_highlight()

    if update_thumbnail:
        layer._update_thumbnail()


def add_line(layer: Shapes, event: MouseEvent) -> Generator[None, None, None]:
    """Add a line.

    Adds a line by connecting 2 ndim points. On press one point is set under the mouse cursor and a second point is
    created with a very minor offset to the first point. If moving mouse while mouse is pressed the second point will
    track the cursor. The second point it set upon mouse release.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    # full size is the initial offset of the second point compared to the first point of the line.
    size = layer._normalized_vertex_radius / 2
    full_size = np.zeros(layer.ndim, dtype=float)
    for i in layer._slice_input.displayed:
        full_size[i] = size

    coordinates = layer.world_to_data(event.position)
    layer._moving_coordinates = coordinates

    # corner is first datapoint defining the line
    corner = np.array(coordinates)
    data = np.array([corner, corner + full_size])

    # adds data to layer.data and handles mouse move (cursor tracking) and release event (setting second point)
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='line'
    )


def add_ellipse(
    layer: Shapes, event: MouseEvent
) -> Generator[None, None, None]:
    """
    Add an ellipse to the shapes layer.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    size = layer._normalized_vertex_radius / 2
    size_h = np.zeros(layer.ndim, dtype=float)
    size_h[layer._slice_input.displayed[0]] = size
    size_v = np.zeros(layer.ndim, dtype=float)
    size_v[layer._slice_input.displayed[1]] = size

    coordinates = layer.world_to_data(event.position)
    corner = np.array(coordinates)
    data = np.array(
        [corner, corner + size_v, corner + size_h + size_v, corner + size_h]
    )
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='ellipse'
    )


def add_rectangle(
    layer: Shapes, event: MouseEvent
) -> Generator[None, None, None]:
    """Add a rectangle to the shapes layer.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    size = layer._normalized_vertex_radius / 2
    size_h = np.zeros(layer.ndim, dtype=float)
    size_h[layer._slice_input.displayed[0]] = size
    size_v = np.zeros(layer.ndim, dtype=float)
    size_v[layer._slice_input.displayed[1]] = size

    coordinates = layer.world_to_data(event.position)
    corner = np.array(coordinates)
    data = np.array(
        [corner, corner + size_v, corner + size_h + size_v, corner + size_h]
    )

    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='rectangle'
    )


def _add_line_rectangle_ellipse(
    layer: Shapes, event: MouseEvent, data: npt.NDArray, shape_type: str
) -> Generator[None, None, None]:
    """Helper function for adding a line, rectangle or ellipse.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    data : np.NDarray
        Array containing the initial datapoints of the shape in image data space.
    shape_type : str
        String indicating the type of shape to be added.
    """
    # on press
    # reset layer._aspect_ratio for a new shape
    layer._aspect_ratio = 1
    # Start drawing rectangle / ellipse / line
    layer.add(data, shape_type=shape_type, gui=True)
    layer.selected_data = {layer.nshapes - 1}
    layer._value = (layer.nshapes - 1, 4)
    layer._moving_value = copy(layer._value)
    layer.refresh()
    yield

    # on move
    while event.type == 'mouse_move':
        # Drag any selected shapes
        coordinates = layer.world_to_data(event.position)
        layer._moving_coordinates = coordinates
        _move_active_element_under_cursor(layer, coordinates)
        yield

    # on release
    layer._finish_drawing()


def finish_drawing_shape(layer: Shapes, event: MouseEvent) -> None:
    """Finish drawing of shape.

    Calls the finish drawing method of the shapes layer which resets all the properties used for shape drawing
    and deletes the shape if the number of vertices do not meet the threshold of 3.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event. Not used here, but passed as argument due to being a
        double click callback of the shapes layer.
    """
    layer._finish_drawing()


def initiate_polygon_draw(
    layer: Shapes, coordinates: tuple[float, ...]
) -> None:
    """Start drawing of polygon.

    Creates the polygon shape when initializing the draw, adding to layer and selecting the initiatlized shape and
    setting required layer attributes for drawing.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    coordinates : Tuple[float, ...]
        A tuple with the coordinates of the initial vertex in image data space.
    """
    layer._is_creating = True
    data = np.array([coordinates, coordinates])
    layer.add(data, shape_type='path', gui=True)
    layer.selected_data = {layer.nshapes - 1}
    layer._value = (layer.nshapes - 1, 1)
    layer._moving_value = copy(layer._value)
    layer._set_highlight()


def add_path_polygon_lasso(
    layer: Shapes, event: MouseEvent
) -> Generator[None, None, None]:
    """Add, draw and finish drawing of polygon.

    Initiates, draws and finishes the lasso polygon in drag mode (tablet) or
    initiates and finishes the lasso polygon when drawing with the mouse.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    # on press
    coordinates = layer.world_to_data(event.position)
    if layer._is_creating is False:
        # Set last cursor position to initial position of the mouse when starting to draw the shape
        layer._last_cursor_position = np.array(event.pos)

        # Start drawing a path
        initiate_polygon_draw(layer, coordinates)
        yield

        while event.type == 'mouse_move':
            polygon_creating(layer, event)
            yield
        index = layer._moving_value[0]
        vertices = layer._data_view.shapes[index].data
        # If number of vertices is higher than 2, tablet draw mode is assumed and shape is finished upon mouse release
        if len(vertices) > 2:
            layer._finish_drawing()
    else:
        # This code block is responsible for finishing drawing in mouse draw mode
        layer._finish_drawing()


def add_vertex_to_path(
    layer: Shapes,
    event: MouseEvent,
    index: int,
    coordinates: tuple[float, ...],
    new_type: str | None,
) -> None:
    """Add a vertex to an existing path or polygon and edit the layer view.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    index : int
        The index of the shape being added, e.g. first shape in the layer has index 0.
    coordinates : Tuple[float, ...]
        The coordinates of the vertex being added to the shape being drawn in image data space
    new_type : Optional[str]
        Type of the shape being added.
    """
    vertices = layer._data_view.shapes[index].data
    vertices = np.concatenate((vertices, [coordinates]), axis=0)
    value = layer.get_value(event.position, world=True)
    # If there was no move event between two clicks value[1] is None
    # and needs to be taken care of.
    if value[1] is None:
        value = layer._moving_value
    layer._value = (value[0], value[1] + 1)
    layer._moving_value = copy(layer._value)
    layer._data_view.edit(index, vertices, new_type=new_type)
    layer._selected_box = layer.interaction_box(layer.selected_data)
    layer._last_cursor_position = np.array(event.pos)


def polygon_creating(layer: Shapes, event: MouseEvent) -> None:
    """Let active vertex follow cursor while drawing polygon, adding it to polygon after a certain distance.

    When drawing a polygon in lasso mode, a vertex follows the cursor, creating a polygon
    visually that is *not* the final polygon to be created: it is the polygon if the current
    mouse position were to be the last position added. After the mouse moves a distance of 10 screen pixels,
    a new vertex is automatically added and the last cursor position is set to the global screen coordinates
    at that moment.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    if layer._is_creating:
        coordinates = layer.world_to_data(event.position)
        move_active_vertex_under_cursor(layer, coordinates)

        if layer._mode == Mode.ADD_POLYGON_LASSO:
            index = layer._moving_value[0]

            position_diff = np.linalg.norm(
                event.pos - layer._last_cursor_position
            )
            if (
                position_diff
                > get_settings().experimental.lasso_vertex_distance
            ):
                add_vertex_to_path(layer, event, index, coordinates, None)


def add_path_polygon(layer: Shapes, event: MouseEvent) -> None:
    """Add a path or polygon or add vertex to an existing one.

    When shape is not yet being created, initiates the drawing of a polygon on mouse press. Else, on subsequent mouse
    presses, add vertex to polygon being created.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    # on press
    coordinates = layer.world_to_data(event.position)
    if layer._is_creating is False:
        # Set last cursor position to initial position of the mouse when starting to draw the shape
        layer._last_cursor_position = np.array(event.pos)
        # Start drawing a path
        initiate_polygon_draw(layer, coordinates)
    else:
        # Add to an existing path or polygon
        index = layer._moving_value[0]
        new_type = Polygon if layer._mode == Mode.ADD_POLYGON else None
        # Ensure the position of the new vertex is different from the previous
        # one before adding it. See napari/napari#6597
        if not np.array_equal(
            np.array(event.pos), layer._last_cursor_position
        ):
            add_vertex_to_path(layer, event, index, coordinates, new_type)


def move_active_vertex_under_cursor(
    layer: Shapes, coordinates: tuple[float, ...]
) -> None:
    """While a path or polygon is being created, move next vertex to be added.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    coordinates : Tuple[float, ...]
        The coordinates in data space of the vertex to be potentially added, e.g. vertex tracks the mouse cursor
        position.
    """
    if layer._is_creating:
        _move_active_element_under_cursor(layer, coordinates)


def vertex_insert(layer: Shapes, event: MouseEvent) -> None:
    """Insert a vertex into a selected shape.

    The vertex will get inserted in between the vertices of the closest edge
    from all the edges in selected shapes. Vertices cannot be inserted into
    Ellipses.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    # Determine all the edges in currently selected shapes
    all_edges = np.empty((0, 2, 2))
    all_edges_shape = np.empty((0, 2), dtype=int)
    for index in layer.selected_data:
        shape_type = type(layer._data_view.shapes[index])
        if shape_type == Ellipse:
            # Adding vertex to ellipse not implemented
            pass
        else:
            vertices = layer._data_view.displayed_vertices[
                layer._data_view.displayed_index == index
            ]
            # Find which edge new vertex should inserted along
            closed = shape_type != Path
            n = len(vertices)
            if closed:
                lines = np.array(
                    [[vertices[i], vertices[(i + 1) % n]] for i in range(n)]
                )
            else:
                lines = np.array(
                    [[vertices[i], vertices[i + 1]] for i in range(n - 1)]
                )
            all_edges = np.append(all_edges, lines, axis=0)
            indices = np.array(
                [np.repeat(index, len(lines)), list(range(len(lines)))]
            ).T
            all_edges_shape = np.append(all_edges_shape, indices, axis=0)

    if len(all_edges) == 0:
        # No appropriate edges were found
        return

    # Determine the closet edge to the current cursor coordinate
    coordinates = layer.world_to_data(event.position)
    coord = [coordinates[i] for i in layer._slice_input.displayed]
    ind, loc = point_to_lines(coord, all_edges)
    index = all_edges_shape[ind][0]
    ind = all_edges_shape[ind][1] + 1
    shape_type = type(layer._data_view.shapes[index])
    if shape_type == Line:
        # Adding vertex to line turns it into a path
        new_type = Path
    elif shape_type == Rectangle:
        # Adding vertex to rectangle turns it into a polygon
        new_type = Polygon
    else:
        new_type = None
    closed = shape_type != Path
    vertices = layer._data_view.shapes[index].data
    if not closed:
        if int(ind) == 1 and loc < 0:
            ind = 0
        elif int(ind) == len(vertices) - 1 and loc > 1:
            ind = ind + 1

    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGING,
        data_indices=(index,),
        vertex_indices=((ind,),),
    )
    # Insert new vertex at appropriate place in vertices of target shape
    vertices = np.insert(vertices, ind, [coordinates], axis=0)
    with layer.events.set_data.blocker():
        layer._data_view.edit(index, vertices, new_type=new_type)
        layer._selected_box = layer.interaction_box(layer.selected_data)
    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGED,
        data_indices=(index,),
        vertex_indices=((ind,),),
    )
    layer.refresh()


def vertex_remove(layer: Shapes, event: MouseEvent) -> None:
    """Remove a vertex from a selected shape.

    If a vertex is clicked on remove it from the shape it is in. If this cause
    the shape to shrink to a size that no longer is valid remove the whole
    shape.

    Parameters
    ----------
    layer : Shapes
        Napari shapes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    value = layer.get_value(event.position, world=True)
    shape_under_cursor, vertex_under_cursor = value
    if vertex_under_cursor is None:
        # No vertex was clicked on so return
        return

    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGING,
        data_indices=(shape_under_cursor,),
        vertex_indices=((vertex_under_cursor,),),
    )

    # Have clicked on a current vertex so remove
    shape_type = type(layer._data_view.shapes[shape_under_cursor])
    if shape_type == Ellipse:
        # Removing vertex from ellipse not implemented
        return
    vertices = layer._data_view.shapes[shape_under_cursor].data
    if len(vertices) <= 2 or (shape_type == Polygon and len(vertices) == 3):
        # If only 2 vertices present, remove whole shape
        with layer.events.set_data.blocker():
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            layer._data_view.remove(shape_under_cursor)
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    else:
        if shape_type == Rectangle:  # noqa SIM108
            # Deleting vertex from a rectangle creates a polygon
            new_type = Polygon
        else:
            new_type = None
        # Remove clicked on vertex
        vertices = np.delete(vertices, vertex_under_cursor, axis=0)
        with layer.events.set_data.blocker():
            layer._data_view.edit(
                shape_under_cursor, vertices, new_type=new_type
            )
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    layer.events.data(
        value=layer.data,
        action=ActionType.CHANGED,
        data_indices=(shape_under_cursor,),
        vertex_indices=((vertex_under_cursor,),),
    )
    layer.refresh()


def _drag_selection_box(layer: Shapes, coordinates: tuple[float, ...]) -> None:
    """Drag a selection box.

    Parameters
    ----------
    layer : napari.layers.Shapes
        Shapes layer.
    coordinates : Tuple[float, ...]
        The current position of the cursor during the mouse move event in image data space.
    """
    # If something selected return
    if len(layer.selected_data) > 0:
        return

    coord = [coordinates[i] for i in layer._slice_input.displayed]

    # Create or extend a selection box
    layer._is_selecting = True
    if layer._drag_start is None:
        layer._drag_start = coord
    layer._drag_box = np.array([layer._drag_start, coord])
    layer._set_highlight()


def _set_drag_start(
    layer: Shapes, coordinates: tuple[float, ...]
) -> list[float]:
    """Indicate where in data space a drag event started.

    Sets the coordinates relative to the center of the bounding box of a shape and returns the position
    of where a drag event of a shape started.

    Parameters
    ----------
    layer : Shapes
        The napari layer shape
    coordinates : Tuple[float, ...]
        The position in image data space where dragging started.

    Returns
    -------
    coord: List[float, ...]
        The coordinates of where a shape drag event started.
    """
    coord = [coordinates[i] for i in layer._slice_input.displayed]
    if layer._drag_start is None and len(layer.selected_data) > 0:
        center = layer._selected_box[Box.CENTER]
        layer._drag_start = coord - center
    return coord


def _move_active_element_under_cursor(
    layer: Shapes, coordinates: tuple[float, ...]
) -> None:
    """Moves object at given mouse position and set of indices.

    Parameters
    ----------
    layer : napari.layers.Shapes
        Shapes layer.
    coordinates : Tuple[float, ...]
        Position of mouse cursor in data coordinates.
    """
    # If nothing selected return
    if len(layer.selected_data) == 0:
        return

    vertex = layer._moving_value[1]

    if layer._mode in (
        [Mode.SELECT, Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
    ):
        if layer._mode == Mode.SELECT and not layer._is_moving:
            vertex_indices = tuple(
                tuple(
                    vertex_index
                    for vertex_index, coord in enumerate(layer.data[i])
                )
                for i in layer.selected_data
            )
            layer.events.data(
                value=layer.data,
                action=ActionType.CHANGING,
                data_indices=tuple(layer.selected_data),
                vertex_indices=vertex_indices,
            )

        coord = _set_drag_start(layer, coordinates)
        layer._moving_coordinates = coordinates
        layer._is_moving = True
        if vertex is None:
            # Check where dragging box from to move whole object
            center = layer._selected_box[Box.CENTER]
            shift = coord - center - layer._drag_start
            for index in layer.selected_data:
                layer._data_view.shift(index, shift)
            layer._selected_box = layer._selected_box + shift
            layer.refresh()
        elif vertex < Box.LEN:
            # Corner / edge vertex is being dragged so resize object
            # Also applies while drawing line, rectangle, ellipse
            box = layer._selected_box
            if layer._fixed_vertex is None:
                layer._fixed_index = (vertex + 4) % Box.LEN
                layer._fixed_vertex = box[layer._fixed_index]

            handle_offset = box[Box.HANDLE] - box[Box.CENTER]
            if np.linalg.norm(handle_offset) == 0:
                handle_offset = [1, 1]
            handle_offset_norm = handle_offset / np.linalg.norm(handle_offset)

            rot = np.array(
                [
                    [handle_offset_norm[0], -handle_offset_norm[1]],
                    [handle_offset_norm[1], handle_offset_norm[0]],
                ]
            )
            inv_rot = np.linalg.inv(rot)

            fixed = layer._fixed_vertex
            new = list(coord)
            box_center = box[Box.CENTER]
            if layer._fixed_aspect and layer._fixed_index % 2 == 0:
                # corner
                # ensure line rotates through 45 degree steps if aspect ratio is 1
                if layer._mode == Mode.ADD_LINE and layer._aspect_ratio == 1:
                    new_offset = coord - layer._fixed_vertex
                    angle_rad = np.arctan2(new_offset[0], -new_offset[1])
                    angle_rad = np.round(angle_rad / (np.pi / 4)) * (np.pi / 4)
                    new = (
                        np.array([np.sin(angle_rad), -np.cos(angle_rad)])
                        * np.linalg.norm(new - box_center)
                        + box_center
                    )
                else:
                    new = (box[vertex] - box_center) / np.linalg.norm(
                        box[vertex] - box_center
                    ) * np.linalg.norm(new - box_center) + box_center

            if layer._fixed_index % 2 == 0:
                # corner selected
                drag_scale = (inv_rot @ (new - fixed)) / (
                    inv_rot @ (box[vertex] - fixed)
                )
            elif layer._fixed_index % 4 == 3:
                # top or bottom selected
                drag_scale = np.array(
                    [
                        (inv_rot @ (new - fixed))[0]
                        / (inv_rot @ (box[vertex] - fixed))[0],
                        1,
                    ]
                )
            else:
                # left or right selected
                drag_scale = np.array(
                    [
                        1,
                        (inv_rot @ (new - fixed))[1]
                        / (inv_rot @ (box[vertex] - fixed))[1],
                    ]
                )

            # prevent box from shrinking below a threshold size
            size = (np.linalg.norm(box[Box.TOP_LEFT] - box_center),)
            if (
                np.linalg.norm(size * drag_scale)
                < layer._normalized_vertex_radius
            ):
                drag_scale[:] = 1
            # on vertical/horizontal drags we get scale of 0
            # when we actually simply don't want to scale
            drag_scale[drag_scale == 0] = 1

            # check orientation of box
            if abs(handle_offset_norm[0]) == 1:
                for index in layer.selected_data:
                    layer._data_view.scale(
                        index, drag_scale, center=layer._fixed_vertex
                    )
                layer._scale_box(drag_scale, center=layer._fixed_vertex)
            else:
                scale_mat = np.array([[drag_scale[0], 0], [0, drag_scale[1]]])
                transform = rot @ scale_mat @ inv_rot
                for index in layer.selected_data:
                    layer._data_view.shift(index, -layer._fixed_vertex)
                    layer._data_view.transform(index, transform)
                    layer._data_view.shift(index, layer._fixed_vertex)
                layer._transform_box(transform, center=layer._fixed_vertex)
            layer.refresh()
        elif vertex == 8:
            # Rotation handle is being dragged so rotate object
            handle = layer._selected_box[Box.HANDLE]
            layer._fixed_vertex = layer._selected_box[Box.CENTER]
            offset = handle - layer._fixed_vertex
            layer._drag_start = -np.degrees(np.arctan2(offset[0], -offset[1]))

            new_offset = coord - layer._fixed_vertex
            new_angle = -np.degrees(np.arctan2(new_offset[0], -new_offset[1]))
            fixed_offset = handle - layer._fixed_vertex
            fixed_angle = -np.degrees(
                np.arctan2(fixed_offset[0], -fixed_offset[1])
            )

            if layer._fixed_aspect:
                angle = np.round(new_angle / 45) * 45 - fixed_angle
            else:
                angle = new_angle - fixed_angle

            for index in layer.selected_data:
                layer._data_view.rotate(
                    index, angle, center=layer._fixed_vertex
                )
            layer._rotate_box(angle, center=layer._fixed_vertex)
            layer.refresh()

    elif (
        layer._mode
        in [
            Mode.DIRECT,
            Mode.ADD_PATH,
            Mode.ADD_POLYGON,
            Mode.ADD_POLYGON_LASSO,
        ]
        and vertex is not None
    ):
        layer._moving_coordinates = coordinates
        layer._is_moving = True
        index = layer._moving_value[0]
        shape_type = type(layer._data_view.shapes[index])
        if shape_type == Ellipse:
            # TODO: Implement DIRECT vertex moving of ellipse
            pass
        else:
            new_type = Polygon if shape_type == Rectangle else None
            vertices = layer._data_view.shapes[index].data
            vertices[vertex] = coordinates
            layer._data_view.edit(index, vertices, new_type=new_type)
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
            layer.refresh()


def _set_highlight(layer: Shapes, event: MouseEvent) -> None:
    if event.type in {'mouse_press', 'mouse_wheel'}:
        layer._set_highlight()
