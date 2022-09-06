from ._labels_constants import Mode
from ._labels_utils import (
    count_unique_coordinates,
    find_next_label,
    get_valid_indices,
    measure_coord_distance,
    mouse_event_to_labels_coordinate,
)


def draw(layer, event):
    """Draw with the currently selected label to a coordinate.

    This method have different behavior when draw is called
    with different labeling layer mode.

    In PAINT mode the cursor functions like a paint brush changing any
    pixels it brushes over to the current label. If the background label
    `0` is selected than any pixels will be changed to background and this
    tool functions like an eraser. The size and shape of the cursor can be
    adjusted in the properties widget.

    In FILL mode the cursor functions like a fill bucket replacing pixels
    of the label clicked on with the current label. It can either replace
    all pixels of that label or just those that are contiguous with the
    clicked on pixel. If the background label `0` is selected than any
    pixels will be changed to background and this tool functions like an
    eraser
    """
    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if layer._mode == Mode.ERASE:
        new_label = layer._background_label
    else:
        new_label = layer.selected_label

    # on press
    with layer.block_history():

        layer._draw(new_label, coordinates, coordinates)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.type == 'mouse_move':
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                layer._draw(new_label, last_cursor_coord, coordinates)
            last_cursor_coord = coordinates
            yield


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = (
        layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        or 0
    )


def toggle(layer, event):
    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if event.button == 2:
        # save previous data
        if layer.toggle_draw:
            # turn off drawing mode
            layer.toggle_draw = False
            yield
        else:
            # store current layer data before anything is drawn
            layer._previous_data = layer.data.copy()
            layer._reset_toggle_draw()
            layer.toggle_draw = True
            # add the position where mouse was first clicked
            layer._drawcoords.append(coordinates)
            yield


def toggled_draw(layer, event):
    new_label = layer.selected_label
    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if event.type == "mouse_move" and layer.toggle_draw:
        first_coord = layer._drawcoords[0]
        # measure distance to the first point of the contour
        _d = measure_coord_distance(first_coord, coordinates)
        layer._drawcoords.append(coordinates)
        # draw previous coordinate to current one
        layer._draw(new_label, layer._drawcoords[-2], coordinates)
        ncoords = count_unique_coordinates(layer._drawcoords)
        # if current coordinate are near original point
        # close contour and stop drawing

        if _d < layer.brush_size / 2 and ncoords > 2 * layer.brush_size:
            layer.toggle_draw = False
            # get valid indices for filling in contour
            valid_indices = get_valid_indices(layer, new_label)
            # erase drawn contour to remove invalid regions
            layer.data = layer._previous_data
            layer.data_setitem(valid_indices, new_label)
            # and automatically increment/find the next label
            nextlabel = find_next_label(layer)
            layer.selected_label = nextlabel
