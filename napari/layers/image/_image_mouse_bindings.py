from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

from napari.utils.geometry import (
    clamp_point_to_bounding_box,
    point_in_bounding_box,
)

if TYPE_CHECKING:
    from napari.layers.image.image import Image
    from napari.utils.events import Event


def move_plane_along_normal(
    layer: Image, event: Event
) -> None | Generator[None, None, None]:
    """Move a layers slicing plane along its normal vector on click and drag."""
    # early exit clauses
    if (
        'Shift' not in event.modifiers
        or layer.visible is False
        or layer.mouse_pan is False
        or len(event.dims_displayed) < 3
    ):
        return None

    # Store mouse position at start of drag
    initial_position_world = np.asarray(event.position)
    initial_view_direction_world = np.asarray(event.view_direction)

    initial_position_data = layer._world_to_displayed_data(
        initial_position_world, event.dims_displayed
    )
    initial_view_direction_data = layer._world_to_displayed_data_ray(
        initial_view_direction_world, event.dims_displayed
    )

    # Calculate intersection of click with plane through data in data coordinates
    intersection = layer.plane.intersect_with_line(
        line_position=initial_position_data,
        line_direction=initial_view_direction_data,
    )

    # Check if click was on plane and if not, exit early.
    if not point_in_bounding_box(
        intersection, layer.extent.data[:, event.dims_displayed]
    ):
        return None

    layer.plane.position = intersection

    # Store original plane position and disable interactivity during plane drag
    original_plane_position = np.copy(layer.plane.position)
    layer.mouse_pan = False

    yield None

    while event.type == 'mouse_move':
        # Project mouse drag onto plane normal
        drag_distance = layer.projected_distance_from_mouse_drag(
            start_position=initial_position_world,
            end_position=np.asarray(event.position),
            view_direction=np.asarray(event.view_direction),
            vector=layer.plane.normal,
            dims_displayed=event.dims_displayed,
        )

        # Calculate updated plane position
        updated_position = original_plane_position + (
            drag_distance * np.array(layer.plane.normal)
        )

        clamped_plane_position = clamp_point_to_bounding_box(
            updated_position,
            layer._display_bounding_box_augmented(event.dims_displayed),
        )

        layer.plane.position = clamped_plane_position
        yield None

    # Re-enable volume_layer interactivity after the drag
    layer.mouse_pan = True
    return None


def set_plane_position(layer: Image, event: Event) -> None:
    """Set plane position on double click."""
    # early exit clauses
    if (
        layer.visible is False
        or layer.mouse_pan is False
        or len(event.dims_displayed) < 3
    ):
        return

    # Calculate intersection of click with plane through data in data coordinates
    intersection = layer.plane.intersect_with_line(
        line_position=np.asarray(event.position)[event.dims_displayed],
        line_direction=np.asarray(event.view_direction)[event.dims_displayed],
    )

    # Check if click was on plane and if not, exit early.
    if not point_in_bounding_box(
        intersection, layer.extent.data[:, event.dims_displayed]
    ):
        return

    layer.plane.position = intersection
