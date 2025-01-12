import numpy as np
import numpy.typing as npt

from napari.layers.utils._text_constants import Anchor
from napari.utils.translations import trans


def get_text_anchors(
    view_data: np.ndarray | list,
    ndisplay: int,
    anchor: Anchor = Anchor.CENTER,
) -> tuple[np.ndarray, str, str]:
    # Explicitly convert to an Anchor so that string values can be used.
    text_anchor_func = TEXT_ANCHOR_CALCULATION[Anchor(anchor)]
    text_coords, anchor_x, anchor_y = text_anchor_func(view_data, ndisplay)
    return text_coords, anchor_x, anchor_y


def _calculate_anchor_center(
    view_data: np.ndarray | list, ndisplay: int
) -> tuple[np.ndarray, str, str]:
    text_coords = _calculate_bbox_centers(view_data)

    anchor_x = 'center'
    anchor_y = 'center'

    return text_coords, anchor_x, anchor_y


def _calculate_bbox_centers(view_data: np.ndarray | list) -> np.ndarray:
    """
    Calculate the bounding box of the given centers,

    Parameters
    ----------
    view_data : np.ndarray | list of ndarray
        if an ndarray, return the center across the 0-th axis.
        if a list, return the bbox center for each items.

    Returns
    -------
    An ndarray of the centers.

    """
    if isinstance(view_data, np.ndarray):
        if view_data.ndim == 2:
            # shape[1] is 2 for a 2D center, 3 for a 3D center.
            # It should work is N > 3 Dimension, but this catches mistakes
            # when the caller passed a transposed view_data
            assert view_data.shape[1] in (2, 3), view_data.shape
            # if the data are a list of coordinates, just return the coord (e.g., points)
            bbox_centers = view_data
        else:
            assert view_data.ndim == 3
            bbox_centers = np.mean(view_data, axis=0)
    elif isinstance(view_data, list):
        for coord in view_data:
            assert coord.shape[1] in (2, 3), coord.shape
        bbox_centers = np.array(
            [np.mean(coords, axis=0) for coords in view_data]
        )
    else:
        raise TypeError(
            trans._(
                'view_data should be a numpy array or list when using Anchor.CENTER',
                deferred=True,
            )
        )
    return bbox_centers


def _calculate_anchor_upper_left(
    view_data: np.ndarray | list, ndisplay: int
) -> tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_min[:, 0], bbox_min[:, 1]]).T

        anchor_x = 'left'
        anchor_y = 'top'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_anchor_upper_right(
    view_data: np.ndarray | list, ndisplay: int
) -> tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_min[:, 0], bbox_max[:, 1]]).T

        anchor_x = 'right'
        anchor_y = 'top'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_anchor_lower_left(
    view_data: np.ndarray | list, ndisplay: int
) -> tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_max[:, 0], bbox_min[:, 1]]).T

        anchor_x = 'left'
        anchor_y = 'bottom'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_anchor_lower_right(
    view_data: np.ndarray | list, ndisplay: int
) -> tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_max[:, 0], bbox_max[:, 1]]).T

        anchor_x = 'right'
        anchor_y = 'bottom'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_bbox_extents(
    view_data: np.ndarray | list,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Calculate the extents of the bounding box"""
    if isinstance(view_data, np.ndarray):
        if view_data.ndim == 2:
            # if the data are a list of coordinates, just return the coord (e.g., points)
            bbox_min = view_data
            bbox_max = view_data
        else:
            bbox_min = np.min(view_data, axis=0)
            bbox_max = np.max(view_data, axis=0)
    elif isinstance(view_data, list):
        bbox_min = np.array([np.min(coords, axis=0) for coords in view_data])
        bbox_max = np.array([np.max(coords, axis=0) for coords in view_data])
    else:
        raise TypeError(
            trans._(
                'view_data should be a numpy array or list',
                deferred=True,
            )
        )
    return bbox_min, bbox_max


TEXT_ANCHOR_CALCULATION = {
    Anchor.CENTER: _calculate_anchor_center,
    Anchor.UPPER_LEFT: _calculate_anchor_upper_left,
    Anchor.UPPER_RIGHT: _calculate_anchor_upper_right,
    Anchor.LOWER_LEFT: _calculate_anchor_lower_left,
    Anchor.LOWER_RIGHT: _calculate_anchor_lower_right,
}
