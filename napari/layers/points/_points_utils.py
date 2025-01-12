from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from napari.layers.points._points_constants import (
    SYMBOL_ALIAS,
    SYMBOL_DICT,
    Symbol,
)
from napari.utils.geometry import project_points_onto_plane
from napari.utils.translations import trans


def _create_box_from_corners_3d(
    box_corners: np.ndarray, box_normal: np.ndarray, up_vector: np.ndarray
) -> np.ndarray:
    """Get front corners for 3D box from opposing corners and edge directions.

    The resulting box will include the two corners passed in as box_corners,
    lie in a plane with normal box_normal, and have one of its axes aligned
    with the up_vector.

    Parameters
    ----------
    box_corners : np.ndarray
        The (2 x 3) array containing the two 3D points that are opposing corners
        of the bounding box.
    box_normal : np.ndarray
        The (3,) array containing the normal vector for the plane in which
        the box lies in.
    up_vector : np.ndarray
        The (3,) array containing the vector describing the up direction
        of the box.

    Returns
    -------
    box : np.ndarray
        The (4, 3) array containing the 3D coordinate of each corner of
        the box.
    """
    horizontal_vector = np.cross(box_normal, up_vector)

    diagonal_vector = box_corners[1] - box_corners[0]

    up_displacement = np.dot(diagonal_vector, up_vector) * up_vector
    horizontal_displacement = (
        np.dot(diagonal_vector, horizontal_vector) * horizontal_vector
    )

    corner_1 = box_corners[0] + horizontal_displacement
    corner_3 = box_corners[0] + up_displacement

    box = np.array([box_corners[0], corner_1, box_corners[1], corner_3])
    return box


def create_box(data: npt.NDArray) -> npt.NDArray:
    """Create the axis aligned interaction box of a list of points

    Parameters
    ----------
    data : (N, 2) array
        Points around which the interaction box is created

    Returns
    -------
    box : (4, 2) array
        Vertices of the interaction box
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    box = np.array([tl, tr, br, bl])
    return box


def points_to_squares(points: npt.NDArray, sizes: npt.NDArray) -> npt.NDArray:
    """Expand points to squares defined by their size

    Parameters
    ----------
    points : (N, 2) array
        Points to be turned into squares
    sizes : (N,) array
        Size of each point

    Returns
    -------
    rect : (4N, 2) array
        Vertices of the expanded points
    """
    rect = np.concatenate(
        [
            points + 0.5 * np.array([sizes, sizes]).T,
            points + 0.5 * np.array([sizes, -sizes]).T,
            points + 0.5 * np.array([-sizes, sizes]).T,
            points + 0.5 * np.array([-sizes, -sizes]).T,
        ],
        axis=0,
    )
    return rect


def _points_in_box_3d(
    box_corners: np.ndarray,
    points: np.ndarray,
    sizes: np.ndarray,
    box_normal: np.ndarray,
    up_direction: np.ndarray,
) -> list[int]:
    """Determine which points are inside of 2D bounding box.

    The 2D bounding box extends infinitely in both directions along its normal
    vector.

    Point selection algorithm:
        1. Project the points in view and drag box corners on to a plane
        parallel to the canvas (i.e., normal direction is the view direction).
        2. Rotate the points/bbox corners to a new basis comprising
        the bbox normal, the camera up direction, and the
        "horizontal direction" (i.e., vector orthogonal to bbox normal
        and camera up direction). This makes it such that the bounding box
        is axis aligned (i.e., the major and minor axes of the bounding
        box are aligned with the new 0 and 1 axes).
        3. Determine which points are in the bounding box in 2D. We can
        simplify to 2D since the points and bounding box now lie in the same
        plane.

    Parameters
    ----------
    box_corners : np.ndarray
        The (2 x 3) array containing the two 3D points that are opposing corners
        of the bounding box.
    points : np.ndarray
        The (n x3) array containing the n 3D points that are to be tested for
        being inside of the bounding box.
    sizes : np.ndarray
        The (n,) array containing the diameters of the points.
    box_normal : np.ndarray
        The (3,) array containing the normal vector for the plane in which
        the box lies in.
    up_direction : np.ndarray
        The (3,) array containing the vector describing the up direction
        of the box.

    Returns
    -------
    inside : list
        Indices of points inside the box.
    """
    # the the corners for a bounding box that is has one axis aligned
    # with the camera up direction and is normal to the view direction.
    bbox_corners = _create_box_from_corners_3d(
        box_corners, box_normal, up_direction
    )

    # project points onto the same plane as the box
    projected_points, _ = project_points_onto_plane(
        points=points,
        plane_point=bbox_corners[0],
        plane_normal=box_normal,
    )

    # create a new basis in which the bounding box is
    # axis aligned
    horz_direction = np.cross(box_normal, up_direction)
    plane_basis = np.column_stack([up_direction, horz_direction, box_normal])

    # transform the points and bounding box into a new basis
    # such that the bounding box is axis aligned
    bbox_corners_axis_aligned = bbox_corners @ plane_basis
    bbox_corners_axis_aligned = bbox_corners_axis_aligned[:, :2]
    points_axis_aligned = projected_points @ plane_basis
    points_axis_aligned = points_axis_aligned[:, :2]

    # determine which points are in the box using the
    # axis-aligned basis
    return points_in_box(bbox_corners_axis_aligned, points_axis_aligned, sizes)


def points_in_box(
    corners: np.ndarray, points: np.ndarray, sizes: np.ndarray
) -> list[int]:
    """Find which points are in an axis aligned box defined by its corners.

    Parameters
    ----------
    corners : (2, 2) array
        The top-left and bottom-right corners of the box.
    points : (N, 2) array
        Points to be checked
    sizes : (N,) array
        Size of each point

    Returns
    -------
    inside : list
        Indices of points inside the box
    """
    box = create_box(corners)[[0, 2]]
    # Check all four corners in a square around a given point. If any corner
    # is inside the box, then that point is considered inside
    point_corners = points_to_squares(points, sizes)
    below_top = np.all(box[1] >= point_corners, axis=1)
    above_bottom = np.all(point_corners >= box[0], axis=1)
    point_corners_in_box = np.where(np.logical_and(below_top, above_bottom))[0]
    # Determine indices of points which have at least one corner inside box
    inside = np.unique(point_corners_in_box % len(points))
    return list(inside)


def fix_data_points(
    points: np.ndarray | None, ndim: int | None
) -> tuple[np.ndarray, int]:
    """
    Ensure that points array is 2d and have second dimension of size ndim (default 2 for empty arrays)

    Parameters
    ----------
    points : (N, M) array or None
        Points to be checked
    ndim : int or None
        number of expected dimensions

    Returns
    -------
    points : (N, M) array
        Points array
    ndim : int
        number of dimensions

    Raises
    ------
    ValueError
        if ndim does not match with second dimensions of points
    """
    if points is None or len(points) == 0:
        if ndim is None:
            ndim = 2
        points = np.empty((0, ndim))
    else:
        points = np.atleast_2d(points)
        data_ndim = points.shape[1]
        if ndim is not None and ndim != data_ndim:
            raise ValueError(
                trans._(
                    'Points dimensions must be equal to ndim',
                    deferred=True,
                )
            )
        ndim = data_ndim
    return points, ndim


def symbol_conversion(symbol: str | Symbol) -> Symbol:
    """
    Convert a string or Symbol to a Symbol instance.
    """
    if isinstance(symbol, str):
        symbol = SYMBOL_ALIAS.get(symbol, symbol)
    return Symbol(symbol)


def fast_dict_get(symbols: np.ndarray | list, d: dict) -> np.ndarray:
    """
    Get the values from a dictionary using a list of keys.
    """
    # dtype has to be object, otherwise np.vectorize will cut it down to `U(N)`,
    # where N is the biggest string currently in the array.
    return np.vectorize(d.__getitem__, otypes=[object])(symbols)


def coerce_symbols(
    symbol: str | Symbol | Sequence[str | Symbol],
) -> np.ndarray:
    """
    Parse an array of symbols and convert it to the correct strings.
    If single value is given, it is converted to single element array.

    Ensures that all strings are valid symbols and convert aliases.

    Parameters
    ----------
    symbol : str or Symbol or Sequence of str or Symbol
        data to be convert to array of Symbols.

    Returns
    -------
    symbols : np.ndarray
        array of Symbols
    """
    # if a symbol is a unique string or Symbol instance, convert it to a
    # proper Symbol instance
    if isinstance(symbol, (str, Symbol)):
        return np.array([symbol_conversion(symbol)], dtype=object)

    if not isinstance(symbol, np.ndarray):
        symbol = np.array(symbol)

    return fast_dict_get(symbol, SYMBOL_DICT)
