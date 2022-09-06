from functools import lru_cache

import numpy as np
from scipy.ndimage import binary_fill_holes


def interpolate_coordinates(old_coord, new_coord, brush_size):
    """Interpolates coordinates depending on brush size.

    Useful for ensuring painting is continuous in labels layer.

    Parameters
    ----------
    old_coord : np.ndarray, 1x2
        Last position of cursor.
    new_coord : np.ndarray, 1x2
        Current position of cursor.
    brush_size : float
        Size of brush, which determines spacing of interpolation.

    Returns
    -------
    coords : np.array, Nx2
        List of coordinates to ensure painting is continuous
    """
    if old_coord is None:
        old_coord = new_coord
    if new_coord is None:
        new_coord = old_coord
    num_step = round(
        max(abs(np.array(new_coord) - np.array(old_coord))) / brush_size * 4
    )
    coords = [
        np.linspace(old_coord[i], new_coord[i], num=int(num_step + 1))
        for i in range(len(new_coord))
    ]
    coords = np.stack(coords).T
    if len(coords) > 1:
        coords = coords[1:]

    return coords


@lru_cache(maxsize=64)
def sphere_indices(radius, scale):
    """Generate centered indices within circle or n-dim ellipsoid.

    Parameters
    -------
    radius : float
        Radius of circle/sphere
    scale : tuple of float
        The scaling to apply to the sphere along each axis

    Returns
    -------
    mask_indices : array
        Centered indices within circle/sphere
    """
    ndim = len(scale)
    abs_scale = np.abs(scale)
    scale_normalized = np.asarray(abs_scale, dtype=float) / np.min(abs_scale)
    # Create multi-dimensional grid to check for
    # circle/membership around center
    r_normalized = radius / scale_normalized + 0.5
    slices = [
        slice(-int(np.ceil(r)), int(np.floor(r)) + 1) for r in r_normalized
    ]

    indices = np.mgrid[slices].T.reshape(-1, ndim)
    distances_sq = np.sum((indices * scale_normalized) ** 2, axis=1)
    # Use distances within desired radius to mask indices in grid
    mask_indices = indices[distances_sq <= radius**2].astype(int)

    return mask_indices


def indices_in_shape(idxs, shape):
    """Return idxs after filtering out indices that are not in given shape.

    Parameters
    ----------
    idxs : tuple of array of int, or 2D array of int
        The input coordinates. These should be in one of two formats:

        - a tuple of 1D arrays, as for NumPy fancy indexing, or
        - a 2D array of shape (ncoords, ndim), as a list of coordinates

    shape : tuple of int
        The shape in which all indices must fit.

    Returns
    -------
    idxs_filtered : tuple of array of int, or 2D array of int
        The subset of the input idxs that falls within shape.

    Examples
    --------
    >>> idxs0 = (np.array([5, 45, 2]), np.array([6, 5, -5]))
    >>> indices_in_shape(idxs0, (10, 10))
    (array([5]), array([6]))
    >>> idxs1 = np.transpose(idxs0)
    >>> indices_in_shape(idxs1, (10, 10))
    array([[5, 6]])
    """
    np_index = isinstance(idxs, tuple)
    if np_index:  # normalize to 2D coords array
        idxs = np.transpose(idxs)
    keep_coords = np.logical_and(
        np.all(idxs >= 0, axis=1), np.all(idxs < np.array(shape), axis=1)
    )
    filtered = idxs[keep_coords]
    if np_index:  # convert back to original format
        filtered = tuple(filtered.T)
    return filtered


def get_dtype(layer):
    """Returns dtype of layer data

    Parameters
    ----------
    layer : Labels
        Labels layer (may be multiscale)

    Returns
    -------
    dtype
        dtype of Layer data
    """
    layer_data = layer.data
    if not isinstance(layer_data, list):
        layer_data = [layer_data]
    layer_data_level = layer_data[0]
    if hasattr(layer_data_level, 'dtype'):
        layer_dtype = layer_data_level[0].dtype
    else:
        layer_dtype = type(layer_data_level)

    return layer_dtype


def first_nonzero_coordinate(data, start_point, end_point):
    """Coordinate of the first nonzero element between start and end points.

    Parameters
    ----------
    data : nD array, shape (N1, N2, ..., ND)
        A data volume.
    start_point : array, shape (D,)
        The start coordinate to check.
    end_point : array, shape (D,)
        The end coordinate to check.

    Returns
    -------
    coordinates : array of int, shape (D,)
        The coordinates of the first nonzero element along the ray, or None.
    """
    shape = np.asarray(data.shape)
    length = np.linalg.norm(end_point - start_point)
    length_int = np.round(length).astype(int)
    coords = np.linspace(start_point, end_point, length_int + 1, endpoint=True)
    clipped_coords = np.clip(np.round(coords), 0, shape - 1).astype(int)
    nonzero = np.flatnonzero(data[tuple(clipped_coords.T)])
    return None if len(nonzero) == 0 else clipped_coords[nonzero[0]]


def mouse_event_to_labels_coordinate(layer, event):
    """Return the data coordinate of a Labels layer mouse event in 2D or 3D.

    In 2D, this is just the event's position transformed by the layer's
    world_to_data transform.

    In 3D, a ray is cast in data coordinates, and the coordinate of the first
    nonzero value along that ray is returned. If the ray only contains zeros,
    None is returned.

    Parameters
    ----------
    layer : napari.layers.Labels
        The Labels layer.
    event : vispy MouseEvent
        The mouse event, containing position and view direction attributes.

    Returns
    -------
    coordinates : array of int or None
        The data coordinates for the mouse event.
    """
    ndim = len(layer._dims_displayed)
    if ndim == 2:
        coordinates = layer.world_to_data(event.position)
    else:  # 3d
        start, end = layer.get_ray_intersections(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=layer._dims_displayed,
            world=True,
        )
        if start is None and end is None:
            return None
        coordinates = first_nonzero_coordinate(layer.data, start, end)
    return coordinates


def measure_coord_distance(refcoord, coord):
    a = np.array(refcoord)
    b = np.array(coord)
    return np.linalg.norm(a - b)


def get_valid_indices(layer, label):
    # get current view of layer data
    curslice_index = layer._slice_indices
    curdata = layer.data[curslice_index].copy()
    # fill holes for current slice
    curlabel = binary_fill_holes(curdata == label)
    prevlabel = layer._previous_data[curslice_index].copy()
    occupied_region = prevlabel > 0
    valid_region = (curlabel) ^ (curlabel & occupied_region)
    # to get correct indexing, use a blank array with the label shape
    blankdata = np.zeros(layer.data.shape, dtype=bool)
    # then fill in the current viewed slice with valid region
    blankdata[curslice_index] = valid_region
    return np.where(blankdata)


def count_unique_coordinates(coords):
    coordsarr = np.round(np.array(coords)).astype(int)
    uniquecoords = np.unique(coordsarr, axis=0)
    return len(uniquecoords)


def find_next_label(layer):
    uniquelabels = np.delete(np.unique(layer.data), 0)
    sequentialset = np.arange(1, uniquelabels.max() + 1)
    # find next sequence
    missing_sequence = np.setdiff1d(sequentialset, uniquelabels)
    # if there are not missing sequence
    if len(missing_sequence) > 0:
        return missing_sequence.min()
    else:
        # otherwise, increment label by 1
        return uniquelabels.max() + 1
