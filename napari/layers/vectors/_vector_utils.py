import numpy as np
import numpy.typing as npt

from napari.utils.translations import trans


def convert_image_to_coordinates(vectors: npt.NDArray) -> npt.NDArray:
    """To convert an image-like array with elements (y-proj, x-proj) into a
    position list of coordinates
    Every pixel position (n, m) results in two output coordinates of (N,2)

    Parameters
    ----------
    vectors : (N1, N2, ..., ND, D) array
        "image-like" data where there is a length D vector of the
        projections at each pixel.

    Returns
    -------
    coords : (N, 2, D) array
        A list of N vectors with start point and projections of the vector
        in D dimensions.
    """
    # create coordinate spacing for image
    spacing = [list(range(r)) for r in vectors.shape[:-1]]
    grid = np.meshgrid(*spacing)

    # create empty vector of necessary shape
    nvect = np.prod(vectors.shape[:-1])
    coords = np.empty((nvect, 2, vectors.ndim - 1), dtype=np.float32)

    # assign coordinates to all pixels
    for i, g in enumerate(grid):
        coords[:, 0, i] = g.flatten()
    coords[:, 1, :] = np.reshape(vectors, (-1, vectors.ndim - 1))

    return coords


def fix_data_vectors(
    vectors: np.ndarray | None, ndim: int | None
) -> tuple[np.ndarray, int]:
    """
    Ensure that vectors array is 3d and have second dimension of size 2
    and third dimension of size ndim (default 2 for empty arrays)

    Parameters
    ----------
    vectors : (N, 2, D) or (N1, N2, ..., ND, D) array
        A (N, 2, D) array is interpreted as "coordinate-like" data and a list
        of N vectors with start point and projections of the vector in D
        dimensions. A (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.
    ndim : int or None
        number of expected dimensions

    Returns
    -------
    vectors : (N, 2, D) array
        Vectors array
    ndim : int
        number of dimensions

    Raises
    ------
    ValueError
        if ndim does not match with third dimensions of vectors
    """
    if vectors is None:
        vectors = np.array([])
    vectors = np.asarray(vectors)

    if vectors.ndim == 3 and vectors.shape[1] == 2:
        # an (N, 2, D) array that is coordinate-like, we're good to go
        pass
    elif vectors.size == 0:
        if ndim is None:
            ndim = 2
        vectors = np.empty((0, 2, ndim))
    elif vectors.shape[-1] == vectors.ndim - 1:
        # an (N1, N2, ..., ND, D) array that is image-like
        vectors = convert_image_to_coordinates(vectors)
    else:
        # np.atleast_3d does not reshape (2, 3) to (1, 2, 3) as one would expect
        # when passing a single vector
        if vectors.ndim == 2:
            vectors = vectors[np.newaxis]
        if vectors.ndim != 3 or vectors.shape[1] != 2:
            raise ValueError(
                trans._(
                    'could not reshape Vector data from {vectors_shape} to (N, 2, {dimensions})',
                    deferred=True,
                    vectors_shape=vectors.shape,
                    dimensions=ndim or 'D',
                )
            )

    data_ndim = vectors.shape[2]
    if ndim is not None and ndim != data_ndim:
        raise ValueError(
            trans._(
                'Vectors dimensions ({data_ndim}) must be equal to ndim ({ndim})',
                deferred=True,
                data_ndim=data_ndim,
                ndim=ndim,
            )
        )
    return vectors, data_ndim
