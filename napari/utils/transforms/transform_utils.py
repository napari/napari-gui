import numpy as np
import scipy.linalg

from napari.utils.translations import trans


def compose_linear_matrix(rotate, scale, shear) -> np.array:
    """Compose linear transform matrix from rotate, shear, scale.

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.

    Returns
    -------
    matrix : array
        nD array representing the composed linear transform.
    """
    rotate_mat = _make_rotate_mat(rotate)
    scale_mat = np.diag(scale)
    shear_mat = _make_shear_mat(shear)

    ndim = max(mat.shape[0] for mat in (rotate_mat, scale_mat, shear_mat))
    full_scale = embed_in_identity_matrix(scale_mat, ndim)
    full_rotate = embed_in_identity_matrix(rotate_mat, ndim)
    full_shear = embed_in_identity_matrix(shear_mat, ndim)

    return full_rotate @ full_shear @ full_scale


def infer_ndim(
    *, scale=None, translate=None, rotate=None, shear=None, linear_matrix=None
):
    """Infer the dimensionality of a transformation from its input components.

    This is most useful when the dimensions of the inputs do not match, either
    when broadcasting is desired or when an input represents a parameterization
    of a transform component (e.g. rotate as an angle of a 2D rotation).

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.

    Returns
    -------
    ndim : int
        The maximum dimensionality of the component inputs.
    """
    ndim = 0
    if scale is not None:
        ndim = max(ndim, len(scale))
    if translate is not None:
        ndim = max(ndim, len(translate))
    if rotate is not None:
        ndim = max(ndim, _make_rotate_mat(rotate).shape[0])
    if shear is not None:
        ndim = max(ndim, _make_shear_mat(shear).shape[0])
    return ndim


def translate_to_vector(translate, *, ndim):
    """Convert a translate input into an n-dimensional transform component.

    Parameters
    ----------
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is padded
        with 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty vector implies no translation.
    ndim : int
        The desired dimensionality of the output transform component.

    Returns
    -------
    np.ndarray
        The translate component as a 1D numpy array of length ndim.
    """
    translate_arr = np.zeros(ndim)
    if translate is not None:
        if len(translate) > ndim:
            translate_arr[:] = translate[:ndim]
        else:
            translate_arr[-len(translate) :] = translate
    return translate_arr


def scale_to_vector(scale, *, ndim):
    """Convert a scale input into an n-dimensional transform component.

    Parameters
    ----------
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is padded with 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty vector implies no scaling.
    ndim : int
        The desired dimensionality of the output transform component.

    Returns
    -------
    np.ndarray
        The scale component as a 1D numpy array of length ndim.
    """
    scale_arr = np.ones(ndim)
    if scale is not None:
        scale_arr[-len(scale) :] = scale
    return scale_arr


def rotate_to_matrix(rotate, *, ndim):
    """Convert a rotate input into an n-dimensional transform component.

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    ndim : int
        The desired dimensionality of the output transform component.

    Returns
    -------
    np.ndarray
        The rotate component as a 2D numpy array of size ndim.
    """
    full_rotate_mat = np.eye(ndim)
    if rotate is not None:
        rotate_mat = _make_rotate_mat(rotate)
        rotate_mat_ndim = rotate_mat.shape[0]
        full_rotate_mat[-rotate_mat_ndim:, -rotate_mat_ndim:] = rotate_mat
    return full_rotate_mat


def _make_rotate_mat(rotate):
    if np.isscalar(rotate):
        return _make_2d_rotation(rotate)
    if np.array(rotate).ndim == 1 and len(rotate) == 3:
        return _make_3d_rotation(*rotate)
    return np.array(rotate)


def _make_2d_rotation(theta_degrees):
    """Makes a 2D rotation matrix from an angle in degrees."""
    cos_theta, sin_theta = _cos_sin_degrees(theta_degrees)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def _make_3d_rotation(alpha_degrees, beta_degrees, gamma_degrees):
    """Makes a 3D rotation matrix from roll, pitch, and yaw in degrees.

    For more details, see: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    """
    cos_alpha, sin_alpha = _cos_sin_degrees(alpha_degrees)
    R_alpha = np.array(
        [
            [cos_alpha, -sin_alpha, 0],
            [sin_alpha, cos_alpha, 0],
            [0, 0, 1],
        ]
    )

    cos_beta, sin_beta = _cos_sin_degrees(beta_degrees)
    R_beta = np.array(
        [
            [cos_beta, 0, sin_beta],
            [0, 1, 0],
            [-sin_beta, 0, cos_beta],
        ]
    )

    cos_gamma, sin_gamma = _cos_sin_degrees(gamma_degrees)
    R_gamma = np.array(
        [
            [1, 0, 0],
            [0, cos_gamma, -sin_gamma],
            [0, sin_gamma, cos_gamma],
        ]
    )

    return R_alpha @ R_beta @ R_gamma


def _cos_sin_degrees(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    return np.cos(angle_radians), np.sin(angle_radians)


def shear_to_matrix(shear, *, ndim):
    """Convert a shear input into an n-dimensional transform component.

    Parameters
    ----------
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    ndim : int
        The desired dimensionality of the output transform matrix.

    Returns
    -------
    np.ndarray
        The shear component as a triangular 2D numpy array of size ndim.
    """
    full_shear_mat = np.eye(ndim)
    if shear is not None:
        shear_mat = _make_shear_mat(shear)
        shear_mat_ndim = shear_mat.shape[0]
        full_shear_mat[-shear_mat_ndim:, -shear_mat_ndim:] = shear_mat
    return full_shear_mat


def _make_shear_mat(shear):
    # Check if an upper-triangular representation of shear or
    # a full nD shear matrix has been passed
    if np.isscalar(shear):
        raise ValueError(
            trans._(
                'Scalars are not valid values for shear. Shear must be an upper triangular vector or square matrix with ones along the main diagonal.',
                deferred=True,
            )
        )
    if np.array(shear).ndim == 1:
        return expand_upper_triangular(shear)

    if not is_matrix_triangular(shear):
        raise ValueError(
            trans._(
                'Only upper triangular or lower triangular matrices are accepted for shear, got {shear}. For other matrices, set the affine_matrix or linear_matrix directly.',
                deferred=True,
                shear=shear,
            )
        )
    return np.array(shear)


def expand_upper_triangular(vector):
    """Expand a vector into an upper triangular matrix.

    Decomposition is based on code from https://github.com/matthew-brett/transforms3d.
    In particular, the `striu2mat` function in the `shears` module.
    https://github.com/matthew-brett/transforms3d/blob/0.3.1/transforms3d/shears.py#L30-L77.

    Parameters
    ----------
    vector : np.array
        1D vector of length M

    Returns
    -------
    upper_tri : np.array shape (N, N)
        Upper triangular matrix.
    """
    n = len(vector)
    N = ((-1 + np.sqrt(8 * n + 1)) / 2.0) + 1  # n+1 th root
    if np.floor(N) != N:
        raise ValueError(
            trans._(
                '{number} is a strange number of shear elements',
                deferred=True,
                number=n,
            )
        )

    N = int(N)
    inds = np.triu(np.ones((N, N)), 1).astype(bool)
    upper_tri = np.eye(N)
    upper_tri[inds] = vector
    return upper_tri


def embed_in_identity_matrix(matrix, ndim):
    """Embed an MxM matrix bottom right of larger NxN identity matrix.

    Parameters
    ----------
    matrix : np.array
        2D square matrix, MxM.
    ndim : int
        Integer with N >= M.

    Returns
    -------
    full_matrix : np.array shape (N, N)
        Larger matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            trans._(
                'Improper transform matrix {matrix}',
                deferred=True,
                matrix=matrix,
            )
        )

    if matrix.shape[0] == ndim:
        return matrix

    full_matrix = np.eye(ndim)
    full_matrix[-matrix.shape[0] :, -matrix.shape[1] :] = matrix
    return full_matrix


def decompose_linear_matrix(
    matrix, upper_triangular=True
) -> (np.array, np.array, np.array):
    """Decompose linear transform matrix into rotate, scale, shear.

    Decomposition is based on code from https://github.com/matthew-brett/transforms3d.
    In particular, the `decompose` function in the `affines` module.
    https://github.com/matthew-brett/transforms3d/blob/0.3.1/transforms3d/affines.py#L156-L246.

    Parameters
    ----------
    matrix : np.array shape (N, N)
        nD array representing the composed linear transform.
    upper_triangular : bool
        Whether to decompose shear into an upper triangular or
        lower triangular matrix.

    Returns
    -------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array or n-D array
        1-D array of upper triangular values or an n-D matrix if lower
        triangular.
    """
    n = matrix.shape[0]

    if upper_triangular:
        rotate, tri = scipy.linalg.qr(matrix)
    else:
        upper_tri, rotate = scipy.linalg.rq(matrix.T)
        rotate = rotate.T
        tri = upper_tri.T

    scale = np.diag(tri).copy()

    # Take any reflection into account
    if np.linalg.det(rotate) < 0:
        scale[0] *= -1
        tri[0] *= -1
        rotate = matrix @ np.linalg.inv(tri)

    tri_normalized = tri @ np.linalg.inv(np.diag(scale))

    if upper_triangular:
        shear = tri_normalized[np.triu(np.ones((n, n)), 1).astype(bool)]
    else:
        shear = tri_normalized

    return rotate, scale, shear


def shear_matrix_from_angle(angle, ndim=3, axes=(-1, 0)):
    """Create a shear matrix from an angle.

    Parameters
    ----------
    angle : float
        Angle in degrees.
    ndim : int
        Dimensionality of the shear matrix
    axes : 2-tuple of int
        Location of the angle in the shear matrix.
        Default is the lower left value.

    Returns
    -------
    matrix : np.ndarray
        Shear matrix with ones along the main diagonal
    """
    matrix = np.eye(ndim)
    matrix[axes] = np.tan(np.deg2rad(90 - angle))
    return matrix


def is_matrix_upper_triangular(matrix):
    """Check if a matrix is upper triangular.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Whether matrix is upper triangular or not.
    """
    return np.allclose(matrix, np.triu(matrix))


def is_matrix_lower_triangular(matrix):
    """Check if a matrix is lower triangular.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Whether matrix is lower triangular or not.
    """
    return np.allclose(matrix, np.tril(matrix))


def is_matrix_triangular(matrix):
    """Check if a matrix is triangular.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Whether matrix is triangular or not.
    """
    return is_matrix_upper_triangular(matrix) or is_matrix_lower_triangular(
        matrix
    )


def is_diagonal(matrix, tol=1e-8):
    """Determine whether a matrix is diagonal up to some tolerance.

    Parameters
    ----------
    matrix : 2-D array
        The matrix to test.
    tol : float, optional
        Consider any entries with magnitude < `tol` as 0.

    Returns
    -------
    is_diag : bool
        True if matrix is diagonal, False otherwise.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            trans._(
                "matrix must be square, but shape={shape}",
                deferred=True,
                shape=matrix.shape,
            )
        )
    non_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if tol == 0:
        return np.count_nonzero(non_diag) == 0

    return np.max(np.abs(non_diag)) <= tol
