from itertools import permutations

import numpy as np
import pytest

from napari.utils.transforms.transform_utils import (
    compose_linear_matrix,
    decompose_linear_matrix,
    get_permutation,
    is_diagonal,
    is_matrix_lower_triangular,
    is_matrix_triangular,
    is_matrix_upper_triangular,
    is_permutation,
    shear_matrix_from_angle,
)


@pytest.mark.parametrize('upper_triangular', [True, False])
def test_decompose_linear_matrix(upper_triangular):
    """Test composing and decomposing a linear matrix."""
    np.random.seed(0)

    # Decompose linear matrix
    A = np.random.random((2, 2))
    rotate, scale, shear = decompose_linear_matrix(
        A, upper_triangular=upper_triangular
    )

    # Compose linear matrix and check it matches
    B = compose_linear_matrix(rotate, scale, shear)
    np.testing.assert_almost_equal(A, B)

    # Decompose linear matrix and check it matches
    rotate_B, scale_B, shear_B = decompose_linear_matrix(
        B, upper_triangular=upper_triangular
    )
    np.testing.assert_almost_equal(rotate, rotate_B)
    np.testing.assert_almost_equal(scale, scale_B)
    np.testing.assert_almost_equal(shear, shear_B)

    # Compose linear matrix and check it matches
    C = compose_linear_matrix(rotate_B, scale_B, shear_B)
    np.testing.assert_almost_equal(B, C)


def test_composition_order():
    """Test composition order."""
    # Order is rotate, shear, scale
    rotate = np.array([[0, -1], [1, 0]])
    shear = np.array([[1, 3], [0, 1]])
    scale = [2, 5]

    matrix = compose_linear_matrix(rotate, scale, shear)
    np.testing.assert_almost_equal(matrix, rotate @ shear @ np.diag(scale))


def test_shear_matrix_from_angle():
    """Test creating a shear matrix from an angle."""
    matrix = shear_matrix_from_angle(35)
    np.testing.assert_almost_equal(np.diag(matrix), [1] * 3)
    np.testing.assert_almost_equal(matrix[-1, 0], np.tan(np.deg2rad(55)))


upper = np.array([[1, 1], [0, 1]])
lower = np.array([[1, 0], [1, 1]])
full = np.array([[1, 1], [1, 1]])


def test_is_matrix_upper_triangular():
    """Test if a matrix is upper triangular."""
    assert is_matrix_upper_triangular(upper)
    assert not is_matrix_upper_triangular(lower)
    assert not is_matrix_upper_triangular(full)


def test_is_matrix_lower_triangular():
    """Test if a matrix is lower triangular."""
    assert not is_matrix_lower_triangular(upper)
    assert is_matrix_lower_triangular(lower)
    assert not is_matrix_lower_triangular(full)


def test_is_matrix_triangular():
    """Test if a matrix is triangular."""
    assert is_matrix_triangular(upper)
    assert is_matrix_triangular(lower)
    assert not is_matrix_triangular(full)


def test_is_diagonal():
    assert is_diagonal(np.eye(3))
    assert not is_diagonal(np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))

    # affine with tiny off-diagonal elements will be considered diagonal
    m = np.full((3, 3), 1e-10)
    m[0, 0] = 1
    m[1, 1] = 1
    m[2, 2] = 1
    assert is_diagonal(m)

    # will be considered non-diagonal with stricter tolerance
    assert not is_diagonal(m, tol=1e-12)


@pytest.mark.parametrize('ndim', [2, 3, 4])
@pytest.mark.parametrize('tol', [1e-8, 0.0])
@pytest.mark.parametrize('off_diag_value', [0.0, 1e-12, 1e-3])
def test_is_permutation(ndim, off_diag_value, tol):
    # empty affine matrix
    blank = np.full((ndim + 1, ndim + 1), off_diag_value, dtype=float)
    blank[-1, -1] = 1
    # set non-zero translations
    blank[:ndim, -1] = np.random.randn(ndim)

    within_tolerance = off_diag_value <= tol

    # test all possible axis permutations
    perm_diag = tuple(range(ndim))
    for perm in tuple(permutations(range(ndim), ndim)):
        # initialize the affine with randomized positive scale values
        affine = blank.copy()
        for r, c in enumerate(perm):
            affine[r, c] = 0.1 + abs(np.random.randn(1))

        linear_matrix = affine[:ndim, :ndim]
        if perm == perm_diag:
            assert is_diagonal(linear_matrix, tol=tol) == within_tolerance
            assert not is_permutation(
                linear_matrix, tol=tol, exclude_diagonal=True
            )
        is_perm = is_permutation(linear_matrix, tol=tol)
        assert is_perm == within_tolerance
        # test that get_permutation returns the expected axis order
        if is_perm:
            assert perm == get_permutation(linear_matrix, tol=tol)
        else:
            assert get_permutation(linear_matrix, tol=tol) is None
