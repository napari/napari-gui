import numpy as np
import pytest
from numpy import array

from napari.layers.base import Layer


@pytest.mark.parametrize(
    ('dims', 'nworld', 'nshape', 'expected'),
    [
        ([1, 2, 3], 4, 4, [1, 2, 3]),
        ([2, 1, 0, 3], 4, 2, [0, 1]),
        ([2, 1, 0, 3], 4, 3, [1, 0, 2]),
        ([2, 1, 0, 3], 4, 4, [2, 1, 0, 3]),
        ([0, 1, 2, 3, 4, 5, 6, 7], 4, 4, [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_world_to_layer(dims, nworld, nshape, expected):
    assert np.array_equal(
        Layer._world_to_layer_dims_impl(array(dims), nworld, nshape), expected
    )
