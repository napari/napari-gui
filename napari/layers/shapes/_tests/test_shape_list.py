import numpy as np
import numpy.testing as npt
import pytest

from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_models import Path, Polygon, Rectangle


def test_empty_shape_list():
    """Test instantiating empty ShapeList."""
    shape_list = ShapeList()
    assert len(shape_list.shapes) == 0


def test_adding_to_shape_list():
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)
    assert len(shape_list.shapes) == 1
    assert shape_list.shapes[0] == shape


def test_reset_bounding_box_rotation():
    """Test if rotating shape resets bounding box."""
    shape = Rectangle(np.array([[0, 0], [10, 10]]))
    shape_list = ShapeList()
    shape_list.add(shape)
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, np.array([[[-0.5, -0.5]], [[10.5, 10.5]]])
    )
    shape_list.rotate(0, 45, (5, 5))
    p = 5 * np.sqrt(2) + 0.5
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[5 - p, 5 - p], [5 + p, 5 + p]])
    )
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )


def test_reset_bounding_box_shift():
    """Test if shifting shape resets bounding box."""
    shape = Rectangle(np.array([[0, 0], [10, 10]]))
    shape_list = ShapeList()
    shape_list.add(shape)
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )
    shape_list.shift(0, np.array([5, 5]))
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[4.5, 4.5], [15.5, 15.5]])
    )
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )


def test_reset_bounding_box_scale():
    """Test if scaling shape resets the bounding box."""
    shape = Rectangle(np.array([[0, 0], [10, 10]]))
    shape_list = ShapeList()
    shape_list.add(shape)
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )
    shape_list.scale(0, 2, (5, 5))
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[-5.5, -5.5], [15.5, 15.5]])
    )
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )


def test_shape_list_outline():
    """Test ShapeList outline method."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)

    # Check passing an int
    outline_by_index = shape_list.outline(0)
    assert isinstance(outline_by_index, tuple)

    # Check passing a list
    outline_by_index_list = shape_list.outline([0])
    assert isinstance(outline_by_index_list, tuple)

    # Check return value for `int` and `list` are the same
    for value_by_idx, value_by_idx_list in zip(
        outline_by_index, outline_by_index_list, strict=False
    ):
        assert np.array_equal(value_by_idx, value_by_idx_list)

    # Check passing a `numpy.int_` (`numpy.int32/64` depending on platform)
    outline_by_index_np = shape_list.outline(np.int_(0))
    assert isinstance(outline_by_index_np, tuple)

    # Check return value for `int` and `numpy.int_` are the same
    for value_by_idx, value_by_idx_np in zip(
        outline_by_index, outline_by_index_np, strict=False
    ):
        assert np.array_equal(value_by_idx, value_by_idx_np)


def test_shape_list_outline_two_shapes():
    shape1 = Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    shape2 = Polygon([[20, 20], [20, 30], [30, 30], [30, 20]])
    shape_list = ShapeList()
    shape_list.add([shape1, shape2])

    # check if the outline contains triangle with vertex of number 16

    triangles = shape_list.outline([0, 1])[2]
    assert np.any(triangles == 16)


def test_nD_shapes():
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    shape_list = ShapeList()
    data = 20 * np.random.random((6, 3))
    data[:, 0] = 0
    shape_a = Polygon(data)
    shape_list.add(shape_a)

    data = 20 * np.random.random((6, 3))
    data[:, 0] = 1
    shape_b = Path(data)
    shape_list.add(shape_b)

    assert len(shape_list.shapes) == 2
    assert shape_list.shapes[0] == shape_a
    assert shape_list.shapes[1] == shape_b

    assert shape_list._vertices.shape[1] == 2
    assert shape_list._mesh.vertices.shape[1] == 2

    shape_list.ndisplay = 3
    assert shape_list._vertices.shape[1] == 3
    assert shape_list._mesh.vertices.shape[1] == 3


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_bad_color_array(attribute):
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)

    # test setting color with a color array of the wrong shape
    bad_color_array = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
    with pytest.raises(ValueError, match='must have shape'):
        setattr(shape_list, f'{attribute}_color', bad_color_array)


def test_inside():
    shape1 = Polygon(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]))
    shape2 = Polygon(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]))
    shape3 = Polygon(np.array([[2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 0, 1]]))

    shape_list = ShapeList()
    shape_list.add([shape1, shape2, shape3])
    shape_list.slice_key = (1,)
    assert shape_list.inside((0.5, 0.5)) == 1
