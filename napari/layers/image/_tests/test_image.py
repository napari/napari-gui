import dask.array as da
import numpy as np
import pytest
import xarray as xr

from napari._tests.utils import check_layer_world_data_extent
from napari.components.dims import Dims
from napari.layers import Image
from napari.layers.image._image_constants import ImageRendering
from napari.layers.utils.plane import ClippingPlaneList, SlicingPlane
from napari.utils import Colormap
from napari.utils.transforms.transform_utils import rotate_to_matrix


def test_random_image():
    """Test instantiating Image layer with random 2D data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer.multiscale is False
    assert layer._data_view.shape == shape[-2:]


def test_negative_image():
    """Test instantiating Image layer with negative data."""
    shape = (10, 15)
    np.random.seed(0)
    # Data between -1.0 and 1.0
    data = 2 * np.random.random(shape) - 1.0
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]

    # Data between -10 and 10
    data = 20 * np.random.random(shape) - 10
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_all_zeros_image():
    """Test instantiating Image layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=float)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_integer_image():
    """Test instantiating Image layer with integer data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.round(10 * np.random.random(shape)).astype(int)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_bool_image():
    """Test instantiating Image layer with bool data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=bool)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_3D_image():
    """Test instantiating Image layer with random 3D data."""
    shape = (10, 15, 6)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_3D_image_shape_1():
    """Test instantiating Image layer with random 3D data with shape 1 axis."""
    shape = (1, 10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_4D_image():
    """Test instantiating Image layer with random 4D data."""
    shape = (10, 15, 6, 8)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_5D_image_shape_1():
    """Test instantiating Image layer with random 5D data with shape 1 axis."""
    shape = (4, 1, 2, 10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


def test_rgb_image():
    """Test instantiating Image layer with RGB data."""
    shape = (10, 15, 3)
    np.random.seed(0)
    data = np.random.randint(0, 255, shape, np.uint8)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape) - 1
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape[:-1]]
    )
    assert layer.rgb is True
    assert layer._data_view.shape == shape[-3:]


def test_rgba_image():
    """Test instantiating Image layer with RGBA data."""
    shape = (10, 15, 4)
    np.random.seed(0)
    data = np.random.randint(0, 255, shape, np.uint8)
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape) - 1
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape[:-1]]
    )
    assert layer.rgb is True
    assert layer._data_view.shape == shape[-3:]


def test_negative_rgba_image():
    """Test instantiating Image layer with negative RGBA data."""
    shape = (10, 15, 4)
    np.random.seed(0)
    # Data between -1.0 and 1.0
    data = 2 * np.random.random(shape) - 1
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape) - 1
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape[:-1]]
    )
    assert layer.rgb is True
    assert layer._data_view.shape == shape[-3:]

    # Data between -10 and 10
    data = 20 * np.random.random(shape) - 10
    layer = Image(data)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape) - 1
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape[:-1]]
    )
    assert layer.rgb is True
    assert layer._data_view.shape == shape[-3:]


def test_non_rgb_image():
    """Test forcing Image layer to be 3D and not rgb."""
    shape = (10, 15, 3)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = Image(data, rgb=False)
    assert np.array_equal(layer.data, data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1], [s - 1 for s in shape])
    assert layer.rgb is False
    assert layer._data_view.shape == shape[-2:]


@pytest.mark.parametrize("shape", [(10, 15, 6), (10, 10)])
def test_error_non_rgb_image(shape):
    """Test error on trying non rgb as rgb."""
    # If rgb is set to be True in constructor but the last dim has a
    # size > 4 or ndim not >= 3 then data cannot actually be rgb
    data = np.empty(shape)
    with pytest.raises(ValueError, match="'rgb' was set to True but"):
        Image(data, rgb=True)


def test_changing_image():
    """Test changing Image data."""
    shape_a = (10, 15)
    shape_b = (20, 12)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_b = np.random.random(shape_b)
    layer = Image(data_a)
    layer.data = data_b
    assert np.array_equal(layer.data, data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape_b]
    )
    assert layer.rgb is False
    assert layer._data_view.shape == shape_b[-2:]


def test_changing_image_dims():
    """Test changing Image data including dimensionality."""
    shape_a = (10, 15)
    shape_b = (20, 12, 6)
    np.random.seed(0)
    data_a = np.random.random(shape_a)
    data_b = np.random.random(shape_b)
    layer = Image(data_a)

    # Prep indices for switch to 3D
    layer.data = data_b
    assert np.array_equal(layer.data, data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shape_b]
    )
    assert layer.rgb is False
    assert layer._data_view.shape == shape_b[-2:]


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.name == 'Image'

    layer = Image(data, name='random')
    assert layer.name == 'random'

    layer.name = 'img'
    assert layer.name == 'img'


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Image(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.opacity == 1

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Image(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Image(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'

    layer.blending = 'minimum'
    assert layer.blending == 'minimum'


def test_interpolation():
    """Test setting image interpolation mode."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    with pytest.deprecated_call():
        assert layer.interpolation == 'nearest'
    assert layer.interpolation2d == 'nearest'
    assert layer.interpolation3d == 'linear'

    with pytest.deprecated_call():
        layer = Image(data, interpolation2d='bicubic')
    assert layer.interpolation2d == 'cubic'
    with pytest.deprecated_call():
        assert layer.interpolation == 'cubic'

    layer.interpolation2d = 'linear'
    assert layer.interpolation2d == 'linear'
    with pytest.deprecated_call():
        assert layer.interpolation == 'linear'


def test_colormaps():
    """Test setting test_colormaps."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.colormap.name == 'gray'
    assert isinstance(layer.colormap, Colormap)

    layer.colormap = 'magma'
    assert layer.colormap.name == 'magma'
    assert isinstance(layer.colormap, Colormap)

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer.colormap = 'custom', cmap
    assert layer.colormap.name == 'custom'
    assert layer.colormap == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer.colormap = {'new': cmap}
    assert layer.colormap.name == 'new'
    assert layer.colormap == cmap

    layer = Image(data, colormap='magma')
    assert layer.colormap.name == 'magma'
    assert isinstance(layer.colormap, Colormap)

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.2, 1.0]])
    layer = Image(data, colormap=('custom', cmap))
    assert layer.colormap.name == 'custom'
    assert layer.colormap == cmap

    cmap = Colormap([[0.0, 0.0, 0.0, 0.0], [0.7, 0.2, 0.6, 1.0]])
    layer = Image(data, colormap={'new': cmap})
    assert layer.colormap.name == 'new'
    assert layer.colormap == cmap


def test_contrast_limits():
    """Test setting color limits."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.contrast_limits[0] >= 0
    assert layer.contrast_limits[1] <= 1
    assert layer.contrast_limits[0] < layer.contrast_limits[1]
    assert layer.contrast_limits == layer.contrast_limits_range

    # Change contrast_limits property
    contrast_limits = [0, 2]
    layer.contrast_limits = contrast_limits
    assert layer.contrast_limits == contrast_limits
    assert layer.contrast_limits_range == contrast_limits

    # Set contrast_limits as keyword argument
    layer = Image(data, contrast_limits=contrast_limits)
    assert layer.contrast_limits == contrast_limits
    assert layer.contrast_limits_range == contrast_limits


def test_contrast_limits_range():
    """Test setting color limits range."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.contrast_limits_range[0] >= 0
    assert layer.contrast_limits_range[1] <= 1
    assert layer.contrast_limits_range[0] < layer.contrast_limits_range[1]

    # If all data is the same value the contrast_limits_range and
    # contrast_limits defaults to [0, 1]
    data = np.zeros((10, 15))
    layer = Image(data)
    assert layer.contrast_limits_range == [0, 1]
    assert layer.contrast_limits == [0.0, 1.0]


def test_set_contrast_limits_range():
    """Test setting color limits range."""
    np.random.seed(0)
    data = np.random.random((10, 15)) * 100
    layer = Image(data)
    layer.contrast_limits_range = [0, 100]
    layer.contrast_limits = [20, 40]
    assert layer.contrast_limits_range == [0, 100]
    assert layer.contrast_limits == [20, 40]

    # clim values should stay within the contrast limits range
    layer.contrast_limits_range = [0, 30]
    assert layer.contrast_limits == [20, 30]
    # setting clim range outside of clim should override clim
    layer.contrast_limits_range = [0, 10]
    assert layer.contrast_limits == [0, 10]

    # in both directions...
    layer.contrast_limits_range = [0, 100]
    layer.contrast_limits = [20, 40]
    layer.contrast_limits_range = [60, 100]
    assert layer.contrast_limits == [60, 100]


@pytest.mark.parametrize(
    'contrast_limits_range',
    (
        [-2, -1],  # range below lower boundary of [0, 1]
        [-1, 0],  # range on lower boundary of [0, 1]
        [1, 2],  # range on upper boundary of [0, 1]
        [2, 3],  # range above upper boundary of [0, 1]
    ),
)
def test_set_contrast_limits_range_at_boundary_of_contrast_limits(
    contrast_limits_range,
):
    """See https://github.com/napari/napari/issues/5257"""
    layer = Image(np.zeros((6, 5)), contrast_limits=[0, 1])
    layer.contrast_limits_range = contrast_limits_range
    assert layer.contrast_limits == contrast_limits_range


def test_gamma():
    """Test setting gamma."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.gamma == 1

    # Change gamma property
    gamma = 0.7
    layer.gamma = gamma
    assert layer.gamma == gamma

    # Set gamma as keyword argument
    layer = Image(data, gamma=gamma)
    assert layer.gamma == gamma


def test_rendering():
    """Test setting rendering."""
    np.random.seed(0)
    data = np.random.random((20, 10, 15))
    layer = Image(data)
    assert layer.rendering == 'mip'

    # Change rendering property
    layer.rendering = 'translucent'
    assert layer.rendering == 'translucent'

    # Change rendering property
    layer.rendering = 'attenuated_mip'
    assert layer.rendering == 'attenuated_mip'

    # Change rendering property
    layer.rendering = 'iso'
    assert layer.rendering == 'iso'

    # Change rendering property
    layer.rendering = 'additive'
    assert layer.rendering == 'additive'


def test_iso_threshold():
    """Test setting iso_threshold."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert np.min(data) <= layer.iso_threshold <= np.max(data)

    # Change iso_threshold property
    iso_threshold = 0.7
    layer.iso_threshold = iso_threshold
    assert layer.iso_threshold == iso_threshold

    # Set iso_threshold as keyword argument
    layer = Image(data, iso_threshold=iso_threshold)
    assert layer.iso_threshold == iso_threshold


def test_attenuation():
    """Test setting attenuation."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.attenuation == 0.05

    # Change attenuation property
    attenuation = 0.07
    layer.attenuation = attenuation
    assert layer.attenuation == attenuation

    # Set attenuation as keyword argument
    layer = Image(data, attenuation=attenuation)
    assert layer.attenuation == attenuation


def test_metadata():
    """Test setting image metadata."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    assert layer.metadata == {}

    layer = Image(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    value = layer.get_value((0,) * 2)
    assert value == data[0, 0]


@pytest.mark.parametrize(
    'position,view_direction,dims_displayed,world',
    [
        ((0, 0, 0), [1, 0, 0], [0, 1, 2], False),
        ((0, 0, 0), [1, 0, 0], [0, 1, 2], True),
        ((0, 0, 0, 0), [0, 1, 0, 0], [1, 2, 3], True),
    ],
)
def test_value_3d(position, view_direction, dims_displayed, world):
    """Currently get_value should return None in 3D"""
    np.random.seed(0)
    data = np.random.random((10, 15, 15))
    layer = Image(data)
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    value = layer.get_value(
        position,
        view_direction=view_direction,
        dims_displayed=dims_displayed,
        world=world,
    )
    assert value is None


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    msg = layer.get_status((0,) * 2)
    assert isinstance(msg, dict)


def test_message_3d():
    """Test converting values and coords to message in 3D."""
    np.random.seed(0)
    data = np.random.random((10, 15, 15))
    layer = Image(data)
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    msg = layer.get_status(
        (0, 0, 0), view_direction=[1, 0, 0], dims_displayed=[0, 1, 2]
    )
    assert isinstance(msg, dict)


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.random((30, 30))
    layer = Image(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_narrow_thumbnail():
    """Ensure that the thumbnail generation works for very narrow images.

    See: https://github.com/napari/napari/issues/641 and
    https://github.com/napari/napari/issues/489
    """
    image = np.random.random((1, 2048))
    layer = Image(image)
    layer._update_thumbnail()
    thumbnail = layer.thumbnail[..., :3]  # ignore alpha channel
    middle_row = thumbnail.shape[0] // 2
    assert np.array_equiv(thumbnail[: middle_row - 1], 0)
    assert np.array_equiv(thumbnail[middle_row + 1 :], 0)
    assert np.mean(thumbnail[middle_row - 1 : middle_row + 1]) > 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_out_of_range_image(dtype):
    data = -1.7 - 0.001 * np.random.random((10, 15)).astype(dtype)
    layer = Image(data)
    layer._update_thumbnail()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_out_of_range_no_contrast(dtype):
    data = np.full((10, 15), -3.2, dtype=dtype)
    layer = Image(data)
    layer._update_thumbnail()


@pytest.mark.parametrize(
    "scale",
    [
        (None),
        ([1, 1]),
        (np.array([1, 1])),
        (da.from_array([1, 1], chunks=1)),
        (da.from_array([1, 1], chunks=2)),
        (xr.DataArray(np.array([1, 1]))),
        (xr.DataArray(np.array([1, 1]), dims=('dimension_name'))),
    ],
)
def test_image_scale(scale):
    np.random.seed(0)
    data = np.random.random((10, 15))
    Image(data, scale=scale)


@pytest.mark.parametrize(
    "translate",
    [
        (None),
        ([1, 1]),
        (np.array([1, 1])),
        (da.from_array([1, 1], chunks=1)),
        (da.from_array([1, 1], chunks=2)),
        (xr.DataArray(np.array([1, 1]))),
        (xr.DataArray(np.array([1, 1]), dims=('dimension_name'))),
    ],
)
def test_image_translate(translate):
    np.random.seed(0)
    data = np.random.random((10, 15))
    Image(data, translate=translate)


def test_image_scale_broadcast():
    """Test scale is broadcast."""
    data = np.random.random((5, 10, 15))
    layer = Image(data, scale=(2, 2))
    np.testing.assert_almost_equal(layer.scale, (1, 2, 2))


def test_image_translate_broadcast():
    """Test translate is broadcast."""
    data = np.random.random((5, 10, 15))
    layer = Image(data, translate=(2, 2))
    np.testing.assert_almost_equal(layer.translate, (0, 2, 2))


def test_grid_translate():
    np.random.seed(0)
    data = np.random.random((10, 15))
    layer = Image(data)
    translate = np.array([15, 15])
    layer._translate_grid = translate
    np.testing.assert_allclose(layer._translate_grid, translate)


def test_world_data_extent():
    """Test extent after applying transforms."""
    np.random.seed(0)
    shape = (6, 10, 15)
    data = np.random.random(shape)
    layer = Image(data)
    extent = np.array(((0,) * 3, [s - 1 for s in shape]))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


def test_data_to_world_2d_scale_translate_affine_composed():
    data = np.ones((4, 3))
    scale = (3, 2)
    translate = (-4, 8)
    affine = [[4, 0, 0], [0, 1.5, 0], [0, 0, 1]]

    image = Image(data, scale=scale, translate=translate, affine=affine)

    np.testing.assert_array_equal(image.scale, scale)
    np.testing.assert_array_equal(image.translate, translate)
    np.testing.assert_array_equal(image.affine, affine)
    np.testing.assert_almost_equal(
        image._data_to_world.affine_matrix,
        ((12, 0, -16), (0, 3, 12), (0, 0, 1)),
    )


@pytest.mark.parametrize('scale', ((1, 1), (-1, 1), (1, -1), (-1, -1)))
@pytest.mark.parametrize('angle_degrees', range(-180, 180, 30))
def test_rotate_with_reflections_in_scale(scale, angle_degrees):
    # See the GitHub issue for more details:
    # https://github.com/napari/napari/issues/2984
    data = np.ones((4, 3))
    rotate = rotate_to_matrix(angle_degrees, ndim=2)

    image = Image(data, scale=scale, rotate=rotate)

    np.testing.assert_array_equal(image.scale, scale)
    np.testing.assert_array_equal(image.rotate, rotate)


def test_2d_image_with_channels_and_2d_scale_translate_then_scale_translate_padded():
    # See the GitHub issue for more details:
    # https://github.com/napari/napari/issues/2973
    image = Image(np.ones((20, 20, 2)), scale=(1, 1), translate=(3, 4))

    np.testing.assert_array_equal(image.scale, (1, 1, 1))
    np.testing.assert_array_equal(image.translate, (0, 3, 4))


@pytest.mark.parametrize('affine_size', range(3, 6))
def test_2d_image_with_channels_and_affine_broadcasts(affine_size):
    # For more details, see the GitHub issue:
    # https://github.com/napari/napari/issues/3045
    image = Image(np.ones((1, 1, 1, 100, 100)), affine=np.eye(affine_size))
    np.testing.assert_array_equal(image.affine, np.eye(6))


@pytest.mark.parametrize('affine_size', range(3, 6))
def test_2d_image_with_channels_and_affine_assignment_broadcasts(affine_size):
    # For more details, see the GitHub issue:
    # https://github.com/napari/napari/issues/3045
    image = Image(np.ones((1, 1, 1, 100, 100)))
    image.affine = np.eye(affine_size)
    np.testing.assert_array_equal(image.affine, np.eye(6))


def test_image_state_update():
    """Test that an image can be updated from the output of its
    _get_state method()
    """
    image = Image(np.ones((32, 32, 32)))
    state = image._get_state()
    for k, v in state.items():
        setattr(image, k, v)


def test_instantiate_with_plane_parameter_dict():
    """Test that an image layer can be instantiated with plane parameters
    in a dictionary.
    """
    plane_parameters = {
        'position': (32, 32, 32),
        'normal': (1, 1, 1),
        'thickness': 22,
    }
    image = Image(np.ones((32, 32, 32)), plane=plane_parameters)
    for k, v in plane_parameters.items():
        if k == 'normal':
            v = tuple(v / np.linalg.norm(v))
        assert v == getattr(image.plane, k, v)


def test_instiantiate_with_plane():
    """Test that an image layer can be instantiated with plane parameters
    in a Plane.
    """
    plane = SlicingPlane(position=(32, 32, 32), normal=(1, 1, 1), thickness=22)
    image = Image(np.ones((32, 32, 32)), plane=plane)
    for k, v in plane.dict().items():
        assert v == getattr(image.plane, k, v)


def test_instantiate_with_clipping_planelist():
    planes = ClippingPlaneList.from_array(np.ones((2, 2, 3)))
    image = Image(np.ones((32, 32, 32)), experimental_clipping_planes=planes)
    assert len(image.experimental_clipping_planes) == 2


def test_instantiate_with_experimental_clipping_planes_dict():
    planes = [
        {'position': (0, 0, 0), 'normal': (0, 0, 1)},
        {'position': (0, 1, 0), 'normal': (1, 0, 0)},
    ]
    image = Image(np.ones((32, 32, 32)), experimental_clipping_planes=planes)
    for i in range(len(planes)):
        assert (
            image.experimental_clipping_planes[i].position
            == planes[i]['position']
        )
        assert (
            image.experimental_clipping_planes[i].normal == planes[i]['normal']
        )


def test_tensorstore_image():
    """Test an image coming from a tensorstore array."""
    ts = pytest.importorskip('tensorstore')

    data = ts.array(
        np.full(shape=(1024, 1024), fill_value=255, dtype=np.uint8)
    )
    layer = Image(data)
    assert np.array_equal(layer.data, data)


@pytest.mark.parametrize(
    "start_position, end_position, view_direction, vector, expected_value",
    [
        # drag vector parallel to view direction
        # projected onto perpendicular vector
        ([0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], 0),
        # same as above, projection onto multiple perpendicular vectors
        # should produce multiple results
        ([0, 0, 0], [0, 0, 1], [0, 0, 1], [[1, 0, 0], [0, 1, 0]], [0, 0]),
        # drag vector perpendicular to view direction
        # projected onto itself
        ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], 1),
        # drag vector perpendicular to view direction
        # projected onto itself
        ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], 1),
    ],
)
def test_projected_distance_from_mouse_drag(
    start_position, end_position, view_direction, vector, expected_value
):
    image = Image(np.ones((32, 32, 32)))
    image._slice_dims(Dims(ndim=3, ndisplay=3))
    result = image.projected_distance_from_mouse_drag(
        start_position,
        end_position,
        view_direction,
        vector,
        dims_displayed=[0, 1, 2],
    )
    assert np.allclose(result, expected_value)


def test_rendering_init():
    np.random.seed(0)
    data = np.random.rand(10, 10, 10)
    layer = Image(data, rendering='iso')

    assert layer.rendering == ImageRendering.ISO.value


def test_thick_slice():
    data = np.ones((5, 5, 5)) * np.arange(5).reshape(-1, 1, 1)
    layer = Image(data)

    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    np.testing.assert_array_equal(layer._slice.image.raw, data[0])

    # round down if at 0.5 and no margins
    layer._slice_dims(Dims(ndim=3, point=(0.5, 0, 0)))
    np.testing.assert_array_equal(layer._slice.image.raw, data[0])

    # no changes if projection mode is 'none'
    layer._slice_dims(
        Dims(
            ndim=3,
            point=(0, 0, 0),
            margin_left=(1, 0, 0),
            margin_right=(1, 0, 0),
        )
    )
    np.testing.assert_array_equal(layer._slice.image.raw, data[0])

    layer.projection_mode = 'mean'
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.mean(data[:2], axis=0)
    )

    layer._slice_dims(
        Dims(
            ndim=3,
            point=(1, 0, 0),
            margin_left=(1, 0, 0),
            margin_right=(1, 0, 0),
        )
    )
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.mean(data[:3], axis=0)
    )

    layer._slice_dims(
        Dims(
            ndim=3,
            range=((0, 3, 1), (0, 2, 1), (0, 2, 1)),
            point=(2.3, 0, 0),
            margin_left=(0, 0, 0),
            margin_right=(1.7, 0, 0),
        )
    )
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.mean(data[2:5], axis=0)
    )

    layer._slice_dims(
        Dims(
            ndim=3,
            range=((0, 3, 1), (0, 2, 1), (0, 2, 1)),
            point=(2.3, 0, 0),
            margin_left=(0, 0, 0),
            margin_right=(1.6, 0, 0),
        )
    )
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.mean(data[2:4], axis=0)
    )

    layer.projection_mode = 'max'
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.max(data[2:4], axis=0)
    )


def test_thick_slice_multiscale():
    data = np.ones((5, 5, 5)) * np.arange(5).reshape(-1, 1, 1)
    data_zoom = data.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    layer = Image([data_zoom, data])

    # ensure we're slicing level 0. We also need to update corner_pixels
    # to ensure the full image is in view
    layer.corner_pixels = np.array([[0, 0, 0], [10, 10, 10]])
    layer.data_level = 0

    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    np.testing.assert_array_equal(layer._slice.image.raw, data_zoom[0])

    layer.projection_mode = 'mean'
    # NOTE that here we rescale slicing to twice the non-multiscale test
    # in order to get the same results, becase the actual full scale image
    # is doubled in size
    layer._slice_dims(
        Dims(
            ndim=3,
            range=((0, 5, 1), (0, 2, 1), (0, 2, 1)),
            point=(4.6, 0, 0),
            margin_left=(0, 0, 0),
            margin_right=(3.4, 0, 0),
        )
    )
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.mean(data_zoom[4:10], axis=0)
    )

    # check level 1
    layer.corner_pixels = np.array([[0, 0, 0], [5, 5, 5]])
    layer.data_level = 1

    layer._slice_dims(Dims(ndim=3, point=(0, 0, 0)))
    np.testing.assert_array_equal(layer._slice.image.raw, data[0])

    layer.projection_mode = 'mean'
    # here we slice in the same point as earlier, but to get the expected value
    # we need to slice `data` with halved indices
    layer._slice_dims(
        Dims(
            ndim=3,
            range=((0, 5, 1), (0, 2, 1), (0, 2, 1)),
            point=(4.6, 0, 0),
            margin_left=(0, 0, 0),
            margin_right=(3.4, 0, 0),
        )
    )
    np.testing.assert_array_equal(
        layer._slice.image.raw, np.mean(data[2:5], axis=0)
    )


def test_chunk_pixel_threshold_reset_contrast_lims(monkeypatch):
    monkeypatch.setattr("napari.layers.utils.layer_utils.PIXEL_THRESHOLD", 100)
    data = da.random.random((200, 200), chunks=(100, 100))
    layer = Image(data)
    layer.reset_contrast_limits()
    assert layer.contrast_limits
    layer.reset_contrast_limits_range()
    assert layer.contrast_limits_range

    with pytest.warns(UserWarning):
        layer.reset_contrast_limits(mode='data')
        assert layer.contrast_limits
    with pytest.warns(UserWarning):
        layer.reset_contrast_limits_range(mode='data')
        assert layer.contrast_limits_range
