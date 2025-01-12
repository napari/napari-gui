import dask.array as da
import numpy as np
import pytest

from napari.components import ViewerModel
from napari.utils.colormaps import (
    AVAILABLE_COLORMAPS,
    CYMRGB,
    MAGENTA_GREEN,
    SIMPLE_COLORMAPS,
    Colormap,
    ensure_colormap,
)
from napari.utils.misc import ensure_iterable, ensure_sequence_of_iterables

base_colormaps = CYMRGB
two_colormaps = MAGENTA_GREEN
green_cmap = SIMPLE_COLORMAPS['green']
red_cmap = SIMPLE_COLORMAPS['red']
blue_cmap = AVAILABLE_COLORMAPS['blue']
cmap_tuple = ('my_colormap', Colormap(['g', 'm', 'y']))
cmap_dict = {'your_colormap': Colormap(['g', 'r', 'y'])}

MULTI_TUPLES = [[0.3, 0.7], [0.1, 0.9], [0.3, 0.9], [0.4, 0.9], [0.2, 0.9]]

# data shape is (15, 10, 5) unless otherwise set
# channel_axis = -1 is implied unless otherwise set
multi_channel_test_data = [
    # basic multichannel image
    ((), {}),
    # single channel
    ((15, 10, 1), {}),
    # two channels
    ((15, 10, 2), {}),
    # Test adding multichannel image with color channel set.
    ((5, 10, 15), {'channel_axis': 0}),
    # split single RGB image
    ((15, 10, 3), {'colormap': ['red', 'green', 'blue']}),
    # multiple RGB images
    ((45, 40, 5, 3), {'channel_axis': 2, 'rgb': True}),
    # Test adding multichannel image with custom names.
    ((), {'name': ['multi ' + str(i + 3) for i in range(5)]}),
    # Test adding multichannel image with custom contrast limits.
    ((), {'contrast_limits': [0.3, 0.7]}),
    ((), {'contrast_limits': MULTI_TUPLES}),
    ((), {'gamma': 0.5}),
    ((), {'gamma': [0.3, 0.4, 0.5, 0.6, 0.7]}),
    ((), {'visible': [True, False, False, True, True]}),
    # Test adding multichannel image with custom colormaps.
    ((), {'colormap': 'gray'}),
    ((), {'colormap': green_cmap}),
    ((), {'colormap': cmap_tuple}),
    ((), {'colormap': cmap_dict}),
    ((), {'colormap': ['gray', 'blue', 'red', 'green', 'yellow']}),
    (
        (),
        {'colormap': [green_cmap, red_cmap, blue_cmap, blue_cmap, green_cmap]},
    ),
    ((), {'colormap': [green_cmap, 'gray', cmap_tuple, blue_cmap, cmap_dict]}),
    ((), {'scale': MULTI_TUPLES}),
    ((), {'translate': MULTI_TUPLES}),
    ((), {'blending': 'translucent'}),
    ((), {'metadata': {'hi': 'there'}}),
    ((), {'metadata': dict(MULTI_TUPLES)}),
    ((), {'experimental_clipping_planes': []}),
]

ids = [
    'basic_multichannel',
    'one_channel',
    'two_channel',
    'specified_multichannel',
    'split_RGB',
    'list_RGB',
    'names',
    'contrast_limits_broadcast',
    'contrast_limits_list',
    'gamma_broadcast',
    'gamma_list',
    'visibility',
    'colormap_string_broadcast',
    'colormap_cmap_broadcast',
    'colormap_tuple_broadcast',
    'colormap_dict_broadcast',
    'colormap_string_list',
    'colormap_cmap_list',
    'colormap_variable_list',
    'scale',
    'translate',
    'blending',
    'metadata_broadcast',
    'metadata_multi',
    'empty_clipping_planes',
]


@pytest.mark.parametrize(('shape', 'kwargs'), multi_channel_test_data, ids=ids)
def test_multichannel(shape, kwargs):
    """Test adding multichannel image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random(shape or (15, 10, 5))
    channel_axis = kwargs.pop('channel_axis', -1)
    viewer.add_image(data, channel_axis=channel_axis, **kwargs)

    # make sure the right number of layers got added
    n_channels = data.shape[channel_axis]
    assert len(viewer.layers) == n_channels

    for i in range(n_channels):
        # make sure that the data has been divided into layers
        np.testing.assert_array_equal(
            viewer.layers[i].data, data.take(i, axis=channel_axis)
        )
        # make sure colors have been assigned properly
        if 'colormap' not in kwargs:
            if n_channels == 1:
                assert viewer.layers[i].colormap.name == 'gray'
            elif n_channels == 2:
                assert viewer.layers[i].colormap.name == two_colormaps[i]
            else:
                assert viewer.layers[i].colormap.name == base_colormaps[i]
        if 'blending' not in kwargs:
            assert (
                viewer.layers[i].blending == 'translucent_no_depth'
                if i == 0
                else 'additive'
            )
        for key, expectation in kwargs.items():
            # broadcast exceptions
            if key in {
                'scale',
                'translate',
                'rotate',
                'shear',
                'contrast_limits',
                'metadata',
                'experimental_clipping_planes',
            }:
                expectation = ensure_sequence_of_iterables(
                    expectation, repeat_empty=True
                )
            elif key == 'colormap' and expectation is not None:
                if isinstance(expectation, list):
                    exp = [ensure_colormap(c).name for c in expectation]
                else:
                    exp = ensure_colormap(expectation).name
                expectation = ensure_iterable(exp)
            else:
                expectation = ensure_iterable(expectation)
            expectation = [
                v for i, v in zip(range(i + 1), expectation, strict=False)
            ]

            result = getattr(viewer.layers[i], key)
            if key == 'colormap':  # colormaps are tuples of (name, cmap)
                result = result.name
            if isinstance(result, np.ndarray):
                np.testing.assert_almost_equal(result, expectation[i])
            else:
                assert result == expectation[i]


def test_multichannel_multiscale():
    """Test adding multichannel multiscale."""
    viewer = ViewerModel()
    np.random.seed(0)
    shapes = [(40, 20, 4), (20, 10, 4), (10, 5, 4)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_image(data, channel_axis=-1, multiscale=True)
    assert len(viewer.layers) == data[0].shape[-1]
    for i in range(data[0].shape[-1]):
        assert np.all(
            [
                np.array_equal(l_d, d)
                for l_d, d in zip(
                    viewer.layers[i].data,
                    [data[j].take(i, axis=-1) for j in range(len(data))],
                    strict=False,
                )
            ]
        )
        assert viewer.layers[i].colormap.name == base_colormaps[i]


def test_multichannel_implicit_multiscale():
    """Test adding multichannel implicit multiscale."""
    viewer = ViewerModel()
    np.random.seed(0)
    shapes = [(40, 20, 4), (20, 10, 4), (10, 5, 4)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_image(data, channel_axis=-1)
    assert len(viewer.layers) == data[0].shape[-1]
    for i in range(data[0].shape[-1]):
        assert np.all(
            [
                np.array_equal(l_d, d)
                for l_d, d in zip(
                    viewer.layers[i].data,
                    [data[j].take(i, axis=-1) for j in range(len(data))],
                    strict=False,
                )
            ]
        )
        assert viewer.layers[i].colormap.name == base_colormaps[i]


def test_multichannel_dask_array():
    """Test adding multichannel dask array."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = da.random.random((2, 10, 10, 5))
    viewer.add_image(data, channel_axis=0)
    assert len(viewer.layers) == data.shape[0]
    for i in range(data.shape[0]):
        assert viewer.layers[i].data.shape == data.shape[1:]
        assert isinstance(viewer.layers[i].data, type(data))


def test_forgot_multichannel_error_hint():
    """Test that a helpful error is raised when channel_axis is not used."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = da.random.random((15, 10, 5))
    with pytest.raises(TypeError) as e:
        viewer.add_image(data, name=['a', 'b', 'c'])
    assert "did you mean to specify a 'channel_axis'" in str(e)


def test_multichannel_index_error_hint():
    """Test multichannel error when arg length != n_channels."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = da.random.random((5, 10, 5))
    with pytest.raises(IndexError) as e:
        viewer.add_image(data, channel_axis=0, name=['a', 'b'])
    assert (
        'Requested channel_axis (0) had length 5, but the '
        "'name' argument only provided 2 values." in str(e)
    )
