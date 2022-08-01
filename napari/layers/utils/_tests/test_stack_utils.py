import numpy as np
import pytest

from napari.layers import Image
from napari.utils.stack_utils import (
    images_to_stack,
    split_channels,
    stack_to_images,
)
from napari.utils.transforms import Affine


def test_stack_to_images_basic():
    """Test that a 2 channel zcyx stack is split into 2 image layers"""
    data = np.random.randint(0, 100, (10, 2, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=None)

    assert isinstance(images, list)
    assert images[0].colormap.name == 'magenta'
    assert len(images) == 2

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)


def test_stack_to_images_multiscale():
    """Test that a 3 channel multiscale image returns 3 multiscale images."""
    data = list()
    data.append(np.random.randint(0, 200, (3, 128, 128)))
    data.append(np.random.randint(0, 200, (3, 64, 64)))
    data.append(np.random.randint(0, 200, (3, 32, 32)))
    data.append(np.random.randint(0, 200, (3, 16, 16)))

    stack = Image(data)
    images = stack_to_images(stack, 0)

    assert len(images) == 3
    assert len(images[0].data) == 4
    assert images[0].data[-1].shape[-1] == 16
    assert images[1].data[-1].shape[-1] == 16
    assert images[2].data[-1].shape[-1] == 16


def test_stack_to_images_rgb():
    """Test 3 channel RGB image (channel axis = -1) into single channels."""
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = stack_to_images(stack, -1, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 3

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)
        assert i.scale.shape == (3,)
        assert i.rgb is False


def test_stack_to_images_4_channels():
    """Test 4x128x128 stack is split into 4 channels w/ colormap keyword"""
    data = np.random.randint(0, 100, (4, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 0, colormap=['red', 'blue'])

    assert isinstance(images, list)
    assert len(images) == 4
    assert images[-2].colormap.name == 'red'
    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (128, 128)


def test_stack_to_images_0_rgb():
    """Split RGB along the first axis (z or t) so the images remain rgb"""
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = stack_to_images(stack, 0, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 10
    for i in images:
        assert i.rgb
        assert type(stack) == type(i)
        assert i.data.shape == (128, 128, 3)


def test_stack_to_images_1_channel():
    """Split when only one channel"""
    data = np.random.randint(0, 100, (10, 1, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=['magma'])

    assert isinstance(images, list)
    assert len(images) == 1
    for i in images:
        assert i.rgb is False
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)


def test_images_to_stack_with_scale():
    """Test that 3-Image list is combined to stack with scale and translate."""
    images = [
        Image(np.random.randint(0, 255, (10, 128, 128))) for _ in range(3)
    ]

    stack = images_to_stack(
        images, 1, colormap='green', scale=(3, 1, 1, 1), translate=(1, 0, 2, 3)
    )

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap.name == 'green'
    assert list(stack.scale) == [3, 1, 1, 1]
    assert list(stack.translate) == [1, 0, 2, 3]


def test_images_to_stack_none_scale():
    """Test combining images using scale & translate from 1st image in list"""
    images = [
        Image(
            np.random.randint(0, 255, (10, 128, 128)),
            scale=(4, 1, 1),
            translate=(0, -1, 2),
        )
        for _ in range(3)
    ]

    stack = images_to_stack(images, 1, colormap='green')

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap.name == 'green'
    assert list(stack.scale) == [4, 1, 1, 1]
    assert list(stack.translate) == [0, 0, -1, 2]


@pytest.fixture(
    params=[
        {
            'rgb': None,
            'colormap': None,
            'contrast_limits': None,
            'gamma': 1,
            'interpolation': 'nearest',
            'rendering': 'mip',
            'iso_threshold': 0.5,
            'attenuation': 0.5,
            'name': None,
            'metadata': None,
            'scale': None,
            'translate': None,
            'opacity': 1,
            'blending': None,
            'visible': True,
            'multiscale': None,
            'rotate': None,
            'affine': None,
        },
        {
            'rgb': None,
            'colormap': None,
            'rendering': 'mip',
            'attenuation': 0.5,
            'metadata': None,
            'scale': None,
            'opacity': 1,
            'visible': True,
            'multiscale': None,
        },
        {},
    ],
    ids=['full-kwargs', 'partial-kwargs', 'empty-kwargs'],
)
def kwargs(request):
    return request.param


def test_split_channels(kwargs):
    """Test split_channels with shape (3,128,128) expecting 3 (128,128)"""
    data = np.random.randint(0, 200, (3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)


def test_split_channels_multiscale(kwargs):
    """Test split_channels with multiscale expecting List[LayerData]"""
    data = list()
    data.append(np.random.randint(0, 200, (3, 128, 128)))
    data.append(np.random.randint(0, 200, (3, 64, 64)))
    data.append(np.random.randint(0, 200, (3, 32, 32)))
    data.append(np.random.randint(0, 200, (3, 16, 16)))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for ds, m, _ in result_list:
        assert m['multiscale'] is True
        assert ds[0].shape == (128, 128)
        assert ds[1].shape == (64, 64)
        assert ds[2].shape == (32, 32)
        assert ds[3].shape == (16, 16)


def test_split_channels_blending(kwargs):
    """Test split_channels with shape (3,128,128) expecting 3 (128,128)"""
    kwargs['blending'] = 'translucent'
    data = np.random.randint(0, 200, (3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)
        assert meta['blending'] == 'translucent'


def test_split_channels_missing_keywords():
    data = np.random.randint(0, 200, (3, 128, 128))
    result_list = split_channels(data, 0)

    assert len(result_list) == 3
    for chan, layer in enumerate(result_list):
        assert layer[0].shape == (128, 128)
        assert (
            layer[1]['blending'] == 'translucent_no_depth'
            if chan == 0
            else 'additive'
        )


def test_split_channels_affine_nparray(kwargs):
    kwargs['affine'] = np.eye(3)
    data = np.random.randint(0, 200, (3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)
        assert np.array_equal(meta['affine'], np.eye(3))


def test_split_channels_affine_napari(kwargs):
    kwargs['affine'] = Affine(affine_matrix=np.eye(3))
    data = np.random.randint(0, 200, (3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)
        assert np.array_equal(meta['affine'].affine_matrix, np.eye(3))


def test_split_channels_multi_affine_napari(kwargs):
    kwargs['affine'] = [
        Affine(scale=[1, 1]),
        Affine(scale=[2, 2]),
        Affine(scale=[3, 3]),
    ]

    data = np.random.randint(0, 200, (3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for idx, result_data in enumerate(result_list):
        d, meta, _ = result_data
        assert d.shape == (128, 128)
        assert np.array_equal(
            meta['affine'].affine_matrix,
            Affine(scale=[idx + 1, idx + 1]).affine_matrix,
        )
