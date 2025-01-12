from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import npe2
import numpy as np
import pytest
import tifffile

from napari_builtins.io._write import write_csv


@pytest.fixture
def save_image(tmp_path: Path):
    """Create a temporary file."""

    def _save(filename: str, data: np.ndarray | None = None):
        dest = tmp_path / filename
        data_: np.ndarray = np.random.rand(20, 20) if data is None else data
        if filename.endswith(('png', 'jpg')):
            data_ = (data_ * 255).astype(np.uint8)
        if dest.suffix in {'.tif', '.tiff'}:
            tifffile.imwrite(str(dest), data_)
        elif dest.suffix in {'.npy'}:
            np.save(str(dest), data_)
        else:
            iio.imwrite(str(dest), data_)
        return dest

    return _save


@pytest.mark.parametrize('ext', ['.tif', '.npy', '.png', '.jpg'])
@pytest.mark.parametrize('stack', [False, True])
def test_reader_plugin_tif(save_image: Callable[..., Path], ext, stack):
    """Test the builtin reader plugin reads a temporary file."""
    files = [
        str(save_image(f'test_{i}{ext}')) for i in range(5 if stack else 1)
    ]
    layer_data = npe2.read(files, stack=stack)
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)


def test_animated_gif_reader(save_image):
    threeD_data = (np.random.rand(5, 20, 20, 3) * 255).astype(np.uint8)
    dest = save_image('animated.gif', threeD_data)
    layer_data = npe2.read([str(dest)], stack=False)
    assert len(layer_data) == 1
    assert layer_data[0][0].shape == (5, 20, 20, 3)


@pytest.mark.slow
def test_reader_plugin_url():
    layer_data = npe2.read(
        ['https://samples.fiji.sc/FakeTracks.tif'], stack=False
    )
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)


def test_reader_plugin_csv(tmp_path):
    """Test the builtin reader plugin reads a temporary file."""
    dest = str(tmp_path / 'test.csv')
    table = np.random.random((5, 3))
    write_csv(dest, table, column_names=['index', 'axis-0', 'axis-1'])

    layer_data = npe2.read([dest], stack=False)

    assert layer_data is not None
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)
    assert layer_data[0][2] == 'points'
    assert np.allclose(table[:, 1:], layer_data[0][0])
