from pathlib import Path
from typing import TYPE_CHECKING

import npe2
import numpy as np
import pytest

from napari_builtins.io import napari_get_reader

if TYPE_CHECKING:
    from napari import layers

_EXTENSION_MAP = {
    'image': '.tif',
    'labels': '.tif',
    'points': '.csv',
    'shapes': '.csv',
}


@pytest.mark.parametrize('use_ext', [True, False])
def test_layer_save(tmp_path: Path, some_layer: 'layers.Layer', use_ext: bool):
    """Test saving layer data."""
    ext = _EXTENSION_MAP[some_layer._type_string]
    path_with_ext = tmp_path / f'layer_file{ext}'
    path_no_ext = tmp_path / 'layer_file'
    assert not path_with_ext.is_file()
    assert some_layer.save(str(path_with_ext if use_ext else path_no_ext))
    assert path_with_ext.is_file()

    # Read data back in
    reader = napari_get_reader(str(path_with_ext))
    assert callable(reader)
    [(read_data, *rest)] = reader(str(path_with_ext))

    if isinstance(some_layer.data, list):
        for d in zip(read_data, some_layer.data, strict=False):
            np.testing.assert_allclose(*d)
    else:
        np.testing.assert_allclose(read_data, some_layer.data)

    if rest:
        meta, type_string = rest
        assert type_string == some_layer._type_string
        for key, value in meta.items():  # type: ignore
            np.testing.assert_equal(value, getattr(some_layer, key))


# the layer_writer_and_data fixture is defined in napari/conftest.py
def test_no_write_layer_bad_extension(some_layer: 'layers.Layer'):
    """Test not writing layer data with a bad extension."""
    with pytest.warns(UserWarning, match='No data written!'):
        assert not some_layer.save('layer.bad_extension')


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_succeeds(
    tmp_path: Path, layers_list: 'list[layers.Layer]'
):
    """Test writing layers data."""

    path = tmp_path / 'layers_folder'
    written = npe2.write(path=str(path), layer_data=layers_list)  # type: ignore

    # check expected files were written
    expected = {
        str(path / f'{layer.name}{_EXTENSION_MAP[layer._type_string]}')
        for layer in layers_list
    }
    assert path.is_dir()
    assert set(written) == expected
    for expect in expected:
        assert Path(expect).is_file()
