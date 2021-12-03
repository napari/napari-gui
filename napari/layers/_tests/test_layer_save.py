import os

import numpy as np

# the layer_writer_and_data fixture is defined in napari/conftest.py
import pandas as pd


def test_layer_save(tmpdir, layer_writer_and_data):
    """Test saving layer data."""
    writer, layer_data, extension, reader, Layer = layer_writer_and_data
    layer = Layer(layer_data[0], **layer_data[1])
    path = os.path.join(tmpdir, 'layer_file' + extension)

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    assert layer.save(path, plugin='builtins')

    # Check file now exists
    assert os.path.isfile(path)

    # Read data
    read_data, read_meta, layer_type = reader(path)

    # Compare read data to original data on layer
    if type(read_data) is list:
        for rd, ld in zip(read_data, layer_data[0]):
            np.testing.assert_allclose(rd, ld)
    else:
        np.testing.assert_allclose(read_data, layer_data[0])

    # Instantiate layer
    read_layer = Layer(read_data, **read_meta)
    read_layer_data = read_layer.as_layer_data_tuple()

    # Compare layer data
    if type(read_layer_data[0]) is list:
        for ld, rld in zip(layer_data[0], read_layer_data[0]):
            np.testing.assert_allclose(ld, rld)
    else:
        np.testing.assert_allclose(layer_data[0], read_layer_data[0])

    # Compare layer metadata
    assert layer_data[1].keys() == read_layer_data[1].keys()
    for key in layer_data[1]:
        data = layer_data[1][key]
        read_data = read_layer_data[1][key]
        if isinstance(data, pd.DataFrame):
            pd.testing.assert_frame_equal(data, read_data)
        else:
            np.testing.assert_equal(data, read_data)

    # Compare layer type
    assert layer_data[2] == read_layer_data[2]


# the layer fixture is defined in napari/conftest.py
def test_layer_save_svg(tmpdir, layer):
    """Test saving layer data to an svg."""
    path = os.path.join(tmpdir, 'layer_file.svg')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    assert layer.save(path, plugin='svg')

    # Check file now exists
    assert os.path.isfile(path)
