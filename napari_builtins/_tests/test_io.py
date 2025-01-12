import csv
import os
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import dask.array as da
import imageio
import npe2
import numpy as np
import pytest
import tifffile
import zarr

from napari_builtins.io._read import (
    _guess_layer_type_from_column_names,
    _guess_zarr_path,
    csv_to_layer_data,
    magic_imread,
    read_csv,
)
from napari_builtins.io._write import write_csv


class ImageSpec(NamedTuple):
    shape: tuple[int, ...]
    dtype: str
    ext: str
    levels: int = 1


PNG = ImageSpec((10, 10), 'uint8', '.png')
PNG_RGB = ImageSpec((10, 10, 3), 'uint8', '.png')
PNG_RECT = ImageSpec((10, 15), 'uint8', '.png')
TIFF_2D = ImageSpec((15, 10), 'uint8', '.tif')
TIFF_3D = ImageSpec((2, 15, 10), 'uint8', '.tif')
ZARR1 = ImageSpec((10, 20, 20), 'uint8', '.zarr')


@pytest.fixture
def write_spec(tmp_path: Path):
    def writer(spec: ImageSpec):
        image = np.random.random(spec.shape).astype(spec.dtype)
        fname = tmp_path / f'{uuid4()}{spec.ext}'
        if spec.ext == '.tif':
            tifffile.imwrite(str(fname), image)
        elif spec.ext == '.zarr':
            fname.mkdir()
            z = zarr.open(store=str(fname), mode='a', shape=image.shape)
            z[:] = image
        else:
            imageio.imwrite(str(fname), image)
        return fname

    return writer


def test_no_files_raises(tmp_path):
    with pytest.raises(ValueError, match='No files found in'):
        magic_imread(tmp_path)


def test_guess_zarr_path():
    assert _guess_zarr_path('dataset.zarr')
    assert _guess_zarr_path('dataset.zarr/some/long/path')
    assert not _guess_zarr_path('data.tif')
    assert not _guess_zarr_path('no_zarr_suffix/data.png')


def test_zarr(tmp_path):
    image = np.random.random((10, 20, 20))
    data_path = str(tmp_path / 'data.zarr')
    z = zarr.open(store=data_path, mode='a', shape=image.shape)
    z[:] = image
    image_in = magic_imread([data_path])
    np.testing.assert_array_equal(image, image_in)


def test_zarr_nested(tmp_path):
    image = np.random.random((10, 20, 20))
    image_name = 'my_image'
    root_path = tmp_path / 'dataset.zarr'
    grp = zarr.open(store=str(root_path), mode='a')
    grp.create_dataset(image_name, data=image, shape=image.shape)

    image_in = magic_imread([str(root_path / image_name)])
    np.testing.assert_array_equal(image, image_in)


def test_zarr_with_unrelated_file(tmp_path):
    image = np.random.random((10, 20, 20))
    image_name = 'my_image'
    root_path = tmp_path / 'dataset.zarr'
    grp = zarr.open(store=str(root_path), mode='a')
    grp.create_dataset(image_name, data=image, shape=image.shape)

    txt_file_path = root_path / 'unrelated.txt'
    txt_file_path.touch()

    image_in = magic_imread([str(root_path)])
    np.testing.assert_array_equal(image, image_in[0])


def test_zarr_multiscale(tmp_path):
    multiscale = [
        np.random.random((20, 20)),
        np.random.random((10, 10)),
        np.random.random((5, 5)),
    ]
    fout = str(tmp_path / 'multiscale.zarr')

    root = zarr.open_group(fout, mode='a')
    for i in range(len(multiscale)):
        shape = 20 // 2**i
        z = root.create_dataset(str(i), shape=(shape,) * 2, dtype=np.float64)
        z[:] = multiscale[i]
    multiscale_in = magic_imread([fout])
    assert len(multiscale) == len(multiscale_in)
    for images, images_in in zip(multiscale, multiscale_in, strict=False):
        np.testing.assert_array_equal(images, images_in)


def test_write_csv(tmpdir):
    expected_filename = os.path.join(tmpdir, 'test.csv')
    column_names = ['column_1', 'column_2', 'column_3']
    expected_data = np.random.random((5, len(column_names)))

    # Write csv file
    write_csv(expected_filename, expected_data, column_names=column_names)
    assert os.path.exists(expected_filename)

    # Check csv file is as expected
    with open(expected_filename) as output_csv:
        csv.reader(output_csv, delimiter=',')
        for row_index, row in enumerate(output_csv):
            if row_index == 0:
                assert row == 'column_1,column_2,column_3\n'
            else:
                output_row_data = [float(i) for i in row.split(',')]
                np.testing.assert_allclose(
                    np.array(output_row_data), expected_data[row_index - 1]
                )


def test_read_csv(tmpdir):
    expected_filename = os.path.join(tmpdir, 'test.csv')
    column_names = ['column_1', 'column_2', 'column_3']
    expected_data = np.random.random((5, len(column_names)))

    # Write csv file
    write_csv(expected_filename, expected_data, column_names=column_names)
    assert os.path.exists(expected_filename)

    # Read csv file
    read_data, read_column_names, _ = read_csv(expected_filename)
    read_data = np.array(read_data).astype('float')
    np.testing.assert_allclose(expected_data, read_data)

    assert column_names == read_column_names


def test_guess_layer_type_from_column_names():
    points_names = ['index', 'axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(points_names) == 'points'

    shapes_names = ['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(shapes_names) == 'shapes'

    also_points_names = ['no-index', 'axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(also_points_names) == 'points'

    bad_names = ['no-index', 'no-axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(bad_names) is None


def test_read_csv_raises(tmp_path):
    """Test various exception raising circumstances with read_csv."""
    temp = tmp_path / 'points.csv'

    # test that points data is detected with require_type = None, any, points
    # but raises for other shape types.
    data = [['index', 'axis-0', 'axis-1']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert read_csv(temp, require_type=None)[2] == 'points'
    assert read_csv(temp, require_type='any')[2] == 'points'
    assert read_csv(temp, require_type='points')[2] == 'points'
    with pytest.raises(ValueError, match='not recognized as'):
        read_csv(temp, require_type='shapes')

    # test that unrecognized data is detected with require_type = None
    # but raises for specific shape types or "any"
    data = [['some', 'random', 'header']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert read_csv(temp, require_type=None)[2] is None
    with pytest.raises(ValueError, match='not recognized as'):
        assert read_csv(temp, require_type='any')
    with pytest.raises(ValueError, match='not recognized as'):
        assert read_csv(temp, require_type='points')
    with pytest.raises(ValueError, match='not recognized as'):
        read_csv(temp, require_type='shapes')


def test_csv_to_layer_data_raises(tmp_path):
    """Test various exception raising circumstances with csv_to_layer_data."""
    temp = tmp_path / 'points.csv'

    # test that points data is detected with require_type == points, any, None
    # but raises for other shape types.
    data = [['index', 'axis-0', 'axis-1']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert csv_to_layer_data(temp, require_type=None)[2] == 'points'
    assert csv_to_layer_data(temp, require_type='any')[2] == 'points'
    assert csv_to_layer_data(temp, require_type='points')[2] == 'points'
    with pytest.raises(ValueError, match='not recognized as'):
        csv_to_layer_data(temp, require_type='shapes')

    # test that unrecognized data simply returns None when require_type==None
    # but raises for specific shape types or require_type=="any"
    data = [['some', 'random', 'header']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert csv_to_layer_data(temp, require_type=None) is None
    with pytest.raises(ValueError, match='not recognized as'):
        assert csv_to_layer_data(temp, require_type='any')
    with pytest.raises(ValueError, match='not recognized as'):
        assert csv_to_layer_data(temp, require_type='points')
    with pytest.raises(ValueError, match='not recognized as'):
        csv_to_layer_data(temp, require_type='shapes')


@pytest.mark.parametrize('spec', [PNG, PNG_RGB, TIFF_3D, TIFF_2D])
@pytest.mark.parametrize('stacks', [1, 3])
def test_single_file(spec: ImageSpec, write_spec, stacks: int):
    fnames = [str(write_spec(spec)) for _ in range(stacks)]
    [(layer_data,)] = npe2.read(fnames, stack=stacks > 1)
    assert isinstance(layer_data, np.ndarray if stacks == 1 else da.Array)
    assert layer_data.shape == tuple(i for i in (stacks, *spec.shape) if i > 1)
    assert layer_data.dtype == spec.dtype


@pytest.mark.parametrize(
    'spec', [PNG, [PNG], [PNG, PNG], TIFF_3D, [TIFF_3D, TIFF_3D]]
)
@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('use_dask', [True, False, None])
def test_magic_imread(write_spec, spec: ImageSpec, stack, use_dask):
    fnames = (
        [write_spec(s) for s in spec]
        if isinstance(spec, list)
        else write_spec(spec)
    )
    images = magic_imread(fnames, stack=stack, use_dask=use_dask)
    if isinstance(spec, ImageSpec):
        expect_shape = spec.shape
    else:
        expect_shape = (len(spec), *spec[0].shape) if stack else spec[0].shape
    expect_shape = tuple(i for i in expect_shape if i > 1)

    expected_arr_type = (
        da.Array
        if (
            use_dask
            or (use_dask is None and isinstance(spec, list) and len(spec) > 1)
        )
        else np.ndarray
    )
    if isinstance(spec, list) and len(spec) > 1 and not stack:
        assert isinstance(images, list)
        assert all(isinstance(img, expected_arr_type) for img in images)
        assert all(img.shape == expect_shape for img in images)
    else:
        assert isinstance(images, expected_arr_type)
        assert images.shape == expect_shape


@pytest.mark.parametrize('stack', [True, False])
def test_irregular_images(write_spec, stack):
    specs = [PNG, PNG_RECT]
    fnames = [str(write_spec(spec)) for spec in specs]

    # Ideally, this would work "magically" with dask and irregular images,
    # but there is no foolproof way to do this without reading in all the
    # files. We need to be able to inspect the file shape without reading
    # it in first, then we can automatically turn stacking off when shapes
    # are irregular (and create proper dask arrays)
    if stack:
        with pytest.raises(
            ValueError, match='input arrays must have the same shape'
        ):
            magic_imread(fnames, use_dask=False, stack=stack)
        return

    images = magic_imread(fnames, use_dask=False, stack=stack)
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(
        img.shape == spec.shape
        for img, spec in zip(images, specs, strict=False)
    )


def test_add_zarr(write_spec):
    [out] = npe2.read([str(write_spec(ZARR1))], stack=False)
    assert out[0].shape == ZARR1.shape  # type: ignore


def test_add_zarr_1d_array_is_ignored(tmp_path):
    zarr_dir = str(tmp_path / 'data.zarr')
    # For more details: https://github.com/napari/napari/issues/1471

    z = zarr.open(store=zarr_dir, mode='w')
    z.zeros(name='1d', shape=(3,), chunks=(3,), dtype='float32')

    image_path = os.path.join(zarr_dir, '1d')
    assert npe2.read([image_path], stack=False) == [(None,)]


def test_add_many_zarr_1d_array_is_ignored(tmp_path):
    # For more details: https://github.com/napari/napari/issues/1471
    zarr_dir = str(tmp_path / 'data.zarr')

    z = zarr.open(store=zarr_dir, mode='w')

    z.zeros(name='1d', shape=(3,), chunks=(3,), dtype='float32')
    z.zeros(name='2d', shape=(3, 4), chunks=(3, 4), dtype='float32')
    z.zeros(name='3d', shape=(3, 4, 5), chunks=(3, 4, 5), dtype='float32')

    for name in z.array_keys():
        [out] = npe2.read([os.path.join(zarr_dir, name)], stack=False)
        if name.endswith('1d'):
            assert out == (None,)
        else:
            assert isinstance(out[0], da.Array), name
            assert out[0].ndim == int(name[0])
