import csv
import os
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from napari.utils.io import imsave
from napari.utils.misc import abspath_or_url

if TYPE_CHECKING:
    from napari.types import FullLayerData


def write_csv(
    filename: str,
    data: Union[List, np.ndarray],
    column_names: Optional[List[str]] = None,
):
    """Write a csv file.

    Parameters
    ----------
    filename : str
        Filename for saving csv.
    data : list or ndarray
        Table values, contained in a list of lists or an ndarray.
    column_names : list, optional
        List of column names for table data.
    """
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        if column_names is not None:
            writer.writerow(column_names)
        for row in data:
            writer.writerow(row)


def imsave_extensions() -> Tuple[str, ...]:
    """Valid extensions of files that imsave can write to.

    Returns
    -------
    tuple
        Valid extensions of files that imsave can write to.
    """
    # import imageio
    # return tuple(set(x for f in imageio.formats for x in f.extensions))

    # The above method generates a lot of extensions that will fail.  This list
    # is a more realistic set, generated by trying to write a variety of numpy
    # arrays (skimage.data.camera, grass, and some random numpy arrays/shapes).
    # TODO: maybe write a proper imageio plugin.
    return (
        '.bmp',
        '.bsdf',
        '.bw',
        '.eps',
        '.gif',
        '.icns',
        '.ico',
        '.im',
        '.j2c',
        '.j2k',
        '.jfif',
        '.jp2',
        '.jpc',
        '.jpe',
        '.jpeg',
        '.jpf',
        '.jpg',
        '.jpx',
        '.lsm',
        '.mpo',
        '.npz',
        '.pbm',
        '.pcx',
        '.pgm',
        '.png',
        '.ppm',
        '.ps',
        '.rgb',
        '.rgba',
        '.sgi',
        '.stk',
        '.tga',
        '.tif',
        '.tiff',
    )


def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback image writer at the end of the plugin chain.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If meta['rgb'] is ``True`` then the
        data should be interpreted as RGB or RGBA. If ``meta['multiscale']`` is
        ``True``, then the data should be interpreted as a multiscale image.
    meta : dict
        Image metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        path += '.tif'
        ext = '.tif'

    if ext in imsave_extensions():
        imsave(path, data)
        return path

    return None


def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback labels writer at the end of the plugin chain.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If meta['rgb'] is ``True`` then the
        data should be interpreted as RGB or RGBA. If ``meta['multiscale']`` is
        ``True``, then the data should be interpreted as a multiscale image.
    meta : dict
        Image metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    dtype = data.dtype if data.dtype.itemsize >= 4 else np.uint32
    return napari_write_image(path, np.asarray(data, dtype=dtype), meta)


def napari_write_points(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback points writer at the end of the plugin chain.

    Append ``.csv`` extension to the filename if it is not already there.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array (N, D)
        Coordinates for N points in D dimensions.
    meta : dict
        Points metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    ext = os.path.splitext(path)[1]
    if ext == '':
        path += '.csv'
    elif ext != '.csv':
        # If an extension is provided then it must be `.csv`
        return None

    properties = meta.get('properties', {})
    # TODO: we need to change this to the axis names once we get access to them
    # construct table from data
    column_names = [f'axis-{n!s}' for n in range(data.shape[1])]
    if properties:
        column_names += properties.keys()
        prop_table = [
            np.expand_dims(col, axis=1) for col in properties.values()
        ]
    else:
        prop_table = []

    # add index of each point
    column_names = ["index", *column_names]
    indices = np.expand_dims(list(range(data.shape[0])), axis=1)
    table = np.concatenate([indices, data, *prop_table], axis=1)

    # write table to csv file
    write_csv(path, table, column_names)
    return path


def napari_write_shapes(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback points writer at the end of the plugin chain.

    Append ``.csv`` extension to the filename if it is not already there.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : list of array (N, D)
        List of coordinates for shapes, each with for N vertices in D
        dimensions.
    meta : dict
        Points metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    ext = os.path.splitext(path)[1]
    if ext == '':
        path += '.csv'
    elif ext != '.csv':
        # If an extension is provided then it must be `.csv`
        return None

    shape_type = meta.get('shape_type', ['rectangle'] * len(data))
    # No data passed so nothing written
    if len(data) == 0:
        return None

    # TODO: we need to change this to the axis names once we get access to them
    # construct table from data
    n_dimensions = max(s.shape[1] for s in data)
    column_names = [f'axis-{n!s}' for n in range(n_dimensions)]

    # add shape id and vertex id of each vertex
    column_names = ["index", "shape-type", "vertex-index", *column_names]

    # concatenate shape data into 2D array
    len_shapes = [s.shape[0] for s in data]
    all_data = np.concatenate(data)
    all_idx = np.expand_dims(
        np.concatenate([np.repeat(i, s) for i, s in enumerate(len_shapes)]),
        axis=1,
    )
    all_types = np.expand_dims(
        np.concatenate(
            [np.repeat(shape_type[i], s) for i, s in enumerate(len_shapes)]
        ),
        axis=1,
    )
    all_vert_idx = np.expand_dims(
        np.concatenate([np.arange(s) for s in len_shapes]), axis=1
    )

    table = np.concatenate(
        [all_idx, all_types, all_vert_idx, all_data], axis=1
    )

    # write table to csv file
    write_csv(path, table, column_names)
    return path


def write_layer_data_with_plugins(
    path: str, layer_data: List["FullLayerData"]
) -> List[str]:
    """Write layer data out into a folder one layer at a time.

    Call ``napari_write_<layer>`` for each layer using the ``layer.name``
    variable to modify the path such that the layers are written to unique
    files in the folder.

    Parameters
    ----------
    path : str
        path to file/directory
    layer_data : list of napari.types.LayerData
        List of layer_data, where layer_data is ``(data, meta, layer_type)``.

    Returns
    -------
    list of str
        A list of any filepaths that were written.
    """

    import npe2

    # remember whether it was there to begin with
    already_existed = os.path.exists(path)
    # Try and make directory based on current path if it doesn't exist
    if not already_existed:
        os.makedirs(path)

    written: List[str] = []  # the files that were actually written
    try:
        # build in a temporary directory and then move afterwards,
        # it makes cleanup easier if an exception is raised inside.
        with TemporaryDirectory(dir=path) as tmp:
            # Loop through data for each layer
            for layer_data_tuple in layer_data:
                _, meta, type_ = layer_data_tuple

                # Create full path using name of layer
                # Write out data using first plugin found for this hook spec
                # or named plugin if provided
                npe2.write(
                    path=abspath_or_url(os.path.join(tmp, meta['name'])),
                    layer_data=[layer_data_tuple],
                    plugin_name='napari',
                )
            for fname in os.listdir(tmp):
                written.append(os.path.join(path, fname))
                shutil.move(os.path.join(tmp, fname), path)
    except Exception:
        if not already_existed:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return written
