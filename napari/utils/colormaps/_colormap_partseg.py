from typing import TYPE_CHECKING

import numpy as np
from PartSegCore_compiled_backend.napari_mapping import (
    map_array_parallel,
    zero_preserving_modulo_parallel,
)

from napari.utils.colormaps._colormap_numpy import (
    MAPPING_OF_UNKNOWN_VALUE,
    minimum_dtype_for_labels,
)

if TYPE_CHECKING:
    from napari.utils import DirectLabelColormap

__all__ = (
    'zero_preserving_modulo_partseg',
    'labels_raw_to_texture_direct_partseg',
)


def zero_preserving_modulo_partseg(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    out = np.empty(values.size, dtype=dtype)
    zero_preserving_modulo_parallel(values.reshape(-1), n, to_zero, out)
    return out.reshape(values.shape)


def labels_raw_to_texture_direct_partseg(
    data: np.ndarray, direct_colormap: 'DirectLabelColormap'
) -> np.ndarray:
    if direct_colormap.use_selection:
        dkt = {None: 0, direct_colormap.selection: 1}
    else:
        dkt = direct_colormap._label_mapping_and_color_dict[0]
    target_dtype = minimum_dtype_for_labels(
        direct_colormap._num_unique_colors + 2
    )
    out = np.empty(data.size, dtype=target_dtype)
    map_array_parallel(data.reshape(-1), dkt, MAPPING_OF_UNKNOWN_VALUE, out)
    return out.reshape(data.shape)
