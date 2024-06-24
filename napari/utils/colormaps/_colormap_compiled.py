import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from napari.utils.colormaps import DirectLabelColormap


def _dummy_numba(*_: Any, **__: Any) -> np.ndarray:
    raise NotImplementedError('No numba backend available')


def _dummy_partseg(*_: Any, **__: Any) -> np.ndarray:
    raise NotImplementedError('No partseg backend available')


try:
    from napari.utils.colormaps._colormap_numba import (
        labels_raw_to_texture_direct_numba,
        zero_preserving_modulo_numba,
    )

    NUMBA = True
except ImportError:
    NUMBA = False
    zero_preserving_modulo_numba = _dummy_numba
    labels_raw_to_texture_direct_numba = _dummy_numba

try:
    from napari.utils.colormaps._colormap_partseg import (
        labels_raw_to_texture_direct_partseg,
        zero_preserving_modulo_partseg,
    )

    PARTSEG = True
except ImportError:
    PARTSEG = False
    zero_preserving_modulo_partseg = _dummy_partseg
    labels_raw_to_texture_direct_partseg = _dummy_partseg

if not (NUMBA or PARTSEG):
    raise ImportError('No compiled backend available')

if PARTSEG:

    def zero_preserving_modulo(
        values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
    ) -> np.ndarray:
        try:
            return zero_preserving_modulo_partseg(values, n, dtype, to_zero)
        except Exception:
            if NUMBA:
                logging.exception(
                    'Error in PartSeg backend, trying Numba instead'
                )
                return zero_preserving_modulo_numba(values, n, dtype, to_zero)
            raise

    def labels_raw_to_texture_direct(
        data: np.ndarray, direct_colormap: 'DirectLabelColormap'
    ) -> np.ndarray:
        try:
            return labels_raw_to_texture_direct_partseg(data, direct_colormap)
        except Exception:
            if NUMBA:
                logging.exception(
                    'Error in PartSeg backend, trying Numba instead'
                )
                return labels_raw_to_texture_direct_numba(
                    data, direct_colormap
                )
            raise

else:
    zero_preserving_modulo = zero_preserving_modulo_numba
    labels_raw_to_texture_direct = labels_raw_to_texture_direct_numba
