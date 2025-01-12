from dataclasses import dataclass
from itertools import cycle

import numpy as np

from napari.layers.utils.color_transformations import (
    transform_color,
    transform_color_cycle,
)
from napari.utils.translations import trans


@dataclass(eq=False)
class ColorCycle:
    """A dataclass to hold a color cycle for the fallback_colors
    in the CategoricalColormap

    Attributes
    ----------
    values : np.ndarray
        The (Nx4) color array of all colors contained in the color cycle.
    cycle : cycle
        The cycle object that gives fallback colors.
    """

    values: np.ndarray
    cycle: cycle

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        # turn a generic dict into object
        if isinstance(val, dict):
            return _coerce_colorcycle_from_dict(val)
        if isinstance(val, ColorCycle):
            return val

        return _coerce_colorcycle_from_colors(val)

    def _json_encode(self):
        return {'values': self.values.tolist()}

    def __eq__(self, other):
        if isinstance(other, ColorCycle):
            eq = np.array_equal(self.values, other.values)
        else:
            eq = False
        return eq


def _coerce_colorcycle_from_dict(
    val: dict[str, str | list | np.ndarray | cycle],
) -> ColorCycle:
    # validate values
    color_values = val.get('values')
    if color_values is None:
        raise ValueError(
            trans._('ColorCycle requires a values argument', deferred=True)
        )

    transformed_color_values = transform_color(color_values)

    # validate cycle
    color_cycle = val.get('cycle')
    if color_cycle is None:
        transformed_color_cycle = transform_color_cycle(
            color_cycle=color_values,
            elem_name='color_cycle',
            default='white',
        )[0]
    elif isinstance(color_cycle, cycle):
        transformed_color_cycle = color_cycle
    else:
        raise TypeError(f'cycle entry must be type(cycle), got {type(cycle)}')

    return ColorCycle(
        values=transformed_color_values, cycle=transformed_color_cycle
    )


def _coerce_colorcycle_from_colors(
    val: str | list | np.ndarray,
) -> ColorCycle:
    if isinstance(val, str):
        val = [val]
    (
        transformed_color_cycle,
        transformed_color_values,
    ) = transform_color_cycle(
        color_cycle=val,
        elem_name='color_cycle',
        default='white',
    )
    return ColorCycle(
        values=transformed_color_values, cycle=transformed_color_cycle
    )


def compare_colormap_dicts(cmap_1, cmap_2):
    if len(cmap_1) != len(cmap_2):
        return False
    for k, v in cmap_1.items():
        if k not in cmap_2:
            return False
        if not np.allclose(v, cmap_2[k]):
            return False
    return True
