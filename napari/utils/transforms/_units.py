from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Union,
    overload,
)

import pint

UnitsLike = Union[None, str, pint.Unit, Sequence[str | pint.Unit]]
UnitsInfo = Union[None, pint.Unit, tuple[pint.Unit, ...]]


__all__ = (
    'UnitsInfo',
    'UnitsLike',
    'get_units_from_name',
)


@overload
def get_units_from_name(units: None) -> None: ...


@overload
def get_units_from_name(units: str | pint.Unit) -> pint.Unit: ...


@overload
def get_units_from_name(
    units: Sequence[str | pint.Unit],
) -> tuple[pint.Unit, ...]: ...


def get_units_from_name(units: UnitsLike) -> UnitsInfo:
    """Convert a string or sequence of strings to pint units."""
    try:
        if isinstance(units, str):
            return pint.get_application_registry()[units].units
        if isinstance(units, Sequence):
            return tuple(
                pint.get_application_registry()[unit].units
                if isinstance(unit, str)
                else unit
                for unit in units
            )
    except AttributeError as e:
        raise ValueError(f'Could not find unit {units}') from e
    return units
