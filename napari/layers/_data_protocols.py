"""This module holds Protocols that layer.data objects are expected to provide."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Union,
    runtime_checkable,
)

from napari.utils.translations import trans

_OBJ_NAMES = set(dir(Protocol))
_OBJ_NAMES.update({'__annotations__', '__dict__', '__weakref__'})

if TYPE_CHECKING:
    from enum import Enum

    from numpy.typing import DTypeLike

    # https://github.com/python/typing/issues/684#issuecomment-548203158
    class ellipsis(Enum):
        Ellipsis = '...'

    Ellipsis = ellipsis.Ellipsis  # noqa: A001
else:
    ellipsis = type(Ellipsis)


def _raise_protocol_error(obj: Any, protocol: type) -> None:
    """Raise a more helpful error when required protocol members are missing."""
    annotations = getattr(protocol, '__annotations__', {})
    needed = set(dir(protocol)).union(annotations) - _OBJ_NAMES
    missing = needed - set(dir(obj))
    message = trans._(
        'Object of type {type_name} does not implement {protocol_name} Protocol.\nMissing methods: {missing_methods}',
        deferred=True,
        type_name=repr(type(obj).__name__),
        protocol_name=repr(protocol.__name__),
        missing_methods=repr(missing),
    )
    raise TypeError(message)


Index = Union[int, slice, ellipsis]


@runtime_checkable
class LayerDataProtocol(Protocol):
    """Protocol that all layer.data must support.

    We don't explicitly declare the array types we support (i.e. dask, xarray,
    etc...).  Instead, we support protocols.

    This Protocol is a place to document the attributes and methods that must
    be present for an object to be used as `layer.data`. We should aim to
    ensure that napari never accesses a method on `layer.data` that is not in
    this protocol.

    This protocol should remain a subset of the Array API proposed by the
    Python array API standard:
    https://data-apis.org/array-api/latest/API_specification/array_object.html


    WIP: Shapes.data may be an execption.
    """

    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Array dimensions."""

    def __getitem__(
        self, key: Index | tuple[Index, ...] | LayerDataProtocol
    ) -> LayerDataProtocol:
        """Returns self[key]."""

    @property
    def size(self) -> int:
        """The size is necessary to calculate the data range"""

    @property
    def ndim(self) -> int:
        """The number of dimension of the underlying data"""


def assert_protocol(obj: Any, protocol: type = LayerDataProtocol) -> None:
    """Assert `obj` is an instance of `protocol` or raise helpful error."""
    if not isinstance(obj, protocol):
        _raise_protocol_error(obj, protocol)
