"""Evented dictionary"""

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    TypeVar,
)

_K = TypeVar('_K')
_T = TypeVar('_T')


class TypedMutableMapping(MutableMapping[_K, _T]):
    """Dictionary mixin that enforces item type."""

    def __init__(
        self,
        data: Mapping[_K, _T] | None = None,
        basetype: type[_T] | Sequence[type[_T]] = (),
    ) -> None:
        if data is None:
            data = {}
        self._dict: dict[_K, _T] = {}
        self._basetypes = (
            basetype if isinstance(basetype, Sequence) else (basetype,)
        )
        self.update(data)

    # #### START Required Abstract Methods

    def __setitem__(self, key: _K, value: _T) -> None:
        self._dict[key] = self._type_check(value)

    def __delitem__(self, key: _K) -> None:
        del self._dict[key]

    def __getitem__(self, key: _K) -> _T:
        return self._dict[key]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._dict)

    def __repr__(self) -> str:
        return str(self._dict)

    def _type_check(self, e: Any) -> _T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                f'Cannot add object with type {type(e)} to TypedDict expecting type {self._basetypes}',
            )
        return e

    def __newlike__(
        self, iterable: MutableMapping[_K, _T]
    ) -> 'TypedMutableMapping[_K, _T]':
        new = self.__class__()
        # separating this allows subclasses to omit these from their `__init__`
        new._basetypes = self._basetypes
        new.update(iterable)
        return new

    def copy(self) -> 'TypedMutableMapping[_K, _T]':
        """Return a shallow copy of the dictionary."""
        return self.__newlike__(self)
