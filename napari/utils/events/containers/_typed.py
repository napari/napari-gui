import logging
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from typing import (
    Any,
    TypeVar,
    Union,
    overload,
)

# change on import from typing when drop python 3.10 support
from typing_extensions import Self

from napari.utils.translations import trans

logger = logging.getLogger(__name__)


Index = Union[int, slice]

_T = TypeVar('_T')
_L = TypeVar('_L', bound=Any)


class TypedMutableSequence(MutableSequence[_T]):
    """List mixin that enforces item type, and enables custom indexing.

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the list with.
    basetype : type or sequence of types, optional
        Type of the elements in the list.  If a basetype (or multiple) is
        provided, then a TypeError will be raised when attempting to add an
        item to this sequence if it is not an instance of one of the types in
        ``basetype``.
    lookup : dict of Type[L] : function(object) -> L
        Mapping between a type, and a function that converts items in the list
        to that type.  This is used for custom indexing.  For example, if a
        ``lookup`` of {str: lambda x: x.name} is provided, then you can index
        into the list using ``list['frank']`` and it will search for an object
        whos attribute ``.name`` equals ``'frank'``.
    """

    def __init__(
        self,
        data: Iterable[_T] = (),
        *,
        basetype: type[_T] | Sequence[type[_T]] = (),
        lookup: dict[type[_L], Callable[[_T], _T | _L]] | None = None,
    ) -> None:
        if lookup is None:
            lookup = {}
        self._list: list[_T] = []
        self._basetypes: tuple[type[_T], ...] = (
            tuple(basetype) if isinstance(basetype, Sequence) else (basetype,)
        )
        self._lookup = lookup.copy()
        self.extend(data)

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return repr(self._list)

    def __eq__(self, other: object) -> bool:
        return self._list == other

    def __hash__(self) -> int:
        # it's important to add this to allow this object to be hashable
        # given that we've also reimplemented __eq__
        return id(self)

    @overload
    def __setitem__(self, key: int, value: _T): ...  # pragma: no cover

    @overload
    def __setitem__(
        self, key: slice, value: Iterable[_T]
    ): ...  # pragma: no cover

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError(
                    trans._(
                        'Can only assign an iterable to slice',
                        deferred=True,
                    )
                )
            self._list[key] = [self._type_check(v) for v in value]
        else:
            self._list[key] = self._type_check(value)

    def insert(self, index: int, value: _T) -> None:
        self._list.insert(index, self._type_check(value))

    def __contains__(self, key: Any) -> bool:
        if type(key) in self._lookup:
            try:
                self[self.index(key)]
            except ValueError:
                return False
            else:
                return True
        return super().__contains__(key)

    @overload
    def __getitem__(self, key: str) -> _T: ...  # pragma: no cover

    @overload
    def __getitem__(self, key: int) -> _T: ...  # pragma: no cover

    @overload
    def __getitem__(
        self, key: slice
    ) -> 'TypedMutableSequence[_T]': ...  # pragma: no cover

    def __getitem__(self, key):
        """Get an item from the list

        Parameters
        ----------
        key : int, slice, or any type in self._lookup
            The key to get.

        Returns
        -------
        The value at `key`

        Raises
        ------
        IndexError:
            If ``type(key)`` is not in ``self._lookup`` (usually an int, like a regular
            list), and the index is out of range.
        KeyError:
            If type(key) is in self._lookup and the key is not in the list (after)
            applying the self._lookup[key] function to each item in the list
        """
        if type(key) in self._lookup:
            try:
                return self.__getitem__(self.index(key))
            except ValueError as e:
                raise KeyError(str(e)) from e

        result = self._list[key]
        return self.__newlike__(result) if isinstance(result, list) else result

    def __delitem__(self, key) -> None:
        _key = self.index(key) if type(key) in self._lookup else key
        del self._list[_key]

    def _type_check(self, e: Any) -> _T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                trans._(
                    'Cannot add object with type {dtype!r} to TypedList expecting type {basetypes!r}',
                    deferred=True,
                    dtype=type(e),
                    basetypes=self._basetypes,
                )
            )
        return e

    def __newlike__(
        self, iterable: Iterable[_T]
    ) -> 'TypedMutableSequence[_T]':
        new = self.__class__()
        # separating this allows subclasses to omit these from their `__init__`
        new._basetypes = self._basetypes
        new._lookup = self._lookup.copy()
        new.extend(iterable)
        return new

    def copy(self) -> 'TypedMutableSequence[_T]':
        """Return a shallow copy of the list."""
        return self.__newlike__(self)

    def __add__(self, other: Iterable[_T]) -> 'TypedMutableSequence[_T]':
        """Add other to self, return new object."""
        copy = self.copy()
        copy.extend(other)
        return copy

    def __iadd__(self, other: Iterable[_T]) -> Self:
        """Add other to self in place (self += other)."""
        self.extend(other)
        return self

    def __radd__(self, other: list) -> list:
        """Add other to self in place (self += other)."""
        return other + list(self)

    def index(self, value: _L, start: int = 0, stop: int | None = None) -> int:
        """Return first index of value.

        Parameters
        ----------
        value : Any
            A value to lookup.  If `type(value)` is in the lookups functions
            provided for this class, then values in the list will be searched
            using the corresponding lookup converter function.
        start : int, optional
            The starting index to search, by default 0
        stop : int, optional
            The ending index to search, by default None

        Returns
        -------
        int
            The index of the value

        Raises
        ------
        ValueError
            If the value is not present
        """
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        convert = self._lookup.get(type(value), _noop)

        for i in self._iter_indices(start, stop):
            v = convert(self[i])
            if v is value or v == value:
                return i

        raise ValueError(
            trans._(
                '{value!r} is not in list',
                deferred=True,
                value=value,
            )
        )

    def _iter_indices(
        self, start: int = 0, stop: int | None = None
    ) -> Iterable[int]:
        """Iter indices from start to stop.

        While this is trivial for this basic sequence type, this method lets
        subclasses (like NestableEventedList modify how they are traversed).
        """
        yield from range(start, len(self) if stop is None else stop)

    def _ipython_key_completions_(self):
        if str in self._lookup:
            return (self._lookup[str](x) for x in self)
        return None  # type: ignore


def _noop(x: _T) -> _T:
    return x
