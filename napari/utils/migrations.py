import inspect
import warnings
from collections import UserDict
from collections.abc import Callable
from functools import wraps
from typing import Any, NamedTuple

from napari.utils.translations import trans

_UNSET = object()


class _RenamedAttribute(NamedTuple):
    """Captures information about a renamed attribute, property, or argument.

    Useful for storing internal state related to these types of deprecations.
    """

    from_name: str
    to_name: str
    version: str
    since_version: str

    def message(self) -> str:
        return trans._(
            '{from_name} is deprecated since {since_version} and will be removed in {version}. Please use {to_name}',
            deferred=True,
            from_name=self.from_name,
            since_version=self.since_version,
            version=self.version,
            to_name=self.to_name,
        )


def rename_argument(
    from_name: str, to_name: str, version: str, since_version: str = ''
) -> Callable:
    """
    This is decorator for simple rename function argument
    without break backward compatibility.

    Parameters
    ----------
    from_name : str
        old name of argument
    to_name : str
        new name of argument
    version : str
        version when old argument will be removed
    since_version : str
        version when new argument was added
    """

    if not since_version:
        since_version = 'unknown'
        warnings.warn(
            trans._(
                'The since_version argument was added in napari 0.4.18 and will be mandatory since 0.6.0 release.',
                deferred=True,
            ),
            stacklevel=2,
            category=FutureWarning,
        )

    def _wrapper(func):
        if not hasattr(func, '_rename_argument'):
            func._rename_argument = []

        func._rename_argument.append(
            _RenamedAttribute(
                from_name=from_name,
                to_name=to_name,
                version=version,
                since_version=since_version,
            )
        )

        @wraps(func)
        def _update_from_dict(*args, **kwargs):
            if from_name in kwargs:
                if to_name in kwargs:
                    raise ValueError(
                        trans._(
                            'Argument {to_name} already defined, please do not mix {from_name} and {to_name} in one call.',
                            from_name=from_name,
                            to_name=to_name,
                        )
                    )
                warnings.warn(
                    trans._(
                        'Argument {from_name!r} is deprecated, please use {to_name!r} instead. The argument {from_name!r} was deprecated in {since_version} and it will be removed in {version}.',
                        from_name=from_name,
                        to_name=to_name,
                        version=version,
                        since_version=since_version,
                    ),
                    category=FutureWarning,
                    stacklevel=2,
                )
                kwargs = kwargs.copy()
                kwargs[to_name] = kwargs.pop(from_name)
            return func(*args, **kwargs)

        return _update_from_dict

    return _wrapper


def add_deprecated_property(
    obj: Any,
    previous_name: str,
    new_name: str,
    version: str,
    since_version: str,
) -> None:
    """
    Adds deprecated property and links to new property name setter and getter.

    Parameters
    ----------
    obj:
        Class instances to add property
    previous_name : str
        Name of previous property, its methods must be removed.
    new_name : str
        Name of new property, must have its getter (and setter if applicable) implemented.
    version : str
        Version where deprecated property will be removed.
    since_version : str
        version when new property was added
    """

    if hasattr(obj, previous_name):
        raise RuntimeError(
            trans._(
                '{previous_name} property already exists.',
                deferred=True,
                previous_name=previous_name,
            )
        )

    if not hasattr(obj, new_name):
        raise RuntimeError(
            trans._(
                '{new_name} property must exist.',
                deferred=True,
                new_name=new_name,
            )
        )

    name = f'{obj.__name__}.{previous_name}'
    msg = trans._(
        '{name} is deprecated since {since_version} and will be removed in {version}. Please use {new_name}',
        deferred=True,
        name=name,
        since_version=since_version,
        version=version,
        new_name=new_name,
    )

    def _getter(instance) -> Any:
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
        return getattr(instance, new_name)

    def _setter(instance, value: Any) -> None:
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
        setattr(instance, new_name, value)

    setattr(obj, previous_name, property(_getter, _setter))


def deprecated_constructor_arg_by_attr(name: str) -> Callable:
    """
    Decorator to deprecate a constructor argument and remove it from the signature.

    It works by popping the argument from kwargs, and setting it later via setattr.
    The property setter should take care of issuing the deprecation warning.

    Parameters
    ----------
    name : str
        Name of the argument to deprecate.

    Returns
    -------
    function
        decorated function
    """

    def wrapper(func):
        if not hasattr(func, '_deprecated_constructor_args'):
            func._deprecated_constructor_args = []
        func._deprecated_constructor_args.append(name)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            value = _UNSET
            if name in kwargs:
                value = kwargs.pop(name)
            res = func(*args, **kwargs)

            if value is not _UNSET:
                setattr(args[0], name, value)
            return res

        return _wrapper

    return wrapper


def deprecated_class_name(
    new_class: type,
    previous_name: str,
    version: str,
    since_version: str,
) -> type:
    """Function to deprecate a class.

    Usage:

        class NewName:
            pass

        OldName = deprecated_class_name(
            NewName, 'OldName', version='0.5.0', since_version='0.4.19'
        )
    """
    msg = (
        f'{previous_name} is deprecated since {since_version} and will be '
        f'removed in {version}. Please use {new_class.__name__}.'
    )
    prealloc_signature = inspect.signature(new_class.__new__)

    class _OldClass(new_class):
        def __new__(cls, *args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            if super().__new__ is object.__new__:
                return super().__new__(cls)
            return super().__new__(cls, *args, **kwargs)

        def __init_subclass__(cls, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)

    _OldClass.__module__ = new_class.__module__
    _OldClass.__name__ = previous_name
    _OldClass.__qualname__ = previous_name
    _OldClass.__new__.__signature__ = prealloc_signature  # type: ignore [attr-defined]

    return _OldClass


class _DeprecatingDict(UserDict[str, Any]):
    """A dictionary that issues warning messages when deprecated keys are accessed.

    This class is intended to be an implementation detail of napari and may change
    in the future. As such, it should not be used outside of napari. Instead, treat
    this like a plain dictionary.

    Deprecated keys and values are not stored as part of the dictionary, so will not
    appear when iterating over this or its items.

    Instead deprecated items can only be accessed using `__getitem__`, `__setitem__`,
    and `__delitem__`.

    Deprecations from pure renames should keep the old and new corresponding items
    consistent when mutating either the old or new item.
    """

    # Maps from a deprecated key to its renamed key and deprecation information.
    _renamed: dict[str, _RenamedAttribute]

    def __init__(self, *args, **kwargs) -> None:
        self._renamed = {}
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        key = self._maybe_rename_key(key)
        return self.data.__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        key = self._maybe_rename_key(key)
        return self.data.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        key = self._maybe_rename_key(key)
        return self.data.__delitem__(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        key = self._maybe_rename_key(key)
        return self.data.__contains__(key)

    def _maybe_rename_key(self, key: str) -> str:
        if key in self._renamed:
            renamed = self._renamed[key]
            warnings.warn(renamed.message(), FutureWarning)
            key = renamed.to_name
        return key

    @property
    def deprecated_keys(self) -> tuple[str, ...]:
        return tuple(self._renamed.keys())

    def set_deprecated_from_rename(
        self, *, from_name: str, to_name: str, version: str, since_version: str
    ) -> None:
        """Sets a deprecated key with a value that comes from another key.

        A warning message is automatically generated using the given version information.
        """
        self._renamed[from_name] = _RenamedAttribute(
            from_name=from_name,
            to_name=to_name,
            version=version,
            since_version=since_version,
        )
