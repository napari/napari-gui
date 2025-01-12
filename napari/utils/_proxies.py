import os
import re
import sys
import warnings
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union

import wrapt

from napari.utils import misc
from napari.utils.translations import trans

_T = TypeVar('_T')


class ReadOnlyWrapper(wrapt.ObjectProxy):
    """
    Disable item and attribute setting with the exception of  ``__wrapped__``.
    """

    def __init__(self, wrapped: Any, exceptions: tuple[str, ...] = ()):
        super().__init__(wrapped)
        self._self_exceptions = exceptions

    def __setattr__(self, name: str, val: Any) -> None:
        if (
            name not in ('__wrapped__', '_self_exceptions')
            and name not in self._self_exceptions
        ):
            raise TypeError(
                trans._(
                    'cannot set attribute {name}',
                    deferred=True,
                    name=name,
                )
            )

        super().__setattr__(name, val)

    def __setitem__(self, name: str, val: Any) -> None:
        if name not in self._self_exceptions:
            raise TypeError(
                trans._('cannot set item {name}', deferred=True, name=name)
            )
        super().__setitem__(name, val)


_SUNDER = re.compile('^_[^_]')


class PublicOnlyProxy(wrapt.ObjectProxy, Generic[_T]):
    """Proxy to prevent private attribute and item access, recursively."""

    __wrapped__: _T

    @staticmethod
    def _is_private_attr(name: str) -> bool:
        return name.startswith('_') and not (
            name.startswith('__') and name.endswith('__')
        )

    @staticmethod
    def _private_attr_warning(name: str, typ: str) -> None:
        warnings.warn(
            trans._(
                "Private attribute access ('{typ}.{name}') in this context "
                '(e.g. inside a plugin widget or dock widget) is deprecated '
                'and will be unavailable in version 0.6.0',
                deferred=True,
                name=name,
                typ=typ,
            ),
            category=FutureWarning,
            stacklevel=3,
        )

        # This is code prepared for a moment where we want to block access to private attributes
        # raise AttributeError(
        #     trans._(
        #         "Private attribute set/access ('{typ}.{name}') not allowed in this context.",
        #         deferred=True,
        #         name=name,
        #         typ=typ,
        #     )
        # )

    @staticmethod
    def _is_called_from_napari() -> bool:
        """
        Check if the getter or setter is called from inner napari.
        """
        if hasattr(sys, '_getframe'):
            frame = sys._getframe(2)
            return frame.f_code.co_filename.startswith(misc.ROOT_DIR)
        return False

    def __getattr__(self, name: str) -> Any:
        if self._is_private_attr(name):
            # allow napari to access private attributes and get an non-proxy
            if self._is_called_from_napari():
                return super().__getattr__(name)

            typ = type(self.__wrapped__).__name__

            self._private_attr_warning(name, typ)
        with warnings.catch_warnings(record=True) as cx_manager:
            data = self.create(super().__getattr__(name))
        for warning in cx_manager:
            warnings.warn(warning.message, warning.category, stacklevel=2)

        return data

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            os.environ.get('NAPARI_ENSURE_PLUGIN_MAIN_THREAD', '0')
            not in ('0', 'False')
        ) and not in_main_thread():
            raise RuntimeError(
                'Setting attributes on a napari object is only allowed from the main Qt thread.'
            )

        if self._is_private_attr(name):
            if self._is_called_from_napari():
                return super().__setattr__(name, value)

            typ = type(self.__wrapped__).__name__
            self._private_attr_warning(name, typ)

        if isinstance(value, PublicOnlyProxy):
            # if we want to set an attribute on a PublicOnlyProxy *and* the
            # value that we want to set is itself a PublicOnlyProxy, we unwrap
            # the value. This has two benefits:
            #
            # 1. Checking the attribute later will incur a significant
            # performance cost, because _is_called_from_napari() will be
            # checked on each attribute access and it involves inspecting the
            # calling frame, which is expensive.
            # 2. Certain equality checks fail when objects are
            # PublicOnlyProxies. Notably, equality checks fail when such
            # objects are included in a Qt data model. For example, plugins can
            # grab a layer from the viewer; this layer will be wrapped by the
            # PublicOnlyProxy, and then using this object to set the current
            # layer selection will not propagate the selection to the Viewer.
            # See https://github.com/napari/napari/issues/5767
            value = value.__wrapped__

        setattr(self.__wrapped__, name, value)
        return None

    def __getitem__(self, key: Any) -> Any:
        return self.create(super().__getitem__(key))

    def __repr__(self) -> str:
        return repr(self.__wrapped__)

    def __dir__(self) -> list[str]:
        return [x for x in dir(self.__wrapped__) if not _SUNDER.match(x)]

    @classmethod
    def create(cls, obj: Any) -> Union['PublicOnlyProxy', Any]:
        # restrict the scope of this proxy to napari objects
        if type(obj).__name__ == 'method':
            # If the given object is a method, we check the module *of the
            # object to which that method is bound*. Otherwise, the module of a
            # method is just builtins!
            mod = getattr(type(obj.__self__), '__module__', None) or ''
        else:
            # Otherwise, the module is of an object just given by the
            # __module__ attribute.
            mod = getattr(type(obj), '__module__', None) or ''
        if not mod.startswith('napari'):
            return obj
        if isinstance(obj, PublicOnlyProxy):
            return obj  # don't double-wrap
        if callable(obj):
            return CallablePublicOnlyProxy(obj)
        return PublicOnlyProxy(obj)


class CallablePublicOnlyProxy(PublicOnlyProxy[Callable]):
    def __call__(self, *args, **kwargs):  # type: ignore [no-untyped-def]
        # if a PublicOnlyProxy is callable, then when we call it we:
        # - unwrap the arguments, to avoid performance issues detailed in
        #   PublicOnlyProxy.__setattr__,
        # - call the unwrapped callable on the unwrapped arguments
        # - wrap the result in a PublicOnlyProxy
        args = tuple(
            arg.__wrapped__ if isinstance(arg, PublicOnlyProxy) else arg
            for arg in args
        )
        kwargs = {
            k: v.__wrapped__ if isinstance(v, PublicOnlyProxy) else v
            for k, v in kwargs.items()
        }
        return self.create(self.__wrapped__(*args, **kwargs))


def in_main_thread_py() -> bool:
    """
    Check if caller is in main python thread.

    Returns
    -------
    thread_flag : bool
        True if we are in the main thread, False otherwise.
    """
    import threading

    return threading.current_thread() == threading.main_thread()


def _in_main_thread() -> bool:
    """
    General implementation of checking if we are in a proper thread.
    If Qt is available and Application is created then assign :py:func:`in_qt_main_thread` to `in_main_thread`.
    If Qt liba are not available then assign :py:func:`in_main_thread_py` to in_main_thread.
    IF Qt libs are available but there is no Application ti wil emmit warning and return result of in_main_thread_py.

    Returns
    -------
    thread_flag : bool
        True if we are in the main thread, False otherwise.
    """

    global in_main_thread
    try:
        from napari._qt.utils import in_qt_main_thread

        res = in_qt_main_thread()
        in_main_thread = in_qt_main_thread
    except ImportError:
        in_main_thread = in_main_thread_py
        return in_main_thread_py()
    except AttributeError:
        warnings.warn(
            'Qt libs are available but no QtApplication instance is created'
        )
        return in_main_thread_py()
    return res


in_main_thread = _in_main_thread
