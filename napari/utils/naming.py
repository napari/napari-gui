"""Automatically generate names."""

import inspect
import re
from collections import ChainMap, ChainMap as ChainMapType
from collections.abc import Callable
from types import FrameType, TracebackType
from typing import (
    Any,
)

from napari.utils.misc import ROOT_DIR, formatdoc

sep = ' '
start = 1

# Match integer between square brackets at end of string if after space
# or at beginning of string or just match end of string
numbered_patt = re.compile(r'((?<=\A\[)|(?<=\s\[))(?:\d+|)(?=\]$)|$')


def _inc_name_count_sub(match: re.Match) -> str:
    count = match.group(0)

    try:
        count = int(count)
    except ValueError:  # not an int
        count = f'{sep}[{start}]'
    else:
        count = f'{count + 1}'

    return count


@formatdoc
def inc_name_count(name: str) -> str:
    """Increase a name's count matching `{numbered_patt}` by ``1``.

    If the name is not already numbered, append '{sep}[{start}]'.

    Parameters
    ----------
    name : str
        Original name.

    Returns
    -------
    incremented_name : str
        Numbered name incremented by ``1``.
    """
    return numbered_patt.sub(_inc_name_count_sub, name, count=1)


class CallerFrame:
    """
    Context manager to access the namespace in one of the upper caller frames.

    It is a context manager in order to be able to properly cleanup references
    to some frame objects after it is gone.

    Constructor takes a predicate taking a index and frame and returning whether
    to skip this frame and keep walking up the stack. The index starts at 1
    (caller frame), and increases.

    For example the following gives you the caller:
        - at least 5 Frames up
        - at most 42 Frames up
        - first one outside of Napari

        def skip_napari_frames(index, frame):
            if index < 5:
                return True
            if index > 42:
                return False
            return frame.f_globals.get("__name__", '').startswith('napari')

        with CallerFrame(skip_napari_frames) as c:
            print(c.namespace)

    This will be used for two things:
        - find the name of a value in caller frame.
        - capture local namespace of `napari.run()` when starting the qt-console

    For more complex logic you could use a callable that keep track of
    previous/state/frames, though be careful, the predicate is not guarantied to
    be called on all subsequents frames.

    """

    names: tuple[str, ...]
    namespace: ChainMapType[str, Any]
    predicate: Callable[[int, FrameType], bool]

    def __init__(
        self, skip_predicate: Callable[[int, FrameType], bool]
    ) -> None:
        self.predicate = skip_predicate
        self.namespace = ChainMap()
        self.names = ()

    def __enter__(self) -> 'CallerFrame':
        frame = inspect.currentframe()
        try:
            # See issue #1635 regarding potential AttributeError
            # since frame could be None.
            # https://github.com/napari/napari/pull/1635
            for _ in range(2):
                if inspect.isframe(frame):
                    frame = frame.f_back

            # Iterate frames while filename starts with path_prefix (part of Napari)
            n = 1
            while (
                inspect.isframe(frame)
                and inspect.isframe(frame.f_back)
                and inspect.iscode(frame.f_code)
                and (self.predicate(n, frame))
            ):
                n += 1
                frame = frame.f_back
            self.frame = frame
            if inspect.isframe(frame) and inspect.iscode(frame.f_code):
                self.namespace = ChainMap(frame.f_locals, frame.f_globals)
                self.names = (
                    *frame.f_code.co_varnames,
                    *frame.f_code.co_names,
                )

        finally:
            # We need to delete the frame explicitly according to the inspect
            # documentation for deterministic removal of the frame.
            # Otherwise, proper deletion is dependent on a cycle detector and
            # automatic garbage collection.
            # See handle_stackframe_without_leak example at the following URLs:
            # https://docs.python.org/3/library/inspect.html#the-interpreter-stack
            # https://bugs.python.org/issue543148
            del frame

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del self.namespace
        del self.names


def magic_name(value: Any, *, path_prefix: str = ROOT_DIR) -> str | None:
    """Fetch the name of the variable with the given value passed to the calling function.

    Parameters
    ----------
    value : any
        The value of the desired variable.
    path_prefix : absolute path-like, kwonly
        The path prefixes to ignore.

    Returns
    -------
    name : str or None
        Name of the variable, if found.
    """
    # Iterate frames while filename starts with path_prefix (part of Napari)
    with CallerFrame(
        lambda n, frame: frame.f_code.co_filename.startswith(path_prefix)
    ) as w:
        varmap = w.namespace
        names = w.names
        for name in names:
            if (
                name.isidentifier()
                and name in varmap
                and varmap[name] is value
            ):
                return name
        return None
