from __future__ import annotations

import weakref
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from weakref import ReferenceType

from magicgui.widgets import FunctionGui
from typing_extensions import Self

from napari._pydantic_compat import BaseModel, validator
from napari.layers.base.base import Layer


class Source(BaseModel):
    """An object to store the provenance of a layer.

    Parameters
    ----------
    path: str, optional
        filpath/url associated with layer
    reader_plugin: str, optional
        name of reader plugin that loaded the file (if applicable)
    sample: Tuple[str, str], optional
        Tuple of (sample_plugin, sample_name), if layer was loaded via
        `viewer.open_sample`.
    widget: FunctionGui, optional
        magicgui widget, if the layer was added via a magicgui widget.
    parent: Layer, optional
        parent layer if the layer is a duplicate.
    """

    path: str | None = None
    reader_plugin: str | None = None
    sample: tuple[str, str] | None = None
    widget: FunctionGui | None = None
    parent: Layer | None = None

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    @validator('parent', allow_reuse=True)
    def make_weakref(cls, layer: Layer) -> ReferenceType[Layer]:
        return weakref.ref(layer)

    def __deepcopy__(self, memo: Any) -> Self:
        """Custom deepcopy implementation.

        this prevents deep copy. `Source` doesn't really need to be copied
        (i.e. if we deepcopy a layer, it essentially has the same `Source`).
        Moreover, deepcopying a widget is challenging, and maybe odd anyway.
        """
        return self


# layer source context management

_LAYER_SOURCE: ContextVar[dict | None] = ContextVar(
    '_LAYER_SOURCE', default=None
)


@contextmanager
def layer_source(**source_kwargs: Any) -> Generator[None, None, None]:
    """Creates context in which all layers will be given `source_kwargs`.

    The module-level variable `_LAYER_SOURCE` holds a set of key-value pairs
    that can be used to create a new `Source` object.  Any routine in napari
    that may result in the creation of a new layer (such as opening a file,
    using a particular plugin, or calling a magicgui widget) can use this
    context manager to declare that any layers created within the context
    result from a specific source. (This applies even if the layer
    isn't "directly" created in the context, but perhaps in some sub-function
    within the context).

    `Layer.__init__` will call :func:`current_source`, to query the current
    state of the `_LAYER_SOURCE` variable.

    Contexts may be stacked, meaning a given layer.source can reflect the
    actions of multiple events (for instance, an `open_sample` call that in
    turn resulted in a `reader_plugin` opening a file).  However, the "deepest"
    context will "win" in the case where multiple calls to `layer_source`
    provide conflicting values.

    Parameters
    ----------
    **source_kwargs
        keys/values should be valid parameters for :class:`Source`.

    Examples
    --------

    >>> with layer_source(path='file.ext', reader_plugin='plugin'):  # doctest: +SKIP
    ...     points = some_function_that_creates_points()
    ...
    >>> assert points.source == Source(path='file.ext', reader_plugin='plugin')  # doctest: +SKIP

    """
    token = _LAYER_SOURCE.set({**(_LAYER_SOURCE.get() or {}), **source_kwargs})
    try:
        yield
    finally:
        _LAYER_SOURCE.reset(token)


def current_source() -> Source:
    """Get the current layer :class:`Source` (inferred from context).

    The main place this function is used is in :meth:`Layer.__init__`.
    """
    return Source(**(_LAYER_SOURCE.get() or {}))
