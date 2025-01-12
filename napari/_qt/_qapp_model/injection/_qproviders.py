"""Qt providers.

Any non-Qt providers should be added inside `napari/_app_model/injection/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from napari import components, layers, viewer
from napari.utils._proxies import PublicOnlyProxy
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window
    from napari._qt.qt_viewer import QtViewer


def _provide_viewer(public_proxy: bool = True) -> viewer.Viewer | None:
    """Provide `PublicOnlyProxy` (allows internal napari access) of current viewer."""
    if current_viewer := viewer.current_viewer():
        if public_proxy:
            return PublicOnlyProxy(current_viewer)
        return current_viewer
    return None


def _provide_viewer_or_raise(
    msg: str = '', public_proxy: bool = False
) -> viewer.Viewer:
    viewer = _provide_viewer(public_proxy)
    if viewer:
        return viewer
    if msg:
        msg = ' ' + msg
    raise RuntimeError(
        trans._(
            'No current `Viewer` found.{msg}',
            deferred=True,
            msg=msg,
        )
    )


def _provide_qt_viewer() -> QtViewer | None:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._qt_viewer
    return None


def _provide_qt_viewer_or_raise(msg: str = '') -> QtViewer:
    qt_viewer = _provide_qt_viewer()
    if qt_viewer:
        return qt_viewer
    if msg:
        msg = ' ' + msg
    raise RuntimeError(
        trans._(
            'No current `QtViewer` found.{msg}',
            deferred=True,
            msg=msg,
        )
    )


def _provide_window() -> Window | None:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._window
    return None


def _provide_window_or_raise(msg: str = '') -> Window:
    window = _provide_window()
    if window:
        return window
    if msg:
        msg = ' ' + msg
    raise RuntimeError(
        trans._(
            'No current `Window` found.{msg}',
            deferred=True,
            msg=msg,
        )
    )


def _provide_active_layer() -> layers.Layer | None:
    return v.layers.selection.active if (v := _provide_viewer()) else None


def _provide_active_layer_list() -> components.LayerList | None:
    return v.layers if (v := _provide_viewer()) else None


# syntax could be simplified after
# https://github.com/tlambert03/in-n-out/issues/31
QPROVIDERS = [
    (_provide_viewer,),
    (_provide_qt_viewer,),
    (_provide_window,),
    (_provide_active_layer,),
    (_provide_active_layer_list,),
]
