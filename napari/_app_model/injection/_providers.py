"""Non-Qt providers.

Qt providers can be found in `napari/_qt/_qapp_model/injection/_qproviders.py`.

Because `_provide_viewer` needs `_QtMainWindow` (otherwise returns `None`)
tests are in `napari/_tests/test_providers.py`, which are not run in headless mode.
"""

from typing import Optional

from napari import components, layers, viewer
from napari.utils._proxies import PublicOnlyProxy
from napari.utils.translations import trans


def _provide_viewer() -> Optional[viewer.Viewer]:
    """Provide `PublicOnlyProxy` (allows internal napari access) of current viewer."""
    if current_viewer := viewer.current_viewer():
        return PublicOnlyProxy(current_viewer)
    raise RuntimeError(
        trans._(
            "No current `Viewer` found. You can create one with `napari.Viewer()`.",
            deferred=True,
        )
    )


def _provide_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := _provide_viewer()) else None


def _provide_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := _provide_viewer()) else None


# syntax could be simplified after
# https://github.com/tlambert03/in-n-out/issues/31
PROVIDERS = [
    (_provide_viewer,),
    (_provide_active_layer,),
    (_provide_active_layer_list,),
]
