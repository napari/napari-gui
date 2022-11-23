import warnings
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
import napari


def raise_():
    x = 1  # noqa: F841
    y = 'a string'  # noqa: F841
    import something_that_does_not_exist   # noqa: F401


def warn_():
    warnings.warn("warning!")


viewer = napari.Viewer()
layer_buttons = viewer.window._qt_viewer.layerButtons
err_btn = QtViewerPushButton('warning', 'new Error', raise_)
warn_btn = QtViewerPushButton('warning', 'new Warn', warn_)
layer_buttons.layout().insertWidget(3, warn_btn)
layer_buttons.layout().insertWidget(3, err_btn)

napari.run()
