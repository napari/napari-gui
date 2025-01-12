import warnings
from functools import reduce
from itertools import count
from operator import ior
from typing import TYPE_CHECKING, Union
from weakref import ReferenceType, ref

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari._qt.utils import combine_widgets, qt_signals_blocked
from napari.settings import get_settings
from napari.utils.translations import trans

if TYPE_CHECKING:
    from magicgui.widgets import Widget

    from napari._qt.qt_viewer import QtViewer

counter = count()
_sentinel = object()

_SHORTCUT_DEPRECATION_STRING = trans._(
    'The shortcut parameter is deprecated since version 0.4.8, please use the action and shortcut manager APIs. The new action manager and shortcut API allow user configuration and localisation. (got {shortcut})',
    shortcut='{shortcut}',
)

dock_area_to_str = {
    Qt.DockWidgetArea.LeftDockWidgetArea: 'left',
    Qt.DockWidgetArea.RightDockWidgetArea: 'right',
    Qt.DockWidgetArea.TopDockWidgetArea: 'top',
    Qt.DockWidgetArea.BottomDockWidgetArea: 'bottom',
}


class QtViewerDockWidget(QDockWidget):
    """Wrap a QWidget in a QDockWidget and forward viewer events

    Parameters
    ----------
    qt_viewer : QtViewer
        The QtViewer instance that this dock widget will belong to.
    widget : QWidget or magicgui.widgets.Widget
        `widget` that will be added as QDockWidget's main widget.
    name : str
        Name of dock widget.
    area : str
        Side of the main window to which the new dock widget will be added.
        Must be in {'left', 'right', 'top', 'bottom'}
    allowed_areas : list[str], optional
        Areas, relative to main window, that the widget is allowed dock.
        Each item in list must be in {'left', 'right', 'top', 'bottom'}
        By default, all areas are allowed.
    shortcut : str, optional
        Keyboard shortcut to appear in dropdown menu.
        .. deprecated:: 0.4.8

            The shortcut parameter is deprecated since version 0.4.8, please use
            the action and shortcut manager APIs. The new action manager and
            shortcut API allow user configuration and localisation.
    add_vertical_stretch : bool, optional
        Whether to add stretch to the bottom of vertical widgets (pushing
        widgets up towards the top of the allotted area, instead of letting
        them distribute across the vertical space).  By default, True.
    """

    def __init__(
        self,
        qt_viewer,
        widget: Union[QWidget, 'Widget'],
        *,
        name: str = '',
        area: str = 'right',
        allowed_areas: list[str] | None = None,
        shortcut=_sentinel,
        object_name: str = '',
        add_vertical_stretch=True,
        close_btn=True,
    ) -> None:
        self._ref_qt_viewer: ReferenceType[QtViewer] = ref(qt_viewer)
        super().__init__(name)
        self._parent = qt_viewer
        self.name = name
        self._close_btn = close_btn

        areas = {
            'left': Qt.DockWidgetArea.LeftDockWidgetArea,
            'right': Qt.DockWidgetArea.RightDockWidgetArea,
            'top': Qt.DockWidgetArea.TopDockWidgetArea,
            'bottom': Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        if area not in areas:
            raise ValueError(
                trans._(
                    'area argument must be in {areas}',
                    deferred=True,
                    areas=list(areas.keys()),
                )
            )
        self.area = area
        self.qt_area = areas[area]
        if shortcut is not _sentinel:
            warnings.warn(
                _SHORTCUT_DEPRECATION_STRING.format(shortcut=shortcut),
                FutureWarning,
                stacklevel=2,
            )
        else:
            shortcut = None
        self._shortcut = shortcut

        if allowed_areas:
            if not isinstance(allowed_areas, (list, tuple)):
                raise TypeError(
                    trans._(
                        '`allowed_areas` must be a list or tuple',
                        deferred=True,
                    )
                )

            if any(area not in areas for area in allowed_areas):
                raise ValueError(
                    trans._(
                        'all allowed_areas argument must be in {areas}',
                        deferred=True,
                        areas=list(areas.keys()),
                    )
                )
            allowed_areas = reduce(ior, [areas[a] for a in allowed_areas])
        else:
            allowed_areas = Qt.DockWidgetArea.AllDockWidgetAreas
        self.setAllowedAreas(allowed_areas)
        self.setMinimumHeight(50)
        self.setMinimumWidth(50)
        # FIXME:
        self.setObjectName(object_name or name)

        is_vertical = area in {'left', 'right'}
        widget = combine_widgets(widget, vertical=is_vertical)
        self.setWidget(widget)
        if is_vertical and add_vertical_stretch:
            self._maybe_add_vertical_stretch(widget)

        self._features = self.features()
        self.dockLocationChanged.connect(self._set_title_orientation)

        # custom title bar
        self.title = QtCustomTitleBar(
            self, title=self.name, close_btn=close_btn
        )
        self.setTitleBarWidget(self.title)
        self.visibilityChanged.connect(self._on_visibility_changed)

        self.dockLocationChanged.connect(self._update_default_dock_area)

    def _update_default_dock_area(self, value):
        if value not in dock_area_to_str:
            return
        settings = get_settings()
        settings.application.plugin_widget_positions[self.name] = (
            dock_area_to_str[value]
        )
        settings._maybe_save()

    @property
    def _parent(self):
        """
        Let's make sure parent always a weakref:

            1) parent is likely to always exists after child
            2) even if not strictly necessary it make it easier to view reference cycles.
        """
        return self._ref_parent()

    @_parent.setter
    def _parent(self, obj):
        self._ref_parent = ref(obj)

    def destroyOnClose(self):
        """Destroys dock plugin dock widget when 'x' is clicked."""
        from napari.viewer import Viewer

        viewer = self._ref_qt_viewer().viewer
        if isinstance(viewer, Viewer):
            viewer.window.remove_dock_widget(self)

    def _maybe_add_vertical_stretch(self, widget):
        """Add vertical stretch to the bottom of a vertical layout only

        ...if there is not already a widget that wants vertical space
        (like a textedit or listwidget or something).
        """
        exempt_policies = {
            QSizePolicy.Expanding,
            QSizePolicy.MinimumExpanding,
            QSizePolicy.Ignored,
        }
        if widget.sizePolicy().verticalPolicy() in exempt_policies:
            return

        # not uncommon to see people shadow the builtin layout() method
        # which breaks our ability to add vertical stretch...
        try:
            wlayout = widget.layout()
            if wlayout is None:
                return
        except TypeError:
            return

        for i in range(wlayout.count()):
            wdg = wlayout.itemAt(i).widget()
            if (
                wdg is not None
                and wdg.sizePolicy().verticalPolicy() in exempt_policies
            ):
                return

        # not all widgets have addStretch...
        if hasattr(wlayout, 'addStretch'):
            wlayout.addStretch(next(counter))

    @property
    def shortcut(self):
        warnings.warn(
            _SHORTCUT_DEPRECATION_STRING,
            FutureWarning,
            stacklevel=2,
        )
        return self._shortcut

    def setFeatures(self, features):
        super().setFeatures(features)
        self._features = self.features()

    def keyPressEvent(self, event):
        # if you subclass QtViewerDockWidget and override the keyPressEvent
        # method, be sure to call super().keyPressEvent(event) at the end of
        # your method to pass uncaught key-combinations to the viewer.
        return self._ref_qt_viewer().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # if you subclass QtViewerDockWidget and override the keyReleaseEvent
        # method, be sure to call super().keyReleaseEvent(event) at the end of
        # your method to pass uncaught key-combinations to the viewer.
        return self._ref_qt_viewer().keyReleaseEvent(event)

    def _set_title_orientation(self, area):
        if area in (
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        ):
            features = self._features
            if features & self.DockWidgetFeature.DockWidgetVerticalTitleBar:
                features = (
                    features
                    ^ self.DockWidgetFeature.DockWidgetVerticalTitleBar
                )
        else:
            features = (
                self._features
                | self.DockWidgetFeature.DockWidgetVerticalTitleBar
            )
        self.setFeatures(features)

    @property
    def is_vertical(self):
        if not self.isFloating():
            par = self.parent()
            if par and hasattr(par, 'dockWidgetArea'):
                return par.dockWidgetArea(self) in (
                    Qt.DockWidgetArea.LeftDockWidgetArea,
                    Qt.DockWidgetArea.RightDockWidgetArea,
                )
        return self.size().height() > self.size().width()

    def _on_visibility_changed(self, visible):
        if not visible:
            return
        with qt_signals_blocked(self):
            self.setTitleBarWidget(None)
            if not self.isFloating():
                self.title = QtCustomTitleBar(
                    self,
                    title=self.name,
                    vertical=not self.is_vertical,
                    close_btn=self._close_btn,
                )
                self.setTitleBarWidget(self.title)

    def setWidget(self, widget):
        widget._parent = self
        self.setFocusProxy(widget)
        super().setWidget(widget)


class QtCustomTitleBar(QLabel):
    """A widget to be used as the titleBar in the QtViewerDockWidget.

    Keeps vertical size minimal, has a hand cursor and styles (in stylesheet)
    for hover. Close and float buttons.

    Parameters
    ----------
    parent : QDockWidget
        The QtViewerDockWidget to which this titlebar belongs
    title : str
        A string to put in the titlebar.
    vertical : bool
        Whether this titlebar is oriented vertically or not.
    """

    def __init__(
        self, parent, title: str = '', vertical=False, close_btn=True
    ) -> None:
        super().__init__(parent)
        self.setObjectName('QtCustomTitleBar')
        self.setProperty('vertical', str(vertical))
        self.vertical = vertical
        self.setToolTip(trans._('drag to move. double-click to float'))

        line = QFrame(self)
        line.setObjectName('QtCustomTitleBarLine')

        self.hide_button = QPushButton(self)
        self.hide_button.setToolTip(trans._('hide this panel'))
        self.hide_button.setObjectName('QTitleBarHideButton')
        self.hide_button.setCursor(Qt.CursorShape.ArrowCursor)
        self.hide_button.clicked.connect(lambda: self.parent().close())

        self.float_button = QPushButton(self)
        self.float_button.setToolTip(trans._('float this panel'))
        self.float_button.setObjectName('QTitleBarFloatButton')
        self.float_button.setCursor(Qt.CursorShape.ArrowCursor)
        self.float_button.clicked.connect(
            lambda: self.parent().setFloating(not self.parent().isFloating())
        )
        self.title: QLabel = QLabel(title, self)
        self.title.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        )

        if close_btn:
            self.close_button = QPushButton(self)
            self.close_button.setToolTip(trans._('close this panel'))
            self.close_button.setObjectName('QTitleBarCloseButton')
            self.close_button.setCursor(Qt.CursorShape.ArrowCursor)
            self.close_button.clicked.connect(
                lambda: self.parent().destroyOnClose()
            )

        if vertical:
            layout = QVBoxLayout()
            layout.setSpacing(4)
            layout.setContentsMargins(0, 8, 0, 8)
            line.setFixedWidth(1)
            if close_btn:
                layout.addWidget(
                    self.close_button, 0, Qt.AlignmentFlag.AlignHCenter
                )
            layout.addWidget(
                self.hide_button, 0, Qt.AlignmentFlag.AlignHCenter
            )
            layout.addWidget(
                self.float_button, 0, Qt.AlignmentFlag.AlignHCenter
            )
            layout.addWidget(line, 0, Qt.AlignmentFlag.AlignHCenter)
            self.title.hide()

        else:
            layout = QHBoxLayout()
            layout.setSpacing(4)
            layout.setContentsMargins(8, 1, 8, 0)
            line.setFixedHeight(1)
            if close_btn:
                layout.addWidget(self.close_button)

            layout.addWidget(self.hide_button)
            layout.addWidget(self.float_button)
            layout.addWidget(line)
            layout.addWidget(self.title)

        self.setLayout(layout)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def sizeHint(self):
        # this seems to be the correct way to set the height of the titlebar
        szh = super().sizeHint()
        if self.vertical:
            szh.setWidth(20)
        else:
            szh.setHeight(20)
        return szh
