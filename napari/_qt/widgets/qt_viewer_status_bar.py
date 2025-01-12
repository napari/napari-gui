"""Status bar widget on the viewer MainWindow"""

from typing import TYPE_CHECKING, cast

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QFontMetrics, QResizeEvent
from qtpy.QtWidgets import QLabel, QStatusBar, QWidget
from superqt import QElidingLabel

from napari._qt.dialogs.qt_activity_dialog import ActivityToggleItem
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import _QtMainWindow


class ViewerStatusBar(QStatusBar):
    def __init__(self, parent: '_QtMainWindow') -> None:
        super().__init__(parent=parent)

        self._status = QLabel(trans._('Ready'))
        self._status.setContentsMargins(0, 0, 0, 0)

        self._layer_base = QElidingLabel(trans._(''))
        self._layer_base.setObjectName('layer_base status')
        self._layer_base.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._layer_base.setMinimumSize(100, 16)
        self._layer_base.setContentsMargins(0, 0, 0, 0)

        self._plugin_reader = QElidingLabel(trans._(''))
        self._plugin_reader.setObjectName('plugin-reader status')
        self._plugin_reader.setMinimumSize(80, 16)
        self._plugin_reader.setContentsMargins(0, 0, 0, 0)
        self._plugin_reader.setElideMode(Qt.TextElideMode.ElideMiddle)

        self._source_type = QLabel('')
        self._source_type.setObjectName('source-type status')
        self._source_type.setContentsMargins(0, 0, 0, 0)

        self._coordinates = QElidingLabel('')
        self._coordinates.setObjectName('coordinates status')
        self._coordinates.setMinimumSize(100, 16)
        self._coordinates.setContentsMargins(0, 0, 0, 0)

        self._help = QElidingLabel('')
        self._help.setObjectName('help status')
        self._help.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        main_widget = StatusBarWidget(
            self._status,
            self._layer_base,
            self._source_type,
            self._plugin_reader,
            self._coordinates,
            self._help,
        )
        self.addWidget(main_widget, 1)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        # FIXME: feels weird to set this here.
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)

    def setHelpText(self, text: str) -> None:
        self._help.setText(text)

    def setStatusText(
        self,
        text: str = '',
        layer_base: str = '',
        source_type=None,
        plugin: str = '',
        coordinates: str = '',
    ) -> None:
        # The method used to set a single value as the status and not
        # all the layer information.
        self._status.setText(text)

        self._layer_base.setVisible(bool(layer_base))
        self._layer_base.setText(layer_base)

        self._source_type.setVisible(bool(source_type))
        if source_type:
            self._source_type.setText(f'{source_type}: ')

        self._plugin_reader.setVisible(bool(plugin))

        self._plugin_reader.setText(plugin)

        self._coordinates.setVisible(bool(coordinates))
        self._coordinates.setText(coordinates)

    def _toggle_activity_dock(self, visible: bool | None = None):
        par = cast('_QtMainWindow', self.parent())
        if visible is None:
            visible = not par._activity_dialog.isVisible()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(
                Qt.ArrowType.DownArrow
            )
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.ArrowType.UpArrow)


class StatusBarWidget(QWidget):
    def __init__(
        self,
        status_label: QLabel,
        layer_label: QLabel,
        source_label: QLabel,
        plugin_label: QLabel,
        coordinates_label: QLabel,
        help_label: QLabel,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent=parent)
        self._status_label = status_label
        self._layer_label = layer_label
        self._source_label = source_label
        self._plugin_label = plugin_label
        self._coordinates_label = coordinates_label
        self._help_label = help_label

        self._status_label.setParent(self)
        self._layer_label.setParent(self)
        self._source_label.setParent(self)
        self._plugin_label.setParent(self)
        self._coordinates_label.setParent(self)
        self._help_label.setParent(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.do_layout()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.LayoutRequest:
            self.do_layout()
        return super().event(event)

    @staticmethod
    def _calc_width(fm: QFontMetrics, label: QLabel) -> int:
        # magical nuber +2 is from superqt code
        # magical number +12 is from experiments
        # Adding this values is required to avoid the text to be elided
        # if there is enough space to show it.
        return (
            (
                fm.boundingRect(label.text()).width()
                + label.margin() * 2
                + 2
                + 12
            )
            if label.isVisible()
            else 0
        )

    def do_layout(self):
        width = self.width()
        height = self.height()

        fm = QFontMetrics(self._status_label.font())

        status_width = self._calc_width(fm, self._status_label)
        layer_width = self._calc_width(fm, self._layer_label)
        source_width = self._calc_width(fm, self._source_label)
        plugin_width = self._calc_width(fm, self._plugin_label)
        coordinates_width = self._calc_width(fm, self._coordinates_label)

        base_width = (
            status_width
            + layer_width
            + source_width
            + plugin_width
            + coordinates_width
        )

        help_width = max(0, width - base_width)

        if coordinates_width:
            help_width = 0

        if base_width > width:
            self._help_label.setVisible(False)
            layer_width = max(
                int((layer_width / base_width) * layer_width),
                min(self._layer_label.minimumWidth(), layer_width),
            )
            source_width = max(
                int((source_width / base_width) * source_width),
                min(self._source_label.minimumWidth(), source_width),
            )
            plugin_width = max(
                int((plugin_width / base_width) * plugin_width),
                min(self._plugin_label.minimumWidth(), plugin_width),
            )
            coordinates_width = (
                base_width
                - status_width
                - layer_width
                - source_width
                - plugin_width
            )

        else:
            self._help_label.setVisible(True)

        self._status_label.setGeometry(0, 0, status_width, height)
        shift = status_width
        self._layer_label.setGeometry(shift, 0, layer_width, height)
        shift += layer_width
        self._source_label.setGeometry(shift, 0, source_width, height)
        shift += source_width
        self._plugin_label.setGeometry(shift, 0, plugin_width, height)
        shift += plugin_width
        self._coordinates_label.setGeometry(
            shift, 0, coordinates_width, height
        )
        shift += coordinates_width
        self._help_label.setGeometry(shift, 0, help_width, height)
