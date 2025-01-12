from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.utils.migrations import rename_argument
from napari.utils.progress import cancelable_progress, progress
from napari.utils.translations import trans


class QtLabeledProgressBar(QWidget):
    """QProgressBar with QLabels for description and ETA."""

    def __init__(
        self, parent: QWidget | None = None, prog: progress = None
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.progress = prog

        self.qt_progress_bar = QProgressBar()
        self.description_label = QLabel()
        self.eta_label = QLabel()
        self.cancel_button = QPushButton(trans._('Cancel'))
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setVisible(isinstance(prog, cancelable_progress))
        base_layout = QVBoxLayout()

        pbar_layout = QHBoxLayout()
        pbar_layout.addWidget(self.description_label)
        pbar_layout.addWidget(self.qt_progress_bar)
        pbar_layout.addWidget(self.eta_label)
        pbar_layout.addWidget(self.cancel_button)
        base_layout.addLayout(pbar_layout)

        line = QFrame(self)
        line.setObjectName('QtCustomTitleBarLine')
        line.setFixedHeight(1)
        base_layout.addWidget(line)

        self.setLayout(base_layout)

    @rename_argument(
        from_name='min',
        to_name='min_val',
        version='0.6.0',
        since_version='0.4.18',
    )
    @rename_argument(
        from_name='max',
        to_name='max_val',
        version='0.6.0',
        since_version='0.4.18',
    )
    def setRange(self, min_val, max_val):
        self.qt_progress_bar.setRange(min_val, max_val)

    def setValue(self, value):
        self.qt_progress_bar.setValue(value)
        QApplication.processEvents()

    def setDescription(self, value):
        if not value.endswith(': '):
            value = f'{value}: '
        self.description_label.setText(value)
        QApplication.processEvents()

    def _set_value(self, event):
        self.setValue(event.value)

    def _get_value(self):
        return self.qt_progress_bar.value()

    def _set_description(self, event):
        self.setDescription(event.value)

    def _make_indeterminate(self, event):
        self.setRange(0, 0)

    def _set_eta(self, event):
        self.eta_label.setText(event.value)

    def _set_total(self, event):
        self.setRange(0, event.value)

    def _cancel(self):
        self.cancel_button.setText(trans._('Cancelling...'))
        self.progress.cancel()
        self.cancel_button.setText(trans._('Canceled'))


class QtProgressBarGroup(QWidget):
    """One or more QtLabeledProgressBars with a QFrame line separator at the bottom"""

    def __init__(
        self,
        qt_progress_bar: QtLabeledProgressBar,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        pbr_group_layout = QVBoxLayout()
        pbr_group_layout.addWidget(qt_progress_bar)
        pbr_group_layout.setContentsMargins(0, 0, 0, 0)

        line = QFrame(self)
        line.setObjectName('QtCustomTitleBarLine')
        line.setFixedHeight(1)
        pbr_group_layout.addWidget(line)

        self.setLayout(pbr_group_layout)
