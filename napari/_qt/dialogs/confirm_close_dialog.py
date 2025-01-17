from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.settings import get_settings
from napari.utils.translations import trans


class ConfirmCloseDialog(QDialog):
    def __init__(
        self,
        parent,
        close_app=False,
        display_checkbox=True,
        extra_info='',
    ) -> None:
        super().__init__(parent)
        extra_info = f'\n\n{extra_info}' if extra_info else ''
        cancel_btn = QPushButton(trans._('Cancel'))
        close_btn = QPushButton(trans._('Close'))
        close_btn.setObjectName('warning_icon_btn')
        icon_label = QWidget()

        self.do_not_ask = QCheckBox(trans._('Do not ask in future'))
        self.do_not_ask.setVisible(display_checkbox)
        self._display_checkbox = display_checkbox

        if close_app:
            self.setWindowTitle(trans._('Close Application?'))
            text = trans._(
                "Do you want to close the application? ('{shortcut}' to confirm). This will close all Qt Windows in this process{extra_info}",
                shortcut=QKeySequence('Ctrl+Q').toString(
                    QKeySequence.NativeText
                ),
                extra_info=extra_info,
            )
            close_btn.setObjectName('error_icon_btn')
            close_btn.setShortcut(QKeySequence('Ctrl+Q'))
            icon_label.setObjectName('error_icon_element')
        else:
            self.setWindowTitle(trans._('Close Window?'))
            text = trans._(
                "Confirm to close window (or press '{shortcut}'){extra_info}",
                shortcut=QKeySequence('Ctrl+W').toString(
                    QKeySequence.NativeText
                ),
                extra_info=extra_info,
            )
            close_btn.setObjectName('warning_icon_btn')
            close_btn.setShortcut(QKeySequence('Ctrl+W'))
            icon_label.setObjectName('warning_icon_element')

        cancel_btn.clicked.connect(self.reject)
        close_btn.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout2.addWidget(icon_label)
        layout3 = QVBoxLayout()
        layout3.addWidget(QLabel(text))
        layout3.addWidget(self.do_not_ask)
        layout2.addLayout(layout3)
        layout4 = QHBoxLayout()
        layout4.addStretch(1)
        layout4.addWidget(cancel_btn)
        layout4.addWidget(close_btn)
        layout.addLayout(layout2)
        layout.addLayout(layout4)
        self.setLayout(layout)

        # for test purposes because of problem with shortcut testing:
        # https://github.com/pytest-dev/pytest-qt/issues/254
        self.close_btn = close_btn
        self.cancel_btn = cancel_btn

    def accept(self):
        if self._display_checkbox and self.do_not_ask.isChecked():
            get_settings().application.confirm_close_window = False
        super().accept()
