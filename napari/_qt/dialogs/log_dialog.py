from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
)

from napari._qt.dialogs.qt_about import QtCopyToClipboardButton
from napari.utils._logging import LOG_STREAM
from napari.utils.translations import trans


class LogDialog(QDialog):
    def __init__(
        self,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.layout = QVBoxLayout()

        # Description
        title_label = QLabel(trans._('napari log'))
        title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.layout.addWidget(title_label)

        # level selection
        self.level_selection = QComboBox()
        self.level_selection.addItems(
            ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        )
        self.layout.addWidget(self.level_selection)

        # text filter
        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText('text filter')
        self.layout.addWidget(self.text_filter)

        # log text box
        self.log_text_box = QTextEdit()
        self.log_text_box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.log_text_box.setLineWrapMode(QTextEdit.NoWrap)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.log_text_box.setFont(font)

        # Add text copy button
        self.infoCopyButton = QtCopyToClipboardButton(self.log_text_box)
        self.info_layout = QHBoxLayout()
        self.info_layout.addWidget(self.log_text_box, 1)
        self.info_layout.addWidget(
            self.infoCopyButton, 0, Qt.AlignmentFlag.AlignTop
        )
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addLayout(self.info_layout)

        self.setLayout(self.layout)

        self.level_selection.currentTextChanged.connect(self._on_change)
        self.text_filter.textChanged.connect(self._on_change)

        # TODO: super laggy when open :/
        LOG_STREAM.changed.connect(self._on_new_message)
        self._prev_pos = None
        self.log_text_box.verticalScrollBar().rangeChanged.connect(
            self._jump_to_pos
        )
        self._on_change()
        self._jump_to_pos()

    @staticmethod
    def showLog(parent=None):
        d = LogDialog(parent)
        d.setObjectName('LogDialog')
        d.setWindowTitle(trans._('napari Log'))
        d.setModal(False)
        d.show()

    def _on_new_message(self, event=None):
        self._prev_pos = self._scroll_pos()

        log = LOG_STREAM.get_filtered_logs_html(
            self.level_selection.currentText(),
            self.text_filter.text(),
            last_only=True,
        )[0]
        self.log_text_box.append(log)

    def _on_change(self, event=None):
        self._prev_pos = self._scroll_pos()

        logs = LOG_STREAM.get_filtered_logs_html(
            self.level_selection.currentText(), self.text_filter.text()
        )
        self.log_text_box.setHtml('<br>'.join(logs))

    def _jump_to_pos(self, event=None):
        # for some reason using scrollbar.setValue() doesn't keep up,
        # cursor's better
        scrollbar = self.log_text_box.verticalScrollBar()
        if self._prev_pos is None:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(self._prev_pos)

        # self.log_text_box.moveCursor(self.log_text_box.textCursor().End)
        # self.log_text_box.moveCursor(self.log_text_box.textCursor().StartOfLine)
        # self.log_text_box.ensureCursorVisible()

    def _scroll_pos(self):
        scrollbar = self.log_text_box.verticalScrollBar()
        curr = scrollbar.value()
        if curr == scrollbar.maximum():
            return None
        return curr
