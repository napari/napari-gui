"""Provides a QtPluginErrReporter that allows the user report plugin errors."""

import contextlib

from napari_plugin_engine import standard_metadata
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import CodeSyntaxHighlight

from napari.plugins.exceptions import format_exceptions
from napari.settings import get_settings
from napari.utils.theme import get_theme
from napari.utils.translations import trans


class QtPluginErrReporter(QDialog):
    """Dialog that allows users to review and report PluginError tracebacks.

    Parameters
    ----------
    parent : QWidget, optional
        Optional parent widget for this widget.
    initial_plugin : str, optional
        If provided, errors from ``initial_plugin`` will be shown when the
        dialog is created, by default None

    Attributes
    ----------
    text_area : qtpy.QtWidgets.QTextEdit
        The text area where traceback information will be shown.
    plugin_combo : qtpy.QtWidgets.QComboBox
        The dropdown menu used to select the current plugin
    github_button : qtpy.QtWidgets.QPushButton
        A button that, when pressed, will open an issue at the current plugin's
        github issue tracker, prepopulated with a formatted traceback.  Button
        is only visible if a github URL is detected in the package metadata for
        the current plugin.
    clipboard_button : qtpy.QtWidgets.QPushButton
        A button that, when pressed, copies the current traceback information
        to the clipboard.  (HTML tags are removed in the copied text.)
    plugin_meta : qtpy.QtWidgets.QLabel
        A label that will show available plugin metadata (such as home page).
    """

    NULL_OPTION = trans._('select plugin... ')

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        initial_plugin: str | None = None,
    ) -> None:
        super().__init__(parent)
        from napari.plugins import plugin_manager

        self.plugin_manager = plugin_manager

        self.setWindowTitle(trans._('Recorded Plugin Exceptions'))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        self.text_area = QTextEdit()
        theme = get_theme(get_settings().appearance.theme)
        self._highlight = CodeSyntaxHighlight(
            self.text_area.document(), 'python', theme.syntax_style
        )
        self.text_area.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.text_area.setMinimumWidth(360)

        # Create plugin dropdown menu
        self.plugin_combo = QComboBox()
        self.plugin_combo.addItem(self.NULL_OPTION)
        bad_plugins = [e.plugin_name for e in self.plugin_manager.get_errors()]
        self.plugin_combo.addItems(sorted(set(bad_plugins)))
        self.plugin_combo.currentTextChanged.connect(self.set_plugin)
        self.plugin_combo.setCurrentText(self.NULL_OPTION)

        # create github button (gets connected in self.set_plugin)
        self.github_button = QPushButton(trans._('Open issue on GitHub'), self)
        self.github_button.setToolTip(
            trans._(
                "Open a web browser to submit this error log\nto the developer's GitHub issue tracker",
            )
        )
        self.github_button.hide()

        # create copy to clipboard button
        self.clipboard_button = QPushButton()
        self.clipboard_button.hide()
        self.clipboard_button.setObjectName('QtCopyToClipboardButton')
        self.clipboard_button.setToolTip(
            trans._('Copy error log to clipboard')
        )
        self.clipboard_button.clicked.connect(self.copyToClipboard)

        # plugin_meta contains a URL to the home page, (and/or other details)
        self.plugin_meta = QLabel('', parent=self)
        self.plugin_meta.setObjectName('pluginInfo')
        self.plugin_meta.setTextFormat(Qt.TextFormat.RichText)
        self.plugin_meta.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.plugin_meta.setOpenExternalLinks(True)
        self.plugin_meta.setAlignment(Qt.AlignmentFlag.AlignRight)

        # make layout
        row_1_layout = QHBoxLayout()
        row_1_layout.setContentsMargins(11, 5, 10, 0)
        row_1_layout.addStretch(1)
        row_1_layout.addWidget(self.plugin_meta)
        row_2_layout = QHBoxLayout()
        row_2_layout.setContentsMargins(11, 5, 10, 0)
        row_2_layout.addWidget(self.plugin_combo)
        row_2_layout.addStretch(1)
        row_2_layout.addWidget(self.github_button)
        row_2_layout.addWidget(self.clipboard_button)
        row_2_layout.setSpacing(5)
        self.layout.addLayout(row_1_layout)
        self.layout.addLayout(row_2_layout)
        self.layout.addWidget(self.text_area, 1)
        self.setMinimumWidth(750)
        self.setMinimumHeight(600)

        if initial_plugin:
            self.set_plugin(initial_plugin)

    def set_plugin(self, plugin: str) -> None:
        """Set the current plugin shown in the dropdown and text area.

        Parameters
        ----------
        plugin : str
            name of a plugin that has created an error this session.
        """
        self.github_button.hide()
        self.clipboard_button.hide()
        with contextlib.suppress(RuntimeError, TypeError):
            self.github_button.clicked.disconnect()
            # when disconnecting a non-existent signal
            # PySide2 raises runtimeError, PyQt5 raises TypeError

        if not plugin or (plugin == self.NULL_OPTION):
            self.plugin_meta.setText('')
            self.text_area.setText('')
            return

        if not self.plugin_manager.get_errors(plugin):
            raise ValueError(
                trans._(
                    "No errors reported for plugin '{plugin}'", plugin=plugin
                )
            )

        self.plugin_combo.setCurrentText(plugin)

        err_string = format_exceptions(plugin, as_html=False, color='NoColor')
        self.text_area.setText(err_string)
        self.clipboard_button.show()

        # set metadata and outbound links/buttons
        err0 = self.plugin_manager.get_errors(plugin)[0]
        meta = standard_metadata(err0.plugin) if err0.plugin else {}
        meta_text = ''
        if not meta:
            self.plugin_meta.setText(meta_text)
            return

        url = meta.get('url')
        if url:
            meta_text += (
                '<span style="color:#999;">plugin home page:&nbsp;&nbsp;'
                f'</span><a href="{url}" style="color:#999">{url}</a>'
            )
            if 'github.com' in url:

                def onclick():
                    import webbrowser

                    err = format_exceptions(plugin, as_html=False)
                    err = (
                        '<!--Provide detail on the error here-->\n\n\n\n'
                        '<details>\n<summary>Traceback from napari</summary>'
                        f'\n\n```\n{err}\n```\n</details>'
                    )
                    url = f'{meta.get("url")}/issues/new?&body={err}'
                    webbrowser.open(url, new=2)

                self.github_button.clicked.connect(onclick)
                self.github_button.show()
        self.plugin_meta.setText(meta_text)

    def copyToClipboard(self) -> None:
        """Copy current plugin traceback info to clipboard as plain text."""
        plugin = self.plugin_combo.currentText()
        err_string = format_exceptions(plugin, as_html=False)
        cb = QGuiApplication.clipboard()
        cb.setText(err_string)
