from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QFont, QIntValidator
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from napari.utils.translations import trans


class QtFontSizePreview(QFrame):
    """
    Widget that displays a preview text.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    text : str, optional
        Preview text to display. Default is None.
    """

    def __init__(
        self, parent: QWidget = None, text: str | None = None
    ) -> None:
        super().__init__(parent)

        self._text = text or ''

        # Widget
        self._preview = QPlainTextEdit(self)

        # Widget setup
        self._preview.setReadOnly(True)
        self._preview.setPlainText(self._text)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self._preview)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def sizeHint(self):
        """Override Qt method."""
        return QSize(100, 80)

    def text(self) -> str:
        """Return the current preview text.

        Returns
        -------
        str
            The current preview text.
        """
        return self._text

    def setText(self, text: str):
        """Set the current preview text.

        Parameters
        ----------
        text : str
            The current preview text.
        """
        self._text = text
        self._preview.setPlainText(text)


class QtSizeSliderPreviewWidget(QWidget):
    """
    Widget displaying a description, textedit and slider to adjust font size
    with preview.

    Parameters
    ----------
    parent : qtpy.QtWidgets.QWidget, optional
        Default is None.
    description : str, optional
        Default is "".
    preview_text : str, optional
        Default is "".
    value : int, optional
        Default is None.
    min_value : int, optional
        Default is 1.
    max_value : int, optional
        Default is 50.
    unit : str, optional
        Default is "px".
    """

    valueChanged = Signal(int)

    def __init__(
        self,
        parent: QWidget = None,
        description: str | None = None,
        preview_text: str | None = None,
        value: int | None = None,
        min_value: int = 1,
        max_value: int = 50,
        unit: str = 'px',
    ) -> None:
        super().__init__(parent)

        description = description or ''
        preview_text = preview_text or ''
        self._value = value if value else self.fontMetrics().height()
        self._min_value = min_value
        self._max_value = max_value

        # Widget
        self._lineedit = QLineEdit()
        self._description_label = QLabel(self)
        self._unit_label = QLabel(self)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        self._preview = QtFontSizePreview(self)
        self._preview_label = QLabel(self)
        self._validator = None

        # Widgets setup
        self._description_label.setText(description)
        self._description_label.setWordWrap(True)
        self._unit_label.setText(unit)
        self._lineedit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._slider_min_label.setText(str(min_value))
        self._slider_max_label.setText(str(max_value))
        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)
        self._preview.setText(preview_text)
        self._preview_label.setText(trans._('preview'))
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setFocusProxy(self._lineedit)

        # Layout
        left_bottom_layout = QHBoxLayout()
        left_bottom_layout.addWidget(self._lineedit)
        left_bottom_layout.addWidget(self._unit_label)
        left_bottom_layout.addWidget(self._slider_min_label)
        left_bottom_layout.addWidget(self._slider)
        left_bottom_layout.addWidget(self._slider_max_label)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self._description_label)
        left_layout.addLayout(left_bottom_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self._preview)
        right_layout.addWidget(self._preview_label)

        layout = QHBoxLayout()
        layout.addLayout(left_layout, 2)
        layout.addLayout(right_layout, 1)

        self.setLayout(layout)

        # Signals
        self._slider.valueChanged.connect(self._update_value)
        self._lineedit.textChanged.connect(self._update_value)

        self._update_line_width()
        self._update_validator()
        self._update_value(self._value)

    def _update_validator(self):
        self._validator = QIntValidator(self._min_value, self._max_value, self)
        self._lineedit.setValidator(self._validator)

    def _update_line_width(self):
        """Update width ofg line text edit."""
        txt = 'm' * (1 + len(str(self._max_value)))
        fm = self._lineedit.fontMetrics()
        if hasattr(fm, 'horizontalAdvance'):
            # Qt >= 5.11
            size = fm.horizontalAdvance(txt)
        else:
            size = fm.width(txt)

        self._lineedit.setMaximumWidth(size)
        self._lineedit.setMinimumWidth(size)

    def _update_value(self, value: int | str):
        """Update internal value and emit if changed."""
        if value == '':
            value = int(self._value)

        value = int(value)

        if value > self._max_value:
            value = self._max_value
        elif value < self._min_value:
            value = self._min_value

        if value != self._value:
            self.valueChanged.emit(value)

        self._value = value
        self._refresh(self._value)

    def _refresh(self, value: int | None = None):
        """Refresh the value on all subwidgets."""
        value = value or self._value
        self.blockSignals(True)
        self._lineedit.setText(str(value))
        self._slider.setValue(value)
        font = QFont()
        font.setPixelSize(value)
        self._preview.setFont(font)

        font = QFont()
        font.setPixelSize(self.fontMetrics().height() - 4)
        self._preview_label.setFont(font)

        self.blockSignals(False)

    def description(self) -> str:
        """Return the current widget description.

        Returns
        -------
        str
            The description text.
        """
        return self._description_label.text()

    def setDescription(self, text: str):
        """Set the current widget description.

        Parameters
        ----------
        text : str
            The description text.
        """
        self._description_label.setText(text)

    def previewText(self) -> str:
        """Return the current preview text.

        Returns
        -------
        str
            The current preview text.
        """
        return self._preview.text()

    def setPreviewText(self, text: str):
        """Set the current preview text.

        Parameters
        ----------
        text : str
            The current preview text.
        """
        self._preview.setText(text)

    def unit(self) -> str:
        """Return the current unit text.

        Returns
        -------
        str
            The current unit text.
        """
        return self._unit_label.text()

    def setUnit(self, text: str):
        """Set the current unit text.

        Parameters
        ----------
        text : str
            The current preview text.
        """
        self._unit_label.setText(text)

    def minimum(self) -> int:
        """Return the current minimum value for the slider and value in textbox.

        Returns
        -------
        int
            The minimum value for the slider.
        """
        return self._min_value

    def setMinimum(self, value: int):
        """Set the current minimum value for the slider and value in textbox.

        Parameters
        ----------
        value : int
            The minimum value for the slider.
        """
        if value >= self._max_value:
            raise ValueError(
                trans._(
                    'Minimum value must be smaller than {max_value}',
                    max_value=self._max_value,
                )
            )
        self._min_value = value
        self._value = max(self._value, self._min_value)
        self._slider_min_label.setText(str(value))
        self._slider.setMinimum(value)
        self._update_validator()
        self._refresh()

    def maximum(self) -> int:
        """Return the maximum value for the slider and value in textbox.

        Returns
        -------
        int
            The maximum value for the slider.
        """
        return self._max_value

    def setMaximum(self, value: int):
        """Set the maximum value for the slider and value in textbox.

        Parameters
        ----------
        value : int
            The maximum value for the slider.
        """
        if value <= self._min_value:
            raise ValueError(
                trans._(
                    'Maximum value must be larger than {min_value}',
                    min_value=self._min_value,
                )
            )
        self._max_value = value
        self._value = min(self._value, self._max_value)
        self._slider_max_label.setText(str(value))
        self._slider.setMaximum(value)
        self._update_validator()
        self._update_line_width()
        self._refresh()

    def value(self) -> int:
        """Return the current widget value.

        Returns
        -------
        int
            The current value.
        """
        return self._value

    def setValue(self, value: int):
        """Set the current widget value.

        Parameters
        ----------
        value : int
            The current value.
        """
        self._update_value(value)
