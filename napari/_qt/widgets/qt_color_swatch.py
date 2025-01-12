import re
from typing import Union

import numpy as np
from qtpy.QtCore import QEvent, Qt, Signal, Slot
from qtpy.QtGui import QColor, QKeyEvent, QMouseEvent
from qtpy.QtWidgets import (
    QColorDialog,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
from vispy.color import get_color_dict

from napari._qt.dialogs.qt_modal import QtPopup
from napari.utils.colormaps.colormap_utils import ColorType
from napari.utils.colormaps.standardize_color import (
    hex_to_name,
    rgb_to_hex,
    transform_color,
)
from napari.utils.translations import trans

# matches any 3- or 4-tuple of int or float, with or without parens
# captures the numbers into groups.
# this is used to allow users to enter colors as e.g.: "(1, 0.7, 0)"
rgba_regex = re.compile(
    r'\(?([\d.]+),\s*([\d.]+),\s*([\d.]+),?\s*([\d.]+)?\)?'
)

TRANSPARENT = np.array([0, 0, 0, 0], np.float32)
AnyColorType = Union[ColorType, QColor]


class QColorSwatchEdit(QWidget):
    """A widget that combines a QColorSwatch with a QColorLineEdit.

    emits a color_changed event with a 1x4 numpy array when the current color
    changes.  Note, the "model" for the current color is the ``_color``
    attribute on the QColorSwatch.

    Parameters
    ----------
    parent : QWidget, optional
        parent widget, by default None
    initial_color : AnyColorType, optional
        Starting color, by default None
    tooltip : str, optional
        Tooltip when hovering on the swatch,
        by default 'click to set color'

    Attributes
    ----------
    line_edit : QColorLineEdit
        An instance of QColorLineEdit, which takes hex, rgb, or autocompletes
        common color names.  On invalid input, this field will return to the
        previous color value.
    color_swatch : QColorSwatch
        The square that shows the current color, and can be clicked to show a
        color dialog.
    color : np.ndarray
        The current color (just an alias for the colorSwatch.color)

    Signals
    -------
    color_changed : np.ndarray
        Emits the new color when the current color changes.
    """

    color_changed = Signal(np.ndarray)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_color: AnyColorType | None = None,
        tooltip: str | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setObjectName('QColorSwatchEdit')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setLayout(layout)

        self.line_edit = QColorLineEdit(self)
        self.line_edit.editingFinished.connect(self._on_line_edit_edited)

        self.color_swatch = QColorSwatch(self, tooltip=tooltip)
        self.color_swatch.color_changed.connect(self._on_swatch_changed)
        self.setColor = self.color_swatch.setColor
        if initial_color is not None:
            self.setColor(initial_color)

        layout.addWidget(self.color_swatch)
        layout.addWidget(self.line_edit)

    @property
    def color(self):
        """Return the current color."""
        return self.color_swatch.color

    def _on_line_edit_edited(self):
        """When the user hits enter or loses focus on the LineEdit widget."""
        text = self.line_edit.text()
        rgb_match = rgba_regex.match(text)
        if rgb_match:
            text = [float(x) for x in rgb_match.groups() if x]
        self.color_swatch.setColor(text)

    @Slot(np.ndarray)
    def _on_swatch_changed(self, color: np.ndarray):
        """Receive QColorSwatch change event, update the lineEdit, re-emit."""
        self.line_edit.setText(color)
        self.color_changed.emit(color)


class QColorSwatch(QFrame):
    """A QFrame that displays a color and can be clicked to show a QColorPopup.

    Parameters
    ----------
    parent : QWidget, optional
        parent widget, by default None
    tooltip : Optional[str], optional
        Tooltip when hovering on swatch,
        by default 'click to set color'
    initial_color : ColorType, optional
        initial color, by default will be transparent

    Attributes
    ----------
    color : np.ndarray
        The current color

    Signals
    -------
    color_changed : np.ndarray
        Emits the new color when the current color changes.
    """

    color_changed = Signal(np.ndarray)

    def __init__(
        self,
        parent: QWidget | None = None,
        tooltip: str | None = None,
        initial_color: ColorType | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('colorSwatch')
        self.setToolTip(tooltip or trans._('click to set color'))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.color_changed.connect(self._update_swatch_style)
        self._color: np.ndarray = TRANSPARENT
        if initial_color is not None:
            self.setColor(initial_color)

    @property
    def color(self):
        """Return the current color"""
        return self._color

    @Slot(np.ndarray)
    def _update_swatch_style(self, color: np.ndarray) -> None:
        """Convert the current color to rgba() string and update appearance."""
        rgba = f'rgba({",".join(str(int(x * 255)) for x in self._color)})'
        self.setStyleSheet('#colorSwatch {background-color: ' + rgba + ';}')

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Show QColorPopup picker when the user clicks on the swatch."""
        if event.button() == Qt.MouseButton.LeftButton:
            initial = QColor(*(255 * self._color).astype('int'))
            popup = QColorPopup(self, initial)
            popup.colorSelected.connect(self.setColor)
            popup.show_right_of_mouse()

    def setColor(self, color: AnyColorType) -> None:
        """Set the color of the swatch.

        Parameters
        ----------
        color : ColorType
            Can be any ColorType recognized by our
            utils.colormaps.standardize_color.transform_color function.
        """
        if isinstance(color, QColor):
            _color = (np.array(color.getRgb()) / 255).astype(np.float32)
        else:
            try:
                _color = transform_color(color)[0]
            except ValueError:
                return self.color_changed.emit(self._color)
        emit = np.any(self._color != _color)
        self._color = _color
        if emit or np.array_equiv(_color, TRANSPARENT):
            self.color_changed.emit(_color)
            return None
        return None


class QColorLineEdit(QLineEdit):
    """A LineEdit that takes hex, rgb, or autocompletes common color names.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget, by default None
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._compl = QCompleter([*get_color_dict(), 'transparent'])
        self._compl.setCompletionMode(QCompleter.InlineCompletion)
        self.setCompleter(self._compl)
        self.setTextMargins(2, 2, 2, 2)

    def setText(self, color: ColorType):
        """Set the text of the lineEdit using any ColorType.

        Colors will be converted to standard SVG spec names if possible,
        or shown as #RGBA hex if not.

        Parameters
        ----------
        color : ColorType
            Can be any ColorType recognized by our
            utils.colormaps.standardize_color.transform_color function.
        """
        _rgb = transform_color(color)[0]
        _hex = rgb_to_hex(_rgb)[0]
        super().setText(hex_to_name.get(_hex, _hex))


class CustomColorDialog(QColorDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setObjectName('CustomColorDialog')

    def keyPressEvent(self, event: QEvent):
        event.ignore()


class QColorPopup(QtPopup):
    """A QColorDialog inside of our QtPopup.

    Allows all of the show methods of QtPopup (like show relative to mouse).
    Passes through signals from the ColorDialogm, and handles some keypress
    events.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget. by default None
    initial_color : AnyColorType, optional
        The initial color set in the color dialog, by default None

    Attributes
    ----------
    color_dialog : CustomColorDialog
        The main color dialog in the popup
    """

    currentColorChanged = Signal(QColor)
    colorSelected = Signal(QColor)

    def __init__(
        self, parent: QWidget = None, initial_color: AnyColorType = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName('QtColorPopup')
        self.color_dialog = CustomColorDialog(self)

        # native dialog doesn't get added to the QtPopup frame
        # so more would need to be done to use it
        self.color_dialog.setOptions(
            QColorDialog.DontUseNativeDialog | QColorDialog.ShowAlphaChannel
        )
        layout = QVBoxLayout()
        self.frame.setLayout(layout)
        layout.addWidget(self.color_dialog)

        self.color_dialog.currentColorChanged.connect(
            self.currentColorChanged.emit
        )
        self.color_dialog.colorSelected.connect(self._on_color_selected)
        self.color_dialog.rejected.connect(self._on_rejected)
        self.color_dialog.setCurrentColor(QColor(initial_color))

    def _on_color_selected(self, color: QColor):
        """When a color has beeen selected and the OK button clicked."""
        self.colorSelected.emit(color)
        self.close()

    def _on_rejected(self):
        self.close()

    def keyPressEvent(self, event: QKeyEvent):
        """Accept current color on enter, cancel on escape.

        Parameters
        ----------
        event : QKeyEvent
            The keypress event that triggered this method.
        """
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self.color_dialog.accept()
        if event.key() == Qt.Key.Key_Escape:
            return self.color_dialog.reject()
        self.color_dialog.keyPressEvent(event)
        return None
