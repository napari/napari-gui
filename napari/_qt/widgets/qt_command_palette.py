from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from app_model.types import CommandRule, MenuItem
from qtpy import QtCore, QtGui, QtWidgets as QtW
from qtpy.QtCore import Qt, Signal

from napari._app_model import get_app_model
from napari._app_model.context._context import get_context

if TYPE_CHECKING:
    from napari._qt.qt_main_window import _QtMainWindow


class QCommandPalette(QtW.QWidget):
    """A Qt command palette widget."""

    hidden = Signal()

    def __init__(self, parent: QtW.QWidget | None = None):
        super().__init__(parent)

        self._line = QCommandLineEdit()
        self._list = QCommandList()
        _layout = QtW.QVBoxLayout(self)
        _layout.addWidget(self._line)
        _layout.addWidget(self._list)

        self._line.setPlaceholderText('Type to search commands...')
        self._line.textChanged.connect(self._on_text_changed)
        self._list.commandClicked.connect(self._on_command_clicked)
        self._line.editingFinished.connect(self.hide)
        self.hide()

        app = get_app_model()
        # this appears to be a flat list of menu items, even though the
        # type hint suggests menu or submenu
        menu_items = app.menus.get_menu(app.menus.COMMAND_PALETTE_ID)
        self.extend_command(
            [item.command for item in menu_items if isinstance(item, MenuItem)]
        )
        app.menus.menus_changed.connect(self._on_app_menus_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 400)

    def extend_command(self, list_of_commands: list[CommandRule]) -> None:
        self._list.extend_command(list_of_commands)
        return

    def _on_text_changed(self, text: str) -> None:
        self._list.update_for_text(text)
        return

    def _on_command_clicked(self, index: int) -> None:
        self._list.execute(index)
        self.hide()
        return

    def _on_app_menus_changed(self, changed_menus: set[str]) -> None:
        """Connected to app_model.menus.menus_changed."""
        app = get_app_model()
        if app.menus.COMMAND_PALETTE_ID not in changed_menus:
            return
        all_cmds_set = set(self._list.all_commands)
        palette_menu_commands = [
            item.command
            for item in app.menus.get_menu(app.menus.COMMAND_PALETTE_ID)
            if isinstance(item, MenuItem)
        ]
        palette_menu_set = set(palette_menu_commands)
        removed = all_cmds_set - palette_menu_set
        added = palette_menu_set - all_cmds_set
        for elem in removed:
            self._list.all_commands.remove(elem)
        for elem in palette_menu_commands:
            if elem in added:
                self._list.all_commands.append(elem)
        return

    def focusOutEvent(self, a0: QtGui.QFocusEvent | None) -> None:
        """Hide the palette when focus is lost."""
        self.hide()
        return super().focusOutEvent(a0)

    def update_context(self, parent: _QtMainWindow) -> None:
        """Update the context of the palette."""
        context: dict[str, Any] = {}
        context.update(get_context(parent))
        context.update(get_context(parent._qt_viewer.viewer.layers))
        self._list._app_model_context = context
        return

    def show(self) -> None:
        self._line.setText('')
        self._list.update_for_text('')
        super().show()
        if parent := self.parentWidget():
            parent_rect = parent.rect()
            self_size = self.sizeHint()
            w = min(int(parent_rect.width() * 0.8), self_size.width())
            topleft = parent.rect().topLeft()
            topleft.setX(int(topleft.x() + (parent_rect.width() - w) / 2))
            topleft.setY(int(topleft.y() + 3))
            self.move(topleft)
            self.resize(w, self_size.height())

        self.raise_()
        self._line.setFocus()
        return

    def hide(self) -> None:
        """Hide this widget."""
        self.hidden.emit()
        return super().hide()

    def text(self) -> str:
        """Return the text in the line edit."""
        return self._line.text()


class QCommandLineEdit(QtW.QLineEdit):
    """The line edit used in command palette widget."""

    def commandPalette(self) -> QCommandPalette:
        """The parent command palette widget."""
        return cast(QCommandPalette, self.parent())

    def event(self, e: QtCore.QEvent | None) -> bool:
        if e is None or e.type() != QtCore.QEvent.Type.KeyPress:
            return super().event(e)
        e = cast(QtGui.QKeyEvent, e)
        if e.modifiers() in (
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.KeypadModifier,
        ):
            key = e.key()
            if key == Qt.Key.Key_Escape:
                self.commandPalette().hide()
                return True
            if key == Qt.Key.Key_Return:
                palette = self.commandPalette()
                if palette._list.can_execute():
                    self.commandPalette().hide()
                    self.commandPalette()._list.execute()
                    return True
                return False
            if key == Qt.Key.Key_Up:
                self.commandPalette()._list.move_selection(-1)
                return True
            if key == Qt.Key.Key_PageUp:
                self.commandPalette()._list.move_selection(-10)
                return True
            if key == Qt.Key.Key_Down:
                self.commandPalette()._list.move_selection(1)
                return True
            if key == Qt.Key.Key_PageDown:
                self.commandPalette()._list.move_selection(10)
                return True
        return super().event(e)


def bold_colored(text: str, color: str) -> str:
    """Return a bolded and colored HTML text."""
    return f'<b><font color={color!r}>{text}</font></b>'


def colored(text: str, color: str) -> str:
    """Return a colored HTML text."""
    return f'<font color={color!r}>{text}</font>'


class QCommandMatchModel(QtCore.QAbstractListModel):
    """A list model for the command palette."""

    def __init__(self, parent: QtW.QWidget | None = None):
        super().__init__(parent)
        self._commands: list[CommandRule] = []
        self._max_matches = 80

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return self._max_matches

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> Any:
        """Don't show any data. Texts are rendered by the item widget."""
        return None

    def flags(self, index: QtCore.QModelIndex) -> Qt.ItemFlag:
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class QCommandLabel(QtW.QLabel):
    """The label widget to display a command in the palette."""

    DISABLED_COLOR = 'gray'

    def __init__(self, cmd: CommandRule | None = None):
        super().__init__()
        self._command: CommandRule | None = None
        self._command_text: str = ''
        if cmd is not None:
            self.set_command(cmd)

    def command(self) -> CommandRule | None:
        """The app-model Action bound to this label."""
        return self._command

    def set_command(self, cmd: CommandRule) -> None:
        """Set command to this widget."""
        command_text = _format_action_name(cmd)
        self._command_text = command_text
        self._command = cmd
        self.setText(command_text)
        self.setToolTip(cmd.tooltip)

    def command_text(self) -> str:
        """The original command text."""
        return self._command_text

    def set_text_colors(self, input_text: str, color: str) -> None:
        """Set label text color based on the input text."""
        if input_text == '':
            return
        text = self.command_text()
        words = input_text.split(' ')
        pattern = re.compile('|'.join(words), re.IGNORECASE)

        output_texts: list[str] = []
        last_end = 0
        for match_obj in pattern.finditer(text):
            output_texts.append(text[last_end : match_obj.start()])
            word = match_obj.group()
            colored_word = bold_colored(word, color)
            output_texts.append(colored_word)
            last_end = match_obj.end()

        output_texts.append(text[last_end:])
        output_text = ''.join(output_texts)
        self.setText(output_text)
        return

    def set_disabled(self) -> None:
        """Set the label to disabled."""
        text = self.command_text()
        self.setText(colored(text, self.DISABLED_COLOR))
        return


class QCommandList(QtW.QListView):
    commandClicked = Signal(int)  # one of the items is clicked

    def __init__(self, parent: QtW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setModel(QCommandMatchModel(self))
        self.setSelectionMode(QtW.QAbstractItemView.SelectionMode.NoSelection)
        self._selected_index = 0

        # NOTE: maybe useful for fetch-and-scrolling in the future
        self._index_offset = 0

        self._label_widgets: list[QCommandLabel] = []
        self._current_max_index = 0
        for i in range(self.model()._max_matches):
            lw = QCommandLabel()
            self._label_widgets.append(lw)
            self.setIndexWidget(self.model().index(i), lw)
        self.pressed.connect(self._on_clicked)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._match_color = '#468cc6'
        self._app_model_context: dict[str, Any] = {}

    def _on_clicked(self, index: QtCore.QModelIndex) -> None:
        if index.isValid():
            self.commandClicked.emit(index.row())
            return

    def move_selection(self, dx: int) -> None:
        """Move selection by dx, dx can be negative or positive."""
        self._selected_index += dx
        self._selected_index = max(0, self._selected_index)
        self._selected_index = min(
            self._current_max_index - 1, self._selected_index
        )
        self.update_selection()
        return

    def update_selection(self) -> None:
        """Update the widget selection state based on the selected index."""
        index = self.model().index(self._selected_index - self._index_offset)
        if model := self.selectionModel():
            model.setCurrentIndex(
                index, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
        return

    @property
    def all_commands(self) -> list[CommandRule]:
        return self.model()._commands

    def extend_command(self, commands: list[CommandRule]) -> None:
        """Extend the list of commands."""
        self.all_commands.extend(commands)
        return

    def command_at(self, index: int) -> CommandRule | None:
        i = index - self._index_offset
        index_widget = self.indexWidget(self.model().index(i))
        if index_widget is None:
            return None
        return index_widget.command()

    def iter_command(self) -> Iterator[CommandRule]:
        """Iterate over all the commands registered to this command list widget."""
        for i in range(self.model().rowCount()):
            if not self.isRowHidden(i):
                command = self.command_at(i)
                if command is not None:
                    yield command

    def execute(self, index: int | None = None) -> None:
        """Execute the currently selected command."""
        if index is None:
            index = self._selected_index
        command = self.command_at(index)
        if command is None:
            return
        _exec_action(command)
        # move to the top
        self.all_commands.remove(command)
        self.all_commands.insert(0, command)
        return

    def can_execute(self) -> bool:
        """Return true if the command can be executed."""
        index = self._selected_index
        command = self.command_at(index)
        if command is None:
            return False
        return _enabled(command, self._app_model_context)

    def update_for_text(self, input_text: str) -> None:
        """Update the list to match the input text."""
        self._selected_index = 0
        max_matches = self.model()._max_matches
        row = 0
        for row, action in enumerate(self.iter_top_hits(input_text)):
            self.setRowHidden(row, False)
            lw = self.indexWidget(self.model().index(row))
            if lw is None:
                self._current_max_index = row
                break
            lw.set_command(action)
            if _enabled(action, self._app_model_context):
                lw.set_text_colors(input_text, color=self._match_color)
            else:
                lw.set_disabled()

            if row >= max_matches:
                self._current_max_index = max_matches
                break
            row = row + 1
        else:
            # if the loop completes without break
            self._current_max_index = row
            for r in range(row, max_matches):
                self.setRowHidden(r, True)
        self.update_selection()
        return

    def iter_top_hits(self, input_text: str) -> Iterator[CommandRule]:
        """Iterate over the top hits for the input text"""
        commands: list[tuple[float, CommandRule]] = []
        for command in self.all_commands:
            score = _match_score(command, input_text)
            if score > 0.0:
                if _enabled(command, self._app_model_context):
                    score += 10.0
                commands.append((score, command))
        commands.sort(key=lambda x: x[0], reverse=True)
        for _, command in commands:
            yield command

    if TYPE_CHECKING:

        def model(self) -> QCommandMatchModel: ...
        def indexWidget(
            self, index: QtCore.QModelIndex
        ) -> QCommandLabel | None: ...


def _enabled(action: CommandRule, context: Mapping[str, Any]) -> bool:
    if action.enablement is None:
        return True
    try:
        return action.enablement.eval(context)
    except NameError:
        return False


def _match_score(action: CommandRule, input_text: str) -> float:
    """Return a match score (between 0 and 1) for the input text."""
    name = _format_action_name(action).lower()
    if all(word in name for word in input_text.lower().split(' ')):
        return 1.0
    return 0.0


def _format_action_name(cmd: CommandRule) -> str:
    sep = ':' if ':' in cmd.id else '.'
    *contexts, _ = cmd.id.split(sep)
    title = ' > '.join(contexts)
    desc = cmd.title
    if title:
        return f'{title} > {desc}'
    return desc


def _exec_action(action: CommandRule) -> Any:
    app = get_app_model()
    return app.commands.execute_command(action.id).result()
