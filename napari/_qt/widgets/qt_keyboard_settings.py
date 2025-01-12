import contextlib
import itertools
import sys
from collections import OrderedDict

from app_model.backends.qt import (
    qkeysequence2modelkeybinding,
)
from qtpy.QtCore import QEvent, QPoint, Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QItemDelegate,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari._qt.widgets.qt_message_popup import WarnPopup
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans


class ShortcutEditor(QWidget):
    """Widget to edit keybindings for napari."""

    valueChanged = Signal(dict)
    VIEWER_KEYBINDINGS = trans._('Viewer key bindings')

    def __init__(
        self,
        parent: QWidget = None,
        description: str = '',
        value: dict | None = None,
    ) -> None:
        super().__init__(parent=parent)

        # Flag to not run _set_keybinding method after setting special symbols.
        # When changing line edit to special symbols, the _set_keybinding
        # method will be called again (and breaks) and is not needed.
        self._skip = False

        layers = [
            Image,
            Labels,
            Points,
            Shapes,
            Surface,
            Tracks,
            Vectors,
        ]

        self.key_bindings_strs = OrderedDict()

        # widgets
        self.layer_combo_box = QComboBox(self)
        self._label = QLabel(self)
        self._table = QTableWidget(self)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setShowGrid(False)
        self._restore_button = QPushButton(trans._('Restore All Keybindings'))

        # Set up dictionary for layers and associated actions.
        all_actions = action_manager._actions.copy()
        self.key_bindings_strs[self.VIEWER_KEYBINDINGS] = {}

        for layer in layers:
            if len(layer.class_keymap) == 0:
                actions = {}
            else:
                actions = action_manager._get_provider_actions(layer)
                for name in actions:
                    all_actions.pop(name)
            self.key_bindings_strs[f'{layer.__name__} layer'] = actions

        # Don't include actions without keymapproviders
        for action_name in all_actions.copy():
            if all_actions[action_name].keymapprovider is None:
                all_actions.pop(action_name)
        # Left over actions can go here.
        self.key_bindings_strs[self.VIEWER_KEYBINDINGS] = all_actions

        # Widget set up
        self.layer_combo_box.addItems(list(self.key_bindings_strs))
        self.layer_combo_box.currentTextChanged.connect(self._set_table)
        self.layer_combo_box.setCurrentText(self.VIEWER_KEYBINDINGS)
        self._set_table()
        self._label.setText(trans._('Group'))
        self._restore_button.clicked.connect(self.restore_defaults)

        # layout
        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self._label)
        hlayout1.addWidget(self.layer_combo_box)
        hlayout1.setContentsMargins(0, 0, 0, 0)
        hlayout1.setSpacing(20)
        hlayout1.addStretch(0)

        hlayout2 = QHBoxLayout()
        hlayout2.addLayout(hlayout1)
        hlayout2.addWidget(self._restore_button)

        layout = QVBoxLayout()
        layout.addLayout(hlayout2)
        layout.addWidget(self._table)
        layout.addWidget(
            QLabel(
                trans._(
                    'To edit, double-click the keybinding. To unbind a shortcut, use Backspace or Delete. To set Backspace or Delete, first unbind.'
                )
            )
        )

        self.setLayout(layout)

    def restore_defaults(self):
        """Launches dialog to confirm restore choice."""
        prev = QApplication.instance().testAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs
        )
        QApplication.instance().setAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True
        )
        response = QMessageBox.question(
            self,
            trans._('Restore Shortcuts'),
            trans._('Are you sure you want to restore default shortcuts?'),
            QMessageBox.StandardButton.RestoreDefaults
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.RestoreDefaults,
        )
        QApplication.instance().setAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs, prev
        )

        if response == QMessageBox.RestoreDefaults:
            self._reset_shortcuts()

    def _reset_shortcuts(self):
        """Reset shortcuts to default settings."""

        get_settings().shortcuts.reset()
        for (
            action,
            shortcuts,
        ) in get_settings().shortcuts.shortcuts.items():
            action_manager.unbind_shortcut(action)
            for shortcut in shortcuts:
                action_manager.bind_shortcut(action, shortcut)

        self._set_table(layer_str=self.layer_combo_box.currentText())

    def _set_table(self, layer_str: str = ''):
        """Builds and populates keybindings table.

        Parameters
        ----------
        layer_str : str
            If layer_str is not empty, then show the specified layers'
            keybinding shortcut table.
        """

        # Keep track of what is in each column.
        self._action_name_col = 0
        self._icon_col = 1
        self._shortcut_col = 2
        self._shortcut_col2 = 3
        self._action_col = 4

        # Set header strings for table.
        header_strs = ['', '', '', '', '']
        header_strs[self._action_name_col] = trans._('Action')
        header_strs[self._shortcut_col] = trans._('Keybinding')
        header_strs[self._shortcut_col2] = trans._('Alternative Keybinding')

        # If no layer_str, then set the page to the viewer keybindings page.
        if not layer_str:
            layer_str = self.VIEWER_KEYBINDINGS

        # If rebuilding the table, then need to disconnect the connection made
        # previously as well as clear the table contents.
        with contextlib.suppress(TypeError, RuntimeError):
            self._table.cellChanged.disconnect(self._set_keybinding)
        self._table.clearContents()

        # Table styling set up.
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setStyleSheet(
            'border-bottom: 2px solid white;'
        )

        # Get all actions for the layer.
        actions = self.key_bindings_strs[layer_str]

        if len(actions) > 0:
            # Set up table based on number of actions and needed columns.
            self._table.setRowCount(len(actions))
            self._table.setColumnCount(5)
            # Set up delegate in order to capture keybindings.
            self._table.setItemDelegateForColumn(
                self._shortcut_col, ShortcutDelegate(self._table)
            )
            self._table.setItemDelegateForColumn(
                self._shortcut_col2, ShortcutDelegate(self._table)
            )
            self._table.setHorizontalHeaderLabels(header_strs)
            self._table.horizontalHeader().setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft
            )
            self._table.verticalHeader().setVisible(False)

            # Hide the column with action names.  These are kept here for reference when needed.
            self._table.setColumnHidden(self._action_col, True)

            # Column set up.
            self._table.setColumnWidth(self._action_name_col, 370)
            self._table.setColumnWidth(self._shortcut_col, 190)
            self._table.setColumnWidth(self._shortcut_col2, 145)
            self._table.setColumnWidth(self._icon_col, 35)
            self._table.setWordWrap(True)

            # Add some padding to rows
            self._table.setStyleSheet('QTableView::item { padding: 6px; }')

            # Go through all the actions in the layer and add them to the table.
            for row, (action_name, action) in enumerate(actions.items()):
                shortcuts = action_manager._shortcuts.get(action_name, [])
                # Set action description.  Make sure its not selectable/editable.
                item = QTableWidgetItem(action.description)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._table.setItem(row, self._action_name_col, item)
                # Ensure long descriptions can be wrapped in cells
                self._table.resizeRowToContents(row)

                # Create empty item in order to make sure this column is not
                # selectable/editable.
                item = QTableWidgetItem('')
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                self._table.setItem(row, self._icon_col, item)

                # Set the shortcuts in table.
                item_shortcut = QTableWidgetItem(
                    Shortcut(next(iter(shortcuts))).platform
                    if shortcuts
                    else ''
                )
                self._table.setItem(row, self._shortcut_col, item_shortcut)

                item_shortcut2 = QTableWidgetItem(
                    Shortcut(list(shortcuts)[1]).platform
                    if len(shortcuts) > 1
                    else ''
                )
                self._table.setItem(row, self._shortcut_col2, item_shortcut2)

                # action_name is stored in the table to use later, but is not shown on dialog.
                item_action = QTableWidgetItem(action_name)
                self._table.setItem(row, self._action_col, item_action)

            # If a cell is changed, run .set_keybinding.
            self._table.cellChanged.connect(self._set_keybinding)
        else:
            # Display that there are no actions for this layer.
            self._table.setRowCount(1)
            self._table.setColumnCount(4)
            self._table.setHorizontalHeaderLabels(header_strs)
            self._table.verticalHeader().setVisible(False)

            self._table.setColumnHidden(self._action_col, True)
            item = QTableWidgetItem(trans._('No key bindings'))
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._table.setItem(0, 0, item)

    def _get_potential_conflicting_actions(self):
        """
        Get all actions we want to avoid keybinding conflicts with.

        If current selected keybinding group is a layer, return
        the selected layer actions and viewer actions.
        If the current selected keybinding group is viewer,
        return viewer actions and actions from all layers.

        Returns
        -------
        actions_all: list[tuple[str, tuple[str, Action]]]
            Tuple of group names and actions, to avoid keybinding
            conflicts with.
            Format:
                [
                    ('keybinding_group', ('action_name', Action)),
                    ...
                ]
        """
        current_layer_text = self.layer_combo_box.currentText()
        actions_all = list(self._get_group_actions(current_layer_text))

        if current_layer_text != self.VIEWER_KEYBINDINGS:
            actions_all = list(self._get_group_actions(current_layer_text))
            actions_all.extend(
                self._get_group_actions(self.VIEWER_KEYBINDINGS)
            )
        else:
            actions_all = []
            for group in self.key_bindings_strs:
                actions_all.extend(self._get_group_actions(group))
        return actions_all

    def _get_group_actions(self, group_name):
        group_actions = self.key_bindings_strs[group_name]
        return zip(itertools.repeat(group_name), group_actions.items())

    def _restore_shortcuts(self, row):
        action_name = self._table.item(row, self._action_col).text()
        shortcuts = action_manager._shortcuts.get(action_name, [])
        with lock_keybind_update(self):
            self._table.item(row, self._shortcut_col).setText(
                Shortcut(next(iter(shortcuts))).platform if shortcuts else ''
            )
            self._table.item(row, self._shortcut_col2).setText(
                Shortcut(list(shortcuts)[1]).platform
                if len(shortcuts) > 1
                else ''
            )

    def _show_conflicts_warning(
        self, new_shortcut, conflicting_actions, conflicting_rows
    ):
        # create string listing info of all the conflicts found
        conflicting_actions_string = '<ul>'
        for group, action_description in conflicting_actions:
            conflicting_actions_string += trans._(
                '<li><b>{action_description}</b> in the <b>{group}</b> group</li>',
                action_description=action_description,
                group=group,
            )
        conflicting_actions_string += '</ul>'

        # show warning symbols
        self._show_warning_icons(conflicting_rows)

        # show warning message
        message = trans._(
            'The keybinding <b>{new_shortcut}</b> is already assigned to:'
            '{conflicting_actions_string}'
            'Change or clear conflicting shortcuts before assigning <b>{new_shortcut}</b> to this one.',
            new_shortcut=new_shortcut,
            conflicting_actions_string=conflicting_actions_string,
        )
        self._show_warning(conflicting_rows[0], message)

        self._restore_shortcuts(conflicting_rows[0])

        self._cleanup_warning_icons(conflicting_rows)

    def _mark_conflicts(self, new_shortcut, row) -> bool:
        # Go through all layer actions to determine if the new shortcut is already here.
        current_action = self._table.item(row, self._action_col).text()
        actions_all = self._get_potential_conflicting_actions()
        current_item = self._table.currentItem()
        conflicting_rows = [row]
        conflicting_actions = []
        for conflicting_row, (group, (action_name, action)) in enumerate(
            actions_all
        ):
            shortcuts = action_manager._shortcuts.get(action_name, [])

            if Shortcut(new_shortcut).qt not in [
                Shortcut(shortcut).qt for shortcut in shortcuts
            ]:
                continue

            # Shortcut is here (either same action or not), don't replace in settings.
            if action_name != current_action:
                # the shortcut is saved to a different action, save conflicting shortcut info
                if conflicting_row < self._table.rowCount():
                    # only save row number for conflicts that are inside the current table
                    conflicting_rows.append(conflicting_row)
                conflicting_actions.append((group, action.description))

            # This shortcut was here.  Reformat and reset text.
            format_shortcut = Shortcut(new_shortcut).platform
            with lock_keybind_update(self):
                current_item.setText(format_shortcut)

        if len(conflicting_actions) > 0:
            # show conflicts message and mark conflicting rows as necessary
            self._show_conflicts_warning(
                new_shortcut, conflicting_actions, conflicting_rows
            )

            return False

        return True

    def _show_bind_shortcut_error(
        self, current_action, current_shortcuts, row, new_shortcut
    ):
        action_manager._shortcuts[current_action] = []
        # need to rebind the old shortcut
        action_manager.unbind_shortcut(current_action)
        for short in current_shortcuts:
            action_manager.bind_shortcut(current_action, short)

        # Show warning message to let user know this shortcut is invalid.
        self._show_warning_icons([row])

        message = trans._(
            '<b>{new_shortcut}</b> is not a valid keybinding.',
            new_shortcut=Shortcut(new_shortcut).platform,
        )
        self._show_warning(row, message)

        self._cleanup_warning_icons([row])
        self._restore_shortcuts(row)

    def _set_keybinding(self, row, col):
        """Checks the new keybinding to determine if it can be set.

        Parameters
        ----------
        row : int
            Row in keybindings table that is being edited.
        col : int
            Column being edited (shortcut column).
        """

        if self._skip:
            return

        self._table.setCurrentItem(self._table.item(row, col))

        if col in {self._shortcut_col, self._shortcut_col2}:
            # get the current item from shortcuts column
            current_item = self._table.currentItem()
            new_shortcut = Shortcut.parse_platform(current_item.text())
            if new_shortcut:
                new_shortcut = new_shortcut[0].upper() + new_shortcut[1:]

            # get the current action name
            current_action = self._table.item(row, self._action_col).text()

            # get the original shortcuts
            current_shortcuts = list(
                action_manager._shortcuts.get(current_action, [])
            )
            for mod in {'Shift', 'Ctrl', 'Alt', 'Cmd', 'Super', 'Meta'}:
                # we want to prevent multiple modifiers but still allow single modifiers.
                if new_shortcut.endswith('-' + mod):
                    self._show_bind_shortcut_error(
                        current_action,
                        current_shortcuts,
                        row,
                        new_shortcut,
                    )
                    return

            # Flag to indicate whether to set the new shortcut.
            replace = self._mark_conflicts(new_shortcut, row)

            if replace is True:
                # This shortcut is not taken.

                #  Unbind current action from shortcuts in action manager.
                action_manager.unbind_shortcut(current_action)
                shortcuts_list = list(current_shortcuts)
                ind = col - self._shortcut_col
                if new_shortcut != '':
                    if ind < len(shortcuts_list):
                        shortcuts_list[ind] = new_shortcut
                    else:
                        shortcuts_list.append(new_shortcut)
                elif ind < len(shortcuts_list):
                    shortcuts_list.pop(col - self._shortcut_col)
                new_value_dict = {}
                # Bind the new shortcut.
                try:
                    for short in shortcuts_list:
                        action_manager.bind_shortcut(current_action, short)
                except TypeError:
                    self._show_bind_shortcut_error(
                        current_action,
                        current_shortcuts,
                        row,
                        new_shortcut,
                    )
                    return

                # The new shortcut is valid and can be displayed in widget.

                # Keep track of what changed.
                new_value_dict = {current_action: shortcuts_list}

                self._restore_shortcuts(row)

                # Emit signal when new value set for shortcut.
                self.valueChanged.emit(new_value_dict)

    def _show_warning_icons(self, rows):
        """Creates and displays the warning icons.

        Parameters
        ----------
        rows : list[int]
            List of row numbers that should have the icon.
        """

        for row in rows:
            self.warning_indicator = QLabel(self)
            self.warning_indicator.setObjectName('error_label')

            self._table.setCellWidget(
                row, self._icon_col, self.warning_indicator
            )

    def _cleanup_warning_icons(self, rows):
        """Remove the warning icons from the shortcut table.

        Parameters
        ----------
        rows : list[int]
            List of row numbers to remove warning icon from.

        """
        for row in rows:
            self._table.setCellWidget(row, self._icon_col, QLabel(''))

    def _show_warning(self, row: int, message: str) -> None:
        """Creates and displays warning message when shortcut is already assigned.

        Parameters
        ----------
        row : int
            Row in table where the shortcut is attempting to be set.
        message : str
            Message to be displayed in warning pop up.
        """

        # Determine placement of warning message.
        delta_y = 105
        delta_x = 10
        global_point = self.mapToGlobal(
            QPoint(
                self._table.columnViewportPosition(self._shortcut_col)
                + delta_x,
                self._table.rowViewportPosition(row) + delta_y,
            )
        )

        # Create warning pop up and move it to desired position.
        self._warn_dialog = WarnPopup(
            text=message,
        )
        self._warn_dialog.move(global_point)
        self._warn_dialog.exec_()

    def value(self):
        """Return the actions and shortcuts currently assigned in action manager.

        Returns
        -------
        value: dict
            Dictionary of action names and shortcuts assigned to them.
        """

        value = {}

        for action_name in action_manager._actions:
            shortcuts = action_manager._shortcuts.get(action_name, [])
            value[action_name] = list(shortcuts)

        return value

    def setValue(self, state):
        for action, shortcuts in state.items():
            action_manager.unbind_shortcut(action)
            for shortcut in shortcuts:
                action_manager.bind_shortcut(action, shortcut)

        self._set_table(layer_str=self.layer_combo_box.currentText())


class ShortcutDelegate(QItemDelegate):
    """Delegate that handles when user types in new shortcut."""

    def createEditor(self, widget, style_option, model_index):
        self._editor = EditorWidget(widget)
        return self._editor

    def setEditorData(self, widget, model_index):
        text = model_index.model().data(model_index, Qt.ItemDataRole.EditRole)
        widget.setText(str(text) if text else '')

    def updateEditorGeometry(self, widget, style_option, model_index):
        widget.setGeometry(style_option.rect)

    def setModelData(self, widget, abstract_item_model, model_index):
        text = widget.text()
        abstract_item_model.setData(
            model_index, text, Qt.ItemDataRole.EditRole
        )


class EditorWidget(QLineEdit):
    """Editor widget set in the delegate column in shortcut table."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def event(self, event):
        """Qt method override."""
        if event.type() == QEvent.Type.ShortcutOverride:
            self.keyPressEvent(event)
            return True
        if event.type() == QEvent.Type.Shortcut:
            return True
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        ):
            # If there is a shortcut set already, two events are being emitted when
            # pressing `Delete` or `Backspace`. First a `ShortcutOverride` event and
            # then a `KeyPress` event. We need to mark the second event (`KeyPress`)
            # as handled for those keys to not end up processing it multiple times.
            # Without that, pressing `Backspace` or `Delete` will set those keys as shortcuts
            # instead of just cleaning/removing the previous shortcut that was set.
            return True

        return super().event(event)

    def _handleEditModifiersOnly(self, event) -> None:
        """
        Shared handler between keyPressEvent and keyReleaseEvent for modifiers.

        This is valid both on keyPress and Keyrelease events during edition as we
        are sure to not have received any real keys events yet. If that was the case
        the shortcut would have been validated and we would not be in edition mode.
        """
        event_key = event.key()
        if event_key not in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
        ):
            return
        if sys.platform == 'darwin':
            # On macOS, the modifiers are not the same as on other platforms.
            # we also use pairs instead of a dict to keep the order.
            modmap = (
                (Qt.ControlModifier, 'Meta+'),
                (Qt.AltModifier, 'Alt+'),
                (Qt.ShiftModifier, 'Shift+'),
                (Qt.MetaModifier, 'Ctrl+'),
            )
        else:
            modmap = (
                (Qt.ControlModifier, 'Ctrl+'),
                (Qt.AltModifier, 'Alt+'),
                (Qt.ShiftModifier, 'Shift+'),
                (Qt.MetaModifier, 'Meta+'),
            )
        modifiers = event.modifiers()
        seq = ''
        for mod, s in modmap:
            if modifiers & mod:
                seq += s
        seq = seq[:-1]

        # in current pyappkit this will have weird effects on the order of modifiers
        # see https://github.com/pyapp-kit/app-model/issues/110
        self.setText(Shortcut(seq).platform)

    def keyReleaseEvent(self, event) -> None:
        self._handleEditModifiersOnly(event)

    def keyPressEvent(self, event) -> None:
        """Qt method override."""
        event_key = event.key()

        if not event_key or event_key == Qt.Key.Key_unknown:
            return

        if (
            event_key in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}
            and self.text() != ''
            and self.hasSelectedText()
        ):
            # Allow user to delete shortcut.
            self.setText('')
            return

        if event_key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
        ):
            self._handleEditModifiersOnly(event)
            return

        if event_key in {
            Qt.Key.Key_Return,
            Qt.Key.Key_Tab,
            Qt.Key.Key_Enter,
        }:
            # Do not allow user to set these keys as shortcut.
            # Use them as a save trigger for modifier only shortcuts.
            return

        # Translate key value to key string.
        translator = ShortcutTranslator()
        event_keyseq = translator.keyevent_to_keyseq(event)
        kb = qkeysequence2modelkeybinding(event_keyseq)
        short = Shortcut(kb)
        self.setText(short.platform)
        self.clearFocus()


class ShortcutTranslator(QKeySequenceEdit):
    """
    Convert QKeyEvent into QKeySequence.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hide()

    def keyevent_to_keyseq(self, event) -> QKeySequence:
        """Return a QKeySequence representation of the provided QKeyEvent.

        This only works for complete key sequence that do not contain only
        modifiers. If the event is only pressing modifiers, this will return an
        empty sequence as QKeySequence does not support only modifiers

        """

        self.keyPressEvent(event)
        event.accept()
        return self.keySequence()

    def keyReleaseEvent(self, event):
        """Qt Override"""
        return False

    def timerEvent(self, event):
        """Qt Override"""
        return False

    def event(self, event):
        """Qt Override"""
        return False


@contextlib.contextmanager
def lock_keybind_update(widget: ShortcutEditor):
    prev = widget._skip
    widget._skip = True
    try:
        yield
    finally:
        widget._skip = prev
