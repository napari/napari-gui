from qtpy.QtWidgets import QRadioButton, QPushButton


class QtModeRadioButton(QRadioButton):
    def __init__(
        self, layer, button_name, mode, *, tool_tip=None, checked=False
    ):
        """Creates a radio button that can enable a specific layer mode.

        Parameters
        ----------
        layer : Layer
            The layer instance that this button controls.
        button_name : str
            Name for the button.  This is mostly used to identify the button
            in stylesheets (e.g. to add a custom icon)
        mode : Enum, optional
            The mode to enable when this button is clicked.
        tool_tip : str, optional
            A tooltip to display when hovering the mouse on this button.
        checked : bool, optional
            Whether the button is activate, by default False.
            One button in a QButtonGroup should be initially checked.
        """
        super().__init__()

        self.layer = layer
        self.setToolTip(tool_tip or button_name)
        self.setChecked(checked)
        self.setProperty('mode', button_name)
        self.setFixedWidth(28)
        self.mode = mode
        if mode is not None:
            self.toggled.connect(self._set_mode)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker(self._set_mode):
            if bool:
                self.layer.mode = self.mode


class QtModePushButton(QPushButton):
    def __init__(self, layer, button_name, slot=None, tool_tip=None):
        """Creates a radio button that can trigger a specific action.

        Parameters
        ----------
        layer : Layer
            The layer instance that this button controls.
        button_name : str
            Name for the button.  This is mostly used to identify the button
            in stylesheets (e.g. to add a custom icon)
        slot : callable, optional
            The function to call when this button is clicked.
        tool_tip : str, optional
            A tooltip to display when hovering the mouse on this button.
        """
        super().__init__()

        self.layer = layer
        self.setProperty('mode', button_name)
        self.setToolTip(tool_tip or button_name)
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        if slot is not None:
            self.clicked.connect(slot)
