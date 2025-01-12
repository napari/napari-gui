import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari.utils.misc import in_ipython
from napari.utils.translations import trans

HOME_DIRECTORY = str(Path.home())


class ScreenshotDialog(QFileDialog):
    """
    Dialog to chose save location of screenshot.

    Parameters
    ----------
    save_function : Callable[[str], Any],
        Function to be called on success of selecting save location
     parent : QWidget, optional
        Optional parent widget for this widget..
    directory : str, optional
        Starting directory to be set to File Dialog

    """

    def __init__(
        self,
        save_function: Callable[[str], Any],
        parent=None,
        directory=HOME_DIRECTORY,
        history=None,
    ) -> None:
        super().__init__(parent, trans._('Save screenshot'))
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setFileMode(QFileDialog.AnyFile)
        self.setNameFilter(
            trans._('Image files (*.png *.bmp *.gif *.tif *.tiff)')
        )
        self.setDirectory(directory)
        self.setHistory(history)

        if in_ipython():
            self.setOptions(QFileDialog.DontUseNativeDialog)

        self.save_function = save_function

    def accept(self):
        save_path = self.selectedFiles()[0]
        if os.path.splitext(save_path)[1] == '':
            save_path = save_path + '.png'
            if os.path.exists(save_path):
                res = QMessageBox().warning(
                    self,
                    trans._('Confirm overwrite'),
                    trans._(
                        '{save_path} already exists. Do you want to replace it?',
                        save_path=save_path,
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res != QMessageBox.Yes:
                    # return in this case since a valid name for the
                    # file is needed so the dialog needs to be visible
                    return
        super().accept()
        if self.result():
            self.save_function(save_path)
