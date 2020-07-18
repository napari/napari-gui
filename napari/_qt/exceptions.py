import logging
import traceback
from types import TracebackType
from typing import Type

from qtpy.QtCore import QObject, Signal


class ExceptionHandler(QObject):
    """General class to handle all raise exception errors in the GUI"""

    error = Signal(tuple)

    def __init__(self):
        super().__init__()

    def handler(
        self,
        etype: Type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        """Our sys.excepthook handler.

        This function handles uncaught exceptions and can delegates to a
        secondary handler, whether it be a GUI dialog, or an IPython traceback
        printout.

        The three parameters here are what would be returned from
        :func:`sys.exc_info()`.

        Parameters
        ----------
        etype : Type[BaseException]
            The type of error raised
        value : BaseException
            The error instance
        tb : TracebackType
            The traceback object associated with the error.
        """
        # etype.__module__ contains the module raising the error
        # Custom exception classes can have different behavior
        # can add custom exception handlers here ...
        text = "".join(traceback.format_exception(etype, value, tb))
        logging.error("Unhandled exception:\n%s", text)
        self.error.emit((etype, value, tb))
