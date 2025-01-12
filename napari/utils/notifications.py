from __future__ import annotations

import logging
import os
import sys
import threading
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import auto
from types import TracebackType

from napari.utils.events import Event, EventEmitter
from napari.utils.misc import StringEnum

name2num = {
    'error': 40,
    'warning': 30,
    'info': 20,
    'debug': 10,
    'none': 0,
}

__all__ = [
    'ErrorNotification',
    'Notification',
    'NotificationManager',
    'NotificationSeverity',
    'WarningNotification',
    'show_console_notification',
    'show_debug',
    'show_error',
    'show_info',
    'show_warning',
]


class NotificationSeverity(StringEnum):
    """Severity levels for the notification dialog.  Along with icons for each."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    DEBUG = auto()
    NONE = auto()

    def as_icon(self):
        return {
            self.ERROR: 'ⓧ',
            self.WARNING: '⚠️',
            self.INFO: 'ⓘ',
            self.DEBUG: '🐛',
            self.NONE: '',
        }[self]

    def __lt__(self, other):
        return name2num[str(self)] < name2num[str(other)]

    def __le__(self, other):
        return name2num[str(self)] <= name2num[str(other)]

    def __gt__(self, other):
        return name2num[str(self)] > name2num[str(other)]

    def __ge__(self, other):
        return name2num[str(self)] >= name2num[str(other)]

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.value)


ActionSequence = Sequence[tuple[str, Callable[[], None]]]


class Notification(Event):
    """A Notifcation event.  Usually created by :class:`NotificationManager`.

    Parameters
    ----------
    message : str
        The main message/payload of the notification.
    severity : str or NotificationSeverity, optional
        The severity of the notification, by default
        `NotificationSeverity.WARNING`.
    actions : sequence of tuple, optional
        Where each tuple is a `(str, callable)` 2-tuple where the first item
        is a name for the action (which may, for example, be put on a button),
        and the callable is a callback to perform when the action is triggered.
        (for example, one might show a traceback dialog). by default ()
    """

    def __init__(
        self,
        message: str,
        severity: str | NotificationSeverity = NotificationSeverity.WARNING,
        actions: ActionSequence = (),
        **kwargs,
    ) -> None:
        self.severity = NotificationSeverity(severity)
        super().__init__(type_name=str(self.severity).lower(), **kwargs)
        self._message = message
        self.actions = actions

        # let's store when the object was created;
        self.date = datetime.now()

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):
        self._message = value

    @classmethod
    def from_exception(cls, exc: BaseException, **kwargs) -> Notification:
        return ErrorNotification(exc, **kwargs)

    @classmethod
    def from_warning(cls, warning: Warning, **kwargs) -> Notification:
        return WarningNotification(warning, **kwargs)

    def __str__(self):
        return f'{str(self.severity).upper()}: {self.message}'


class ErrorNotification(Notification):
    """
    Notification at an Error severity level.
    """

    exception: BaseException

    def __init__(self, exception: BaseException, *args, **kwargs) -> None:
        msg = getattr(exception, 'message', str(exception))
        actions = getattr(exception, 'actions', ())
        super().__init__(msg, NotificationSeverity.ERROR, actions)
        self.exception = exception

    def as_html(self):
        from napari.utils._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=True)

    def as_text(self):
        from napari.utils._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=False, color='NoColor')

    def __str__(self):
        from napari.utils._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=False)


class WarningNotification(Notification):
    """
    Notification at a Warning severity level.
    """

    warning: Warning

    def __init__(
        self, warning: Warning, filename=None, lineno=None, *args, **kwargs
    ) -> None:
        msg = getattr(warning, 'message', str(warning))
        actions = getattr(warning, 'actions', ())
        super().__init__(msg, NotificationSeverity.WARNING, actions)
        self.warning = warning
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        category = type(self.warning).__name__
        return f'{self.filename}:{self.lineno}: {category}: {self.warning}!'


class NotificationManager:
    """
    A notification manager, to route all notifications through.

    Only one instance is in general available through napari; as we need
    notification to all flow to a single location that is registered with the
    sys.except_hook  and showwarning hook.

    This can and should be used a context manager; the context manager will
    properly re-entered, and install/remove hooks and keep them in a stack to
    restore them.

    While it might seem unnecessary to make it re-entrant; or to make the
    re-entrancy no-op; one need to consider that this could be used inside
    another context manager that modify except_hook and showwarning.

    Currently the original except and show warnings hooks are not called; but
    this could be changed in the future; this poses some questions with the
    re-entrency of the hooks themselves.
    """

    records: list[Notification]
    _instance: NotificationManager | None = None

    def __init__(self) -> None:
        self.records: list[Notification] = []
        self.exit_on_error = os.getenv('NAPARI_EXIT_ON_ERROR') in ('1', 'True')
        self.catch_error = os.getenv('NAPARI_CATCH_ERRORS') not in (
            '0',
            'False',
        )
        self.notification_ready = self.changed = EventEmitter(
            source=self, event_class=Notification
        )
        self._originals_except_hooks: list[Callable] = []
        self._original_showwarnings_hooks: list[Callable] = []
        self._originals_thread_except_hooks: list[Callable] = []
        self._seen_warnings: set[tuple[str, type, str, int]] = set()

    def __enter__(self):
        self.install_hooks()
        return self

    def __exit__(self, *args, **kwargs):
        self.restore_hooks()

    def install_hooks(self):
        """
        Install a `sys.excepthook`, a `showwarning` hook and a
        threading.excepthook to display any message in the UI,
        storing the previous hooks to be restored if necessary.
        """
        if getattr(threading, 'excepthook', None):
            # TODO: we might want to display the additional thread information
            self._originals_thread_except_hooks.append(threading.excepthook)
            threading.excepthook = self.receive_thread_error
        else:
            # Patch for Python < 3.8
            _setup_thread_excepthook()

        self._originals_except_hooks.append(sys.excepthook)
        self._original_showwarnings_hooks.append(warnings.showwarning)

        sys.excepthook = self.receive_error
        warnings.showwarning = self.receive_warning

    def restore_hooks(self):
        """
        Remove hooks installed by `install_hooks` and restore previous hooks.
        """
        if getattr(threading, 'excepthook', None):
            # `threading.excepthook` available only for Python >= 3.8
            threading.excepthook = self._originals_thread_except_hooks.pop()

        sys.excepthook = self._originals_except_hooks.pop()
        warnings.showwarning = self._original_showwarnings_hooks.pop()

    def dispatch(self, notification: Notification):
        self.records.append(notification)
        self.notification_ready(notification)

    def receive_thread_error(
        self,
        args: tuple[
            type[BaseException],
            BaseException,
            TracebackType | None,
            threading.Thread | None,
        ],
    ):
        self.receive_error(*args)

    def receive_error(
        self,
        exctype: type[BaseException],
        value: BaseException,
        traceback: TracebackType | None = None,
        thread: threading.Thread | None = None,
    ):
        if isinstance(value, KeyboardInterrupt):
            sys.exit('Closed by KeyboardInterrupt')

        if self.exit_on_error:
            sys.__excepthook__(exctype, value, traceback)
            sys.exit('Exit on error')
        if not self.catch_error:
            sys.__excepthook__(exctype, value, traceback)
            return
        self.dispatch(Notification.from_exception(value))

    def receive_warning(
        self,
        message: Warning,
        category: type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line=None,
    ):
        msg = message if isinstance(message, str) else message.args[0]
        if (msg, category, filename, lineno) in self._seen_warnings:
            return
        self._seen_warnings.add((msg, category, filename, lineno))
        self.dispatch(
            Notification.from_warning(
                message, filename=filename, lineno=lineno
            )
        )

    def receive_info(self, message: str):
        self.dispatch(Notification(message, 'INFO'))


notification_manager = NotificationManager()


def show_debug(message: str):
    """
    Show a debug message in the notification manager.
    """
    notification_manager.dispatch(
        Notification(message, severity=NotificationSeverity.DEBUG)
    )


def show_info(message: str):
    """
    Show an info message in the notification manager.
    """
    notification_manager.dispatch(
        Notification(message, severity=NotificationSeverity.INFO)
    )


def show_warning(message: str):
    """
    Show a warning in the notification manager.
    """
    notification_manager.dispatch(
        Notification(message, severity=NotificationSeverity.WARNING)
    )


def show_error(message: str):
    """
    Show an error in the notification manager.
    """
    notification_manager.dispatch(
        Notification(message, severity=NotificationSeverity.ERROR)
    )


def show_console_notification(notification: Notification):
    """
    Show a notification in the console.
    """
    try:
        from napari.settings import get_settings

        if (
            notification.severity
            < get_settings().application.console_notification_level
        ):
            return

        print(notification)  # noqa: T201
    except Exception:
        logging.exception(
            'An error occurred while trying to format an error and show it in console.\n'
            'You can try to uninstall IPython to disable rich traceback formatting\n'
            'And/or report a bug to napari'
        )
        # this will likely get silenced by QT.
        raise


def _setup_thread_excepthook():
    """
    Workaround for `sys.excepthook` thread bug from:
    http://bugs.python.org/issue1230540
    """
    _init = threading.Thread.__init__

    def init(self, *args, **kwargs):
        _init(self, *args, **kwargs)
        _run = self.run

        def run_with_except_hook(*args2, **kwargs2):
            try:
                _run(*args2, **kwargs2)
            except Exception:  # noqa BLE001
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init
