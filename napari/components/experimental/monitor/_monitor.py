"""Monitor class.

The Monitor class wraps the MonitorServer and MonitorApi. One reason
for having a wrapper class is that so the rest of napari does not
need to import any multiprocessing code unless actually using
the monitor.
"""

import errno
import json
import logging
import os
from pathlib import Path

from napari.utils.translations import trans

LOGGER = logging.getLogger('napari.monitor')

# If False monitor is disabled even if we meet all other requirements.
ENABLE_MONITOR = True


def _load_config(path: str) -> dict:
    """Load the JSON formatted config file.

    Parameters
    ----------
    path : str
        The path of the JSON file we should load.

    Returns
    -------
    dict
        The parsed data from the JSON file.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            errno.ENOENT,
            trans._(
                'Monitor: Config file not found: {path}',
                deferred=True,
                path=path,
            ),
        )

    with path.open() as infile:
        return json.load(infile)


def _load_monitor_config() -> dict | None:
    """Return the MonitorService config file data, or None.

    Returns
    -------
    Optional[dict]
        The parsed config file data or None if no config.
    """
    # We shouldn't even call into this file unless NAPARI_MON is defined
    # but check to be sure.
    value = os.getenv('NAPARI_MON')
    if value in [None, '0']:
        return None

    return _load_config(value)


def _setup_logging(config: dict) -> None:
    """Log "napari.monitor" messages to the configured file.

    Parameters
    ----------
    config : dict
        Monitor configuration
    """
    try:
        log_path = config['log_path']
    except KeyError:
        return  # No log file.

    # Nuke/reset log for now.
    # Path(log_path).unlink()

    fh = logging.FileHandler(log_path)
    LOGGER.addHandler(fh)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.info('Writing to log path %s', log_path)


def _get_monitor_config() -> dict | None:
    """Create and return the configuration for the MonitorService.

    The routine might return None for one serveral reasons:
    1) We're not running under Python 3.9 or now.
    2) The monitor is explicitly disable, ENABLED_MONITOR is False.
    3) The NAPARI_MON environment variable is not defined.
    4) The NAPARI_MON config file cannot be found and parsed.

    Returns
    -------
    Optional[dict]
        The configuration for the MonitorService.
    """

    if not ENABLE_MONITOR:
        logging.warning('Monitor: not starting, disabled')
        return None

    # The NAPARI_MON environment variable points to our config file.
    config = _load_monitor_config()

    if config is None:
        logging.warning('Monitor: not starting, no usable config file')
        return None

    return config


class Monitor:
    """Wraps the monitor service.

    We can't start the monitor service at import time. Under the hood the
    multiprocessing complains about a "partially started process".

    Instead someone must call our start() method explicitly once the
    process has fully started.
    """

    def __init__(self) -> None:
        # Both are set when start() is called, and only if we have
        # a parseable config file, have Python 3.9, etc.
        self._service = None
        self._api = None
        self._running = False

    def __nonzero__(self) -> bool:
        """Return True if the service is running.

        So that callers can do:

            if monitor:
                monitor.add(...)
        """
        return self._running

    @property
    def run_command_event(self):
        """The MonitorAPI fires this event for commands from clients."""
        return self._api.events.run_command

    def start(self) -> bool:
        """Start the monitor service, if it hasn't been started already.

        Returns
        -------
        bool
            True if we started the service or it was already started.
        """
        if self._running:
            return True  # It was already started.

        config = _get_monitor_config()

        if config is None:
            return False  # Can't start without config.

        _setup_logging(config)

        # Late imports so no multiprocessing modules are even
        # imported unless we are going to start the service.
        from napari.components.experimental.monitor._api import MonitorApi
        from napari.components.experimental.monitor._service import (
            MonitorService,
        )

        # Create the API first. It will register our callbacks, then
        # we start the manager that will serve those callbacks.
        self._api = MonitorApi()

        # Now we can start our service.
        self._service = MonitorService(config, self._api.manager)

        self._running = True
        return True  # We started the service.

    def stop(self) -> None:
        """Stop the monitor service."""
        if not self._running:
            return

        self._api.stop()
        self._api = None

        self._service.stop()
        self._service = None

        self._running = False

    def on_poll(self, event) -> None:
        """The QtPoll object polls us.

        Probably we could get rid of polling by creating a thread that
        blocks waiting for client messages. Then it posts those messages as
        Qt Events. So the GUI doesn't block, but gracefully handles
        incoming messages as Qt events.
        """
        if self._running:
            self._api.poll()

        # Handle the event to say "keep polling us".
        event.handled = True

    def add_data(self, data) -> None:
        """Add data to the monitor service.

        Caller should use this pattern:
            if monitor:
                monitor.add(...)

        So no time wasted assembling the dict unless the monitor is running.
        """
        if self._running:
            self._api.add_napari_data(data)

    def send_message(self, message: dict) -> None:
        """Send a message to shared memory clients.

        Parameters
        ----------
        message : dict
            Post this message to clients.
        """
        if self._running:
            self._api.send_napari_message(message)


monitor = Monitor()
