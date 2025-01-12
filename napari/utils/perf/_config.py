"""Perf configuration flags."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import wrapt

from napari.utils.perf._patcher import patch_callables
from napari.utils.perf._timers import perf_timer
from napari.utils.translations import trans

PERFMON_ENV_VAR = 'NAPARI_PERFMON'


class PerfmonConfigError(Exception):
    """Error parsing or interpreting config file."""


def _patch_perf_timer(
    parent: ModuleType | type, callable_name: str, label: str
) -> None:
    """Patches the callable to run it inside a perf_timer.

    Parameters
    ----------
    parent
        The module or class that contains the callable.
    callable_name : str
        The name of the callable (function or method).
    label : str
        The <function> or <class>.<method> we are patching.
    """

    @wrapt.patch_function_wrapper(parent, callable_name)
    def perf_time_callable(
        wrapped: Callable,
        instance: Any,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> Callable:
        with perf_timer(f'{label}'):
            return wrapped(*args, **kwargs)


class PerfmonConfig:
    """Reads the perfmon config file and sets up performance monitoring.

    Parameters
    ----------
    config_path : Path
        Path to the perfmon configuration file (JSON format).

    Config File Format
    ------------------
    {
        "trace_qt_events": true,
        "trace_file_on_start": "/Path/To/latest.json",
        "trace_callables": [
            "my_callables_1",
            "my_callables_2",
        ],
        "callable_lists": {
            "my_callables_1": [
                "module1.module2.Class1.method1",
                "module1.Class2.method2",
                "module2.module3.function1"
            ],
            "my_callables_2": [
                ...
            ]
        }
    }
    """

    def __init__(self, config_path: str | None) -> None:
        # Should only patch once, but it can't be on module load, user
        # should patch once main() as started running during startup.
        self.patched = False

        self.config_path = config_path
        if config_path is None:
            return  # Legacy mode, trace Qt events only.

        path = Path(config_path)
        with path.open() as infile:
            self.data = json.load(infile)

    def patch_callables(self) -> None:
        """Patch callables according to the config file.

        Call once at startup but after main() has started running. Do not
        call at module init or you will likely get circular dependencies.
        This function potentially imports many modules.
        """
        if self.config_path is None:
            return  # disabled

        assert self.patched is False
        self._patch_callables()
        self.patched = True

    def _get_callables(self, list_name: str) -> list[str]:
        """Get the list of callables from the config file.

        list_name : str
            The name of the list to return.
        """
        try:
            return self.data['callable_lists'][list_name]
        except KeyError as e:
            raise PerfmonConfigError(
                trans._(
                    "{path} has no callable list '{list_name}'",
                    deferred=True,
                    path=self.config_path,
                    list_name=list_name,
                )
            ) from e

    def _patch_callables(self) -> None:
        """Add a perf_timer to every callable.

        Notes
        -----
        data["trace_callables"] should contain the names of one or more
        lists of callables which are defined in data["callable_lists"].
        """
        for list_name in self.data['trace_callables']:
            callable_list = self._get_callables(list_name)
            patch_callables(callable_list, _patch_perf_timer)

    @property
    def trace_qt_events(self) -> bool:
        """Return True if we should time Qt events."""
        if self.config_path is None:
            return True  # always trace qt events in legacy mode
        try:
            return self.data['trace_qt_events']
        except KeyError:
            return False

    @property
    def trace_file_on_start(self) -> str | None:
        """Return path of trace file to write or None."""
        if self.config_path is None:
            return None  # don't trace on start in legacy mode
        try:
            path = self.data['trace_file_on_start']

            # Return None if it was empty string or false.
        except KeyError:
            return None
        else:
            return path or None


def _create_perf_config() -> PerfmonConfig | None:
    value = os.getenv('NAPARI_PERFMON')

    if value is None or value == '0':
        return None  # Totally disabled
    if value == '1':
        return PerfmonConfig(None)  # Legacy no config, Qt events only.

    return PerfmonConfig(value)  # Normal parse the config file.


# The global instance
perf_config = _create_perf_config()
