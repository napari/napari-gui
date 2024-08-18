from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from psygnal import Signal

_LOG_SEPARATOR = '<NAPARI_LOG_SEPARATOR>'


if TYPE_CHECKING:
    from typing import Any


class _LogStream:
    changed = Signal()

    def __init__(self) -> None:
        self.logs: deque[tuple[Any, ...]] = deque(maxlen=100_000)

    def write(self, log_msg: str) -> None:
        logger, level_name, time, thread, msg = log_msg.split(_LOG_SEPARATOR)
        levels = logging.getLevelNamesMapping()
        level_value = levels[level_name]
        self.logs.append((logger, level_value, level_name, time, thread, msg))
        # TODO: actually save log to a file somewhere so it can be retrieved?
        self.changed()

    def flush(self) -> None:
        pass

    def get_filtered_logs_html(
        self,
        level: int = logging.DEBUG,
        text_filter: str = '',
        last_only: bool = False,
    ) -> list[str]:
        if isinstance(level, str):
            level = logging.getLevelNamesMapping().get(level, logging.NOTSET)

        logs = [LOG_STREAM.logs[-1]] if last_only else LOG_STREAM.logs

        selected = [
            (logger_name, level_value, *others)
            for logger_name, level_value, *others in logs
            if level_value >= level
        ]

        # TODO: fuzzy search?
        text_filter = text_filter.lower()
        filtered = [
            log
            for log in selected
            if any(text_filter in str(field).lower() for field in log)
        ]

        return [
            f'{_html_level(level_name, level_value)} '
            f'<b>{name}</b> '
            f'<font style="color:gray"><i>[{time}] ({thread})</i></font>: '
            f'{msg}'
            for name, level_value, level_name, time, thread, msg in filtered
        ]


LOG_STREAM = _LogStream()
LOG_HANDLER = logging.StreamHandler(LOG_STREAM)
LOG_HANDLER.setFormatter(
    logging.Formatter(
        f'%(name)s{_LOG_SEPARATOR}%(levelname)s{_LOG_SEPARATOR}%(asctime)s{_LOG_SEPARATOR}%(threadName)s{_LOG_SEPARATOR}%(message)s'
    )
)
LOG_HANDLER.setLevel(logging.DEBUG)


def register_logger(module: str) -> None:
    logger = logging.getLogger(module)
    # ensure the default "last resort" logging to console remains
    if not logger.handlers and logging.lastResort:
        logger.addHandler(logging.lastResort)
    logger.addHandler(LOG_HANDLER)


def _html_level(level_name: str, level_value: int) -> str:
    colors = {
        logging.INFO: 'cyan',
        logging.WARNING: 'orange',
        logging.ERROR: 'red',
        logging.CRITICAL: 'magenta',
    }
    color = 'blue'
    for level, level_color in colors.items():
        if level_value >= level:
            color = level_color
    # this is ugly AF but html is weird and I don't get it
    padding = '&nbsp;' * (8 - len(level_name))
    return f'<font style="color:{color}">{level_name}{padding}</font>'
