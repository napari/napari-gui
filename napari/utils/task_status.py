import datetime
import uuid
from enum import auto
from typing import Optional

from napari.utils.misc import Callable, StringEnum, Tuple


class Status(StringEnum):
    PENDING = auto()
    BUSY = auto()
    DONE = auto()
    FAILED = auto()


class TaskStatusItem:
    def __init__(
        self,
        provider: str,
        status: Status,
        description: str,
        cancel_callback: Optional[Callable] = None,
    ):
        self._id = uuid.uuid4()
        self._provider = provider
        self._timestamp = [self._timestap()]
        self._status = [status]
        self._description = [description]
        self._cancel_callback = cancel_callback

    def _timestap(self) -> str:
        return datetime.datetime.now().isoformat()

    def __str__(self) -> str:
        return f'TaskStatusItem: ({self._provider}, {self._id}, {self._timestamp[-1]}, {self._status[-1]}, {self._description[-1]})'

    def update(self, status: Status, description: str):
        self._timestamp.append(self._timestap())
        self._status.append(status)
        self._description.append(description)

    def cancel(self) -> bool:
        if self._cancel_callback is not None:
            return self._cancel_callback()
        return False

    def state(self) -> Tuple[str, Status, str]:
        return (
            self._provider,
            self._timestamp[-1],
            self._status[-1],
            self._description[-1],
        )


class TaskStatusManager:
    """
    A task status manager, to store status of long running processes/tasks.

    Only one instance is in general available through napari.

    napari methods and plugins can use it to register and update
    long running tasks.
    """

    _tasks: dict[uuid.UUID, TaskStatusItem]

    def __init__(self) -> None:
        self._tasks: dict[uuid.UIID, TaskStatusItem] = {}

    def register_task_status(
        self,
        provider: str,
        task_status: Status,
        description: str,
        cancel_callback: Optional[Callable] = None,
    ) -> uuid.UUID:
        item = TaskStatusItem(
            provider, task_status, description, cancel_callback
        )
        self._tasks[item.id] = item
        return item.id

    def update_task_status(
        self,
        status_id: uuid.UUID,
        task_status: Status,
        description: Optional[str] = None,
    ) -> bool:
        if status_id in self._tasks:
            item = self._tasks[status_id]
            item.update(task_status, description)
            return True

        return False

    def is_busy(self) -> bool:
        return len(self._tasks) > 0

    def get_status(self) -> list[str]:
        messages = []
        for _, item in self._tasks.items():
            provider, ts, status, description = item.state()
            if status in [Status.PENDING, Status.BUSY]:
                messages.append(f'{provider}: {ts} - {description}')

        return messages

    def cancel_all(self) -> None:
        for _, item in self._tasks.items():
            item.cancel()


task_status_manager = TaskStatusManager()


def register_task_status(
    provider: str,
    task_status: Status,
    description: str,
    cancel_callback: Optional[Callable] = None,
) -> uuid.UUID:
    """
    Register a long running task.
    """
    return task_status_manager.register_task_status(
        provider, task_status, description, cancel_callback
    )


def update_task_status(
    task_status_id: uuid.UUID,
    status: Status,
    description: Optional[str] = None,
) -> bool:
    """
    Update a long running task.
    """
    return task_status_manager.update_task_status(
        task_status_id, status, description
    )
