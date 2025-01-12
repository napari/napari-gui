"""QtPoll class.

Poll visuals or other objects so they can do things even when the
mouse/camera are not moving. Usually for just a short period of time.
"""

import time

from qtpy.QtCore import QEvent, QObject, QTimer

from napari.utils.events import EmitterGroup

# When running a timer we use this interval.
POLL_INTERVAL_MS = 16  # About 60HZ, needs to be an int for QTimer setInterval

# If called more often than this we ignore it. Our _on_camera() method can
# be called multiple times in on frame. It can get called because the
# "center" changed and then the "zoom" changed even if it was really from
# the same camera movement.
IGNORE_INTERVAL_MS = 10


class QtPoll(QObject):
    """Polls anything once per frame via an event.

    QtPoll was first created for VispyTiledImageLayer. It polls the visual
    when the camera moves. However, we also want visuals to keep loading
    chunks even when the camera stops. We want the visual to finish up
    anything that was in progress. Before it goes fully idle.

    QtPoll will poll those visuals using a timer. If the visual says the
    event was "handled" it means the visual has more work to do. If that
    happens, QtPoll will continue to poll and draw the visual it until the
    visual is done with the in-progress work.

    An analogy is a snow globe. The user moving the camera shakes up the
    snow globe. We need to keep polling/drawing things until all the snow
    settles down. Then everything will stay completely still until the
    camera is moved again, shaking up the globe once more.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    camera : Camera
        The viewer's main camera.
    """

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)

        self.events = EmitterGroup(source=self, poll=None)

        self.timer = QTimer()
        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(self._on_timer)
        self._interval = IntervalTimer()

    def on_camera(self) -> None:
        """Called when camera view changes."""
        # When the mouse button is down and the camera is being zoomed
        # or panned, timer events are starved out. So we call poll
        # explicitly here. It will start the timer if needed so that
        # polling can continue even after the camera stops moving.
        self._poll()

    def wake_up(self) -> None:
        """Wake up QtPoll so it starts polling."""
        # Start the timer so that we start polling. We used to poll once
        # right away here, but it led to crashes. Because we polled during
        # a paintGL event?
        if not self.timer.isActive():
            self.timer.start()

    def _on_timer(self) -> None:
        """Called when the timer is running.

        The timer is running which means someone we are polling still has
        work to do.
        """
        self._poll()

    def _poll(self) -> None:
        """Called on camera move or with the timer."""

        # Between timers and camera and wake_up() we might be called multiple
        # times in quick succession. Use an IntervalTimer to ignore these
        # near-duplicate calls.
        if self._interval.elapsed_ms < IGNORE_INTERVAL_MS:
            return

        # Poll all listeners.
        event = self.events.poll()

        # Listeners will "handle" the event if they need more polling. If
        # no one needs polling, then we can stop the timer.
        if not event.handled:
            self.timer.stop()
            return

        # Someone handled the event, so they want to be polled even if
        # the mouse doesn't move. So start the timer if needed.
        if not self.timer.isActive():
            self.timer.start()

    def closeEvent(self, _event: QEvent) -> None:
        """Cleanup and close.

        Parameters
        ----------
        _event : QEvent
            The close event.
        """
        self.timer.stop()
        self.deleteLater()


class IntervalTimer:
    """Time the interval between subsequent calls to our elapsed property."""

    def __init__(self) -> None:
        self._last: float | None = None

    @property
    def elapsed_ms(self) -> float:
        """The elapsed time since the last call to this property."""
        now = time.time()
        elapsed_seconds = 0 if self._last is None else now - self._last
        self._last = now
        return elapsed_seconds * 1000
