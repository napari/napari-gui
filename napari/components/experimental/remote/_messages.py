"""RemoteMessages class.

Sends messages to remote clients.
"""

import logging
import time

from napari.components.experimental.monitor import monitor
from napari.components.layerlist import LayerList

LOGGER = logging.getLogger('napari.monitor')


class RemoteMessages:
    """Sends messages to remote clients.

    Parameters
    ----------
    layers : LayerList
        The viewer's layers, so we can call into them.
    """

    def __init__(self, layers: LayerList) -> None:
        self.layers = layers
        self._frame_number = 0
        self._last_time: float | None = None

    def on_poll(self) -> None:
        """Send messages to clients.

        These message go out once per frame. So it might not make sense to
        include static information that rarely changes. Although if it's
        small, maybe it's okay.

        The message looks like:

        {
            "poll": {
                "layers": {
                    13482484: {
                        "tile_state": ...
                        "tile_config": ...
                    }
                }
            }
        }
        """
        self._frame_number += 1

        layers: dict[int, dict] = {}

        monitor.add_data({'poll': {'layers': layers}})
        self._send_frame_time()

    def _send_frame_time(self) -> None:
        """Send the frame time since last poll."""
        now = time.time()
        last = self._last_time
        delta = now - last if last is not None else 0
        delta_ms = delta * 1000

        monitor.send_message(
            {'frame_time': {'time': now, 'delta_ms': delta_ms}}
        )
        self._last_time = now
