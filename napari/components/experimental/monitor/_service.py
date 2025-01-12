"""MonitorService class.

Experimental shared memory service.

Requires Python 3.9, for now at least.

Monitor Config File
-------------------
Only if NAPARI_MON is set and points to a config file will the monitor even
start. The format of the .napari-mon config file is:

{
    "clients": [
        ["python", "/tmp/myclient.py"]
    ]
    "log_path": "/tmp/monitor.log"
}

All of the listed clients will be started. They can be the same program run
with different arguments, or different programs. All clients will have
access to the same shared memory.

Client Config File
-------------------
The client should decode the contents of the NAPARI_MON_CLIENT variable. It
can be decoded like this:

    def _get_client_config() -> dict:
        env_str = os.getenv("NAPARI_MON_CLIENT")
        if env_str is None:
            return None

        env_bytes = env_str.encode('ascii')
        config_bytes = base64.b64decode(env_bytes)
        config_str = config_bytes.decode('ascii')

        return json.loads(config_str)

Today the client configuration is only:

    {
        "server_port": "<number>"
    }

Client Startup
--------------

See a working client: https://github.com/pwinston/webmon

The client can access the MonitorApi by creating a SharedMemoryManager:

    napari_api = ['shutdown_event', 'command_queue', 'data']
    for name in napari_api:
        SharedMemoryManager.register(name)

    SharedMemoryManager.register('command_queue')
    self.manager = SharedMemoryManager(
        address=('localhost', config['server_port']),
        authkey=str.encode('napari')
    )

    # Get the shared resources.
    shutdown = self._manager.shutdown_event()
    commands = self._manager.command_queue()
    data = self._manager.data()

It can send command like:

    commands.put(
       {"test_command": {"value": 42, "names": ["fred", "joe"]}}
    )

Passing Data From Napari To The Client
--------------------------------------
In napari add data like:

    if monitor:
        monitor.add({
            "tiled_image_layer": {
                "num_created": stats.created,
                "num_deleted": stats.deleted,
                "duration_ms": elapsed.duration_ms,
            }
        })

The client can access data['tiled_image_layer']. Clients should be
resilient to missing data. Nn case the napari version is different than
expected, or is just not producing that data for some reason.

Future Work
-----------
We plan to investigate the use of numpy shared memory buffers for bulk
binary data. Possibly using recarray to organized things.
"""

import copy
import logging
import os
import subprocess
from multiprocessing.managers import SharedMemoryManager

from napari.components.experimental.monitor._utils import base64_encoded_json

LOGGER = logging.getLogger('napari.monitor')

# If False we don't start any clients, for debugging.
START_CLIENTS = True

# We pass the data in this template to each client as an encoded
# NAPARI_MON_CLIENT environment variable.
client_config_template: dict[str, str | int] = {
    'server_port': '<number>',
}


def _create_client_env(server_port: int) -> dict:
    """Create and return the environment for the client.

    Parameters
    ----------
    server_port : int
        The port the client should connect to.
    """
    # Every client gets the same config. Copy template and then stuff
    # in the correct values.
    client_config = copy.deepcopy(client_config_template)
    client_config['server_port'] = server_port

    # Start with our environment and just add in the one variable.
    env = os.environ.copy()
    env.update({'NAPARI_MON_CLIENT': base64_encoded_json(client_config)})
    return env


class MonitorService:
    """Make data available to a client via shared memory.

    Originally we used a ShareableList and serialized JSON into one of the
    slots in the list. However now we are using the MonitorApi._data
    dict proxy object instead. That serializes to JSON under the hood,
    but it's nicer that doing int ourselves.

    So this class is not doing much right now. However if we add true
    shared memory buffers for numpy, etc. then this class might manager
    those.
    """

    def __init__(self, config: dict, manager: SharedMemoryManager) -> None:
        super().__init__()
        self._config = config
        self._manager = manager

        if START_CLIENTS:
            self._start_clients()

    def _start_clients(self) -> None:
        """Start every client in our config."""
        # We asked for port 0 which means the OS will pick a port, we
        # save it off so we can send it the clients are starting up.
        server_port = self._manager.address[1]
        LOGGER.info('Listening on port %s', server_port)

        num_clients = len(self._config['clients'])
        LOGGER.info('Starting %d clients...', num_clients)

        env = _create_client_env(server_port)

        # Start every client.
        for args in self._config['clients']:
            LOGGER.info('Starting client %s', args)

            # Use Popen to run and not wait for the process to finish.
            subprocess.Popen(args, env=env)

        LOGGER.info('Started %d clients.', num_clients)

    def stop(self) -> None:
        """Stop the shared memory service."""
        LOGGER.info('MonitorService.stop')
        self._manager.shutdown()
