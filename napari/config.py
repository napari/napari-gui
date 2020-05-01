"""napari configuration

This module is copied, with modifications, from dask.config

We like Dask's "plain python dict" treatment of configurations, handling of
nested namespaces, backwards compatilibity of new settings, context management
(temporary setting) of config values, choice of yaml for persistence, and
environment variable options

see documentation for dask.config at:
https://docs.dask.org/en/latest/configuration.html

Configuration is specified in one of the following ways:

1. YAML files in ~/.config/napari/ or /etc/napari/
2. Environment variables that start with ``NAPARI_``
3. Default settings within sub-libraries

This combination makes it easy to specify configuration in a variety of
settings ranging from personal workstations, to IT-mandated configuration, to
docker images.
"""
import ast
import builtins
import logging
import os
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from types import TracebackType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml

from .utils.misc import StringEnum

no_default = "__no_default__"


paths = [
    os.getenv("NAPARI_ROOT_CONFIG", "/etc/napari"),
    os.path.join(sys.prefix, "etc", "napari"),
    os.path.join(os.path.expanduser("~"), ".config", "napari"),
    os.path.join(os.path.expanduser("~"), ".napari"),
]

if "NAPARI_CONFIG" in os.environ:
    PATH = os.environ["NAPARI_CONFIG"]
    paths.append(PATH)
else:
    PATH = os.path.join(os.path.expanduser("~"), ".config", "napari")


config: Dict[str, Any] = {}
global_config = config

config_lock = threading.Lock()

defaults: List[dict] = []


def canonical_name(k: str, config: Mapping) -> str:
    """Return the canonical name for a key.

    Handles user choice of '-' or '_' conventions by standardizing on whichever
    version was set first. If a key already exists in either hyphen or
    underscore form, the existing version is the canonical name. If neither
    version exists the original key is used as is.
    """
    try:
        if k in config:
            return k
    except TypeError:
        # config is not a mapping, return the same name as provided
        return k

    altk = k.replace("_", "-") if "_" in k else k.replace("-", "_")

    if altk in config:
        return altk

    return k


def update(old: dict, new: dict, priority="new") -> dict:
    """Update a nested dictionary with values from another.

    This is like dict.update except that it smoothly merges nested values

    This operates in-place and modifies old

    Parameters
    ----------
    priority: string {'old', 'new'}
        If new (default) then the new dictionary has preference.
        Otherwise the old dictionary does.

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b)
    {'x': 2, 'y': {'a': 2, 'b': 3}}

    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b, priority='old')
    {'x': 1, 'y': {'a': 2, 'b': 3}}

    See Also
    --------
    napari.config.merge
    """
    for k, v in new.items():
        k = canonical_name(k, old)

        if isinstance(v, Mapping):
            if k not in old or old[k] is None:
                old[k] = {}
            update(old[k], v, priority=priority)  # type: ignore
        else:
            if priority == "new" or k not in old:
                old[k] = v

    return old


def merge(*dicts: dict) -> dict:
    """Update a sequence of nested dictionaries.

    This prefers the values in the latter dictionaries to those in the former

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'y': {'b': 3}}
    >>> merge(a, b)
    {'x': 1, 'y': {'a': 2, 'b': 3}}

    See Also
    --------
    napari.config.update
    """
    result: dict = {}
    for d in dicts:
        update(result, d)
    return result


def collect_yaml(paths: List[str] = paths) -> List[dict]:
    """Collect configuration from yaml files.

    This searches through a list of paths, expands to find all yaml or json
    files, and then parses each file.
    """
    # Find all paths
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    file_paths.extend(
                        sorted(
                            [
                                os.path.join(path, p)
                                for p in os.listdir(path)
                                if os.path.splitext(p)[1].lower()
                                in (".json", ".yaml", ".yml")
                            ]
                        )
                    )
                except OSError:
                    # Ignore permission errors
                    pass
            else:
                file_paths.append(path)

    configs = []

    # Parse yaml files
    for path in file_paths:
        try:
            with open(path) as f:
                data = yaml.safe_load(f.read()) or {}
                configs.append(data)
        except (OSError, IOError):
            # Ignore permission errors
            pass

    return configs


def collect_env(env: Optional[Union[dict, os._Environ]] = None) -> dict:
    """Collect config from environment variables.

    This grabs environment variables of the form "NAPARI_FOO__BAR_BAZ=123" and
    turns these into config variables of the form ``{"foo": {"bar-baz": 123}}``
    It transforms the key and value in the following way:

    -  Lower-cases the key text
    -  Treats ``__`` (double-underscore) as nested access
    -  Calls ``ast.literal_eval`` on the value
    """
    if env is None:
        env = os.environ
    d = {}
    for name, value in env.items():
        if name.startswith("NAPARI_"):
            varname = name[7:].lower().replace("__", ".")
            try:
                d[varname] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                d[varname] = value

    result: dict = {}
    set(d, config=result, clean=True)

    return result


def ensure_file(source: str, destination: Optional[str] = None, comment=True):
    """Copy file to default location if it does not already exist.

    This tries to move a default configuration file to a default location if if
    does not already exist.  It also comments out that file by default.

    This is to be used by downstream modules that may have default
    configuration files that they wish to include in the default configuration
    path.

    Parameters
    ----------
    source : string, filename
        Source configuration file, typically within a source directory.
    destination : string, directory
        Destination directory. Configurable by ``NAPARI_CONFIG`` environment
        variable, falling back to ~/.config/napari.
    comment : bool, True by default
        Whether or not to comment out the config file when copying.
    """
    if destination is None:
        destination = PATH

    # destination is a file and already exists, never overwrite
    if os.path.isfile(destination):
        return

    # If destination is not an existing file, interpret as a directory,
    # use the source basename as the filename
    directory = destination
    destination = os.path.join(directory, os.path.basename(source))

    try:
        if not os.path.exists(destination):
            os.makedirs(directory, exist_ok=True)

            # Atomically create destination.  Parallel testing discovered
            # a race condition where a process can be busy creating the
            # destination while another process reads an empty config file.
            tmp = "%s.tmp.%d" % (destination, os.getpid())
            with open(source) as f:
                lines = list(f)

            if comment:
                lines = [
                    "# " + line
                    if line.strip() and not line.startswith("#")
                    else line
                    for line in lines
                ]

            with open(tmp, "w") as f:
                f.write("".join(lines))

            try:
                os.rename(tmp, destination)
            except OSError:
                os.remove(tmp)
    except (IOError, OSError):
        pass


class set:
    """Temporarily set configuration values within a context manager.

    Parameters
    ----------
    arg : mapping or None, optional
        A mapping of configuration key-value pairs to set.
    clean : bool
        By default, the key `_dirty` is added to the config dict so that the
        :func:`sync` function knows to update the yaml file on disk.
        If ``clean == True``, will not add the key `_dirty` to the config dict.
    **kwargs :
        Additional key-value pairs to set. If ``arg`` is provided, values set
        in ``arg`` will be applied before those in ``kwargs``.
        Double-underscores (``__``) in keyword arguments will be replaced with
        ``.``, allowing nested values to be easily set.

    Examples
    --------
    >>> import napari

    Set ``'foo.bar'`` in a context, by providing a mapping.

    >>> with napari.config.set({'foo.bar': 123}):
    ...     pass

    Set ``'foo.bar'`` in a context, by providing a keyword argument.

    >>> with napari.config.set(foo__bar=123):
    ...     pass

    Set ``'foo.bar'`` globally.

    >>> napari.config.set(foo__bar=123)

    See Also
    --------
    napari.config.get
    """

    def __init__(
        self,
        arg: Optional[dict] = None,
        config: dict = config,
        lock: threading.Lock = config_lock,
        clean: bool = False,
        **kwargs,
    ):
        with lock:
            self.config = config
            self._record: List[Tuple[str, Tuple[str, ...], Any]] = []

            if arg is not None:
                try:
                    for key, value in arg.items():
                        key = check_deprecations(key)
                        self._assign(key.split("."), value, config)
                except AttributeError:
                    if not isinstance(arg, dict):
                        raise TypeError(
                            "First argument to config.set() must be a dict"
                        )
                    else:
                        raise

            if kwargs:
                for key, value in kwargs.items():
                    key = key.replace("__", ".")
                    key = check_deprecations(key)
                    self._assign(key.split("."), value, config)

        if not clean:
            config['_dirty'] = True

    def __enter__(self) -> dict:
        return self.config

    def __exit__(
        self,
        type: Type[Exception],
        value: Exception,
        traceback: Optional[TracebackType],
    ):
        for op, path, value in reversed(self._record):
            d = self.config
            if op == "replace":
                for key in path[:-1]:
                    d = d.setdefault(key, {})
                d[path[-1]] = value
            else:  # insert
                for key in path[:-1]:
                    try:
                        d = d[key]
                    except KeyError:
                        break
                else:
                    d.pop(path[-1], None)

    def _assign(
        self,
        keys: Sequence[str],
        value: Any,
        d: dict,
        path: Tuple[str, ...] = (),
        record: bool = True,
    ):
        """Assign value into a nested configuration dictionary

        Parameters
        ----------
        keys : Sequence[str]
            The nested path of keys to assign the value.
        value : object
        d : dict
            The part of the nested dictionary into which we want to assign the
            value
        path : Tuple[str], optional
            The path history up to this point.
        record : bool, optional
            Whether this operation needs to be recorded to allow for rollback.
        """
        key = canonical_name(keys[0], d)

        path = path + (key,)

        if len(keys) == 1:
            if record:
                if key in d:
                    self._record.append(("replace", path, d[key]))
                else:
                    self._record.append(("insert", path, None))
            d[key] = value

            # might be worth emitting a warning here if we know the value
            # cannot be serialized
            # yaml.dump(value, Dumper=ConfigDumper)

        else:
            if key not in d:
                if record:
                    self._record.append(("insert", path, None))
                d[key] = {}
                # No need to record subsequent operations after an insert
                record = False
            self._assign(keys[1:], value, d[key], path, record=record)


def collect(
    paths: List = paths, env: Optional[Union[dict, os._Environ]] = None
) -> dict:
    """
    Collect configuration from paths and environment variables

    Parameters
    ----------
    paths : List[str]
        A list of paths to search for yaml config files

    env : dict
        The system environment variables

    Returns
    -------
    config: dict

    See Also
    --------
    napari.config.refresh: collect configuration and update into primary config
    """
    if env is None:
        env = os.environ
    configs = []

    if yaml:
        configs.extend(collect_yaml(paths=paths))

    configs.append(collect_env(env=env))

    return merge(*configs)


def refresh(config: dict = config, defaults: List[dict] = defaults, **kwargs):
    """
    Update configuration by re-reading yaml files and env variables

    This mutates the global napari.config.config, or the config parameter if
    passed in.

    This goes through the following stages:

    1.  Clearing out all old configuration
    2.  Updating from the stored defaults from downstream libraries
        (see update_defaults)
    3.  Updating from yaml files and environment variables

    Note that some functionality only checks configuration once at startup and
    may not change behavior, even if configuration changes.  It is recommended
    to restart your python process if convenient to ensure that new
    configuration changes take place.

    See Also
    --------
    napari.config.collect: for parameters
    napari.config.update_defaults
    """
    config.clear()

    for d in defaults:
        update(config, d, priority="old")

    update(config, collect(**kwargs))


def get(key: str, default=no_default, config: dict = config) -> Any:
    """Get elements from global config.

    Use '.' for nested access

    Examples
    --------
    >>> from napari import config
    >>> config.get('foo')
    {'x': 1, 'y': 2}

    >>> config.get('foo.x')
    1

    >>> config.get('foo.x.y', default=123)
    123

    See Also
    --------
    napari.config.set
    """
    keys = key.split(".")
    result = config
    for k in keys:
        k = canonical_name(k, result)
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def pop(key: str, default=no_default, config: dict = config, clean=False):
    """Pop elements from global config.

    Use '.' for nested access

    By default, the key `_dirty` is added to the config dict so that the
    :func:`sync` function knows to update the yaml file on disk.
    If ``clean == True``, will not add the key `_dirty` to the config dict.

    Examples
    --------
    >>> from napari import config
    >>> config.set({'foo': {'a': 1, 'b': 2}})
    >>> config.pop('foo.b')
    2

    >>> config.pop('foo')
    {'a': 1}

    >>> config.pop('foo.x.y', default=123)
    123

    See Also
    --------
    napari.config.get
    """
    keys = key.split(".")
    result = config
    for i, k in enumerate(keys):
        k = canonical_name(k, result)
        try:
            if i == len(keys) - 1:
                result = result.pop(k)
                if not clean:
                    config['_dirty'] = True
            else:
                result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def rename(aliases: dict, config: dict = config):
    """Rename old keys to new keys.

    This helps migrate older configuration versions over time
    """
    old = []
    new = {}
    for o, n in aliases.items():
        value = get(o, None, config=config)
        if value is not None:
            old.append(o)
            new[n] = value

    for k in old:
        del config[canonical_name(k, config)]  # TODO: support nested keys

    set(new, config=config, clean=True)


def update_defaults(
    new: dict, config: dict = config, defaults: List[dict] = defaults
):
    """Add a new set of defaults to the configuration.

    It does two things:

    1.  Add the defaults to a global collection to be used by refresh later
    2.  Updates the global config with the new configuration
        prioritizing older values over newer ones
    """
    defaults.append(new)
    update(config, new, priority="old")


T = TypeVar("T", Mapping, Iterable, str)


def expand_environment_variables(config: T) -> T:
    """Expand environment variables in a nested config dictionary.

    This function will recursively search through any nested dictionaries
    and/or lists.

    Parameters
    ----------
    config : dict, iterable, or str
        Input object to search for environment variables

    Returns
    -------
    config : same type as input

    Examples
    --------
    >>> expand_environment_variables({'x': [1, 2, '$USER']})
    {'x': [1, 2, 'my-username']}
    """
    if isinstance(config, Mapping):
        return {k: expand_environment_variables(v) for k, v in config.items()}
    elif isinstance(config, str):
        return os.path.expandvars(config)
    elif isinstance(config, (list, tuple, builtins.set)):
        return type(config)([expand_environment_variables(v) for v in config])
    else:
        return config


#: This dict is used to mark deprecated config keys.
#:
#: The keys of ``deprecations`` are deprecated config values, and the values
#: are the new namespace for the key.  This deprecations are checked when new
#: keys are added to the config in set()
deprecations: Dict[str, str] = {}


def check_deprecations(key: str, deprecations: dict = deprecations):
    """Check if the provided value has been renamed or removed.

    Parameters
    ----------
    key : str
        The configuration key to check
    deprecations : Dict[str, str]
        The mapping of aliases

    Examples
    --------
    >>> deprecations = {"old_key": "new_key", "invalid": None}
    >>> check_deprecations("old_key", deprecations=deprecations)
    UserWarning: Configuration key "old_key" has been deprecated. Please use "new_key" instead.

    >>> check_deprecations("invalid", deprecations=deprecations)
    Traceback (most recent call last):
        ...
    ValueError: Configuration value "invalid" has been removed

    >>> check_deprecations("another_key", deprecations=deprecations)
    'another_key'

    Returns
    -------
    new: str
        The proper key, whether the original (if no deprecation) or the aliased
        value
    """
    if key in deprecations:
        new = deprecations[key]
        if new:
            warnings.warn(
                'Configuration key "{}" has been deprecated. '
                'Please use "{}" instead'.format(key, new)
            )
            return new
        else:
            raise ValueError(
                'Configuration value "{}" has been removed'.format(key)
            )
    else:
        return key


class ConfigDumper(yaml.SafeDumper):
    """Dumper that prevents yaml aliases, and logs bad objects without error.
    """

    def ignore_aliases(self, data):
        return True

    def represent_undefined(self, data):
        logging.error("Error serializing object: %r", data)
        return self.represent_str('<unserializeable>')

    def coerce_to_str(self, data) -> yaml.nodes.ScalarNode:
        return self.represent_str(str(data))


ConfigDumper.add_multi_representer(StringEnum, ConfigDumper.coerce_to_str)
ConfigDumper.add_multi_representer(None, ConfigDumper.represent_undefined)


def sync(
    config: dict = config,
    destination: str = None,
    prefer_config=True,
    lock: threading.Lock = config_lock,
) -> bool:
    """Synchronize config with a yaml file on disk.

    This function is intended to be run periodically in the background of an
    event loop.  It looks for a special ``_dirty`` in the config to know
    whether it has changed since the last time this function was called (and
    pops that key when this function runs).  It also looks for a
    ``_last_synced`` key, which should contain a float value corresponding to
    the last time this function was called; if the modification time of the
    destination file is greater than ``_last_synced``, it is assumed that the
    yaml on disk has changed.

    Parameters
    ----------
    config : dict, optional
        The config to sync to disk, by default use the global config
    destination : str, optional
        Filename or directory to sync to, by default will sync to
        ``config.PATH/_session.yaml``
    prefer_config : bool
        In the case of conflict bewteen the disk yaml and the config, this
        argument determines which value will be used. By default (``True``),
        the value from the config dict will override the yaml value.
    lock : , optional
        [description], by default config_lock

    Returns
    -------
    synced : bool
        Whether a sync occurred.
    """
    # resolve destination file
    if destination is None:
        destination = PATH
    if os.path.isdir(destination):
        destination = os.path.join(destination, '_session.yaml')
    assert os.fspath(destination).endswith(
        ".yaml"
    ), "Only YAML is currently supported"

    # check whether the config dict has changed since the last sync (i.e. has
    # the key "_dirty") or whether the file on disk has been modified since
    # last sync (i.e. has modified date greater than the key "_last_synced")
    config_is_dirty = config.pop('_dirty', None)
    dest_is_dirty = False
    if os.path.exists(destination):
        last_modified = os.path.getmtime(destination)
        # if it has never been synced before, set _last_synced to 0 (= "never")
        last_synced = config.get('_last_synced', 0)
        if last_modified > last_synced:
            dest_is_dirty = True
    else:
        config_is_dirty = True

    # if nothing has happened, return
    if not (config_is_dirty or dest_is_dirty):
        return False

    with lock:  # aquire file lock on yaml file
        if dest_is_dirty:  # yaml file has changed since last sync
            with open(destination, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            yaml_config = {}
        if config_is_dirty:  # config has changed since last sync
            # merge prefers values in the later arguments, over the earlier
            # so if new keys have shown up on disk, we use them, otherwise
            # we overwrite the disk values with the current config
            priority = "old" if prefer_config else "new"
            update(config, yaml_config, priority=priority)
        elif yaml_config:
            # if config is not dirty, but the outputfile IS dirty, we overwrite
            # the the config with the new values from disk.
            config.clear()
            update(config, yaml_config)

        # write the merged config to disk
        with open(destination, 'w') as f:
            out = config.copy()
            out.pop('_last_synced', None)
            if out:
                try:
                    yaml.dump(out, f, Dumper=ConfigDumper)
                except yaml.YAMLError as exc:
                    msg = f"Failed to write session config to disk: {exc}"
                    raise type(exc)(msg)

            else:  # instead of writing "{}" to file, write "# empty"
                f.write("# empty")

    # update the config with the time of last modification
    config['_last_synced'] = os.path.getmtime(destination)
    return True


refresh()
set(_last_synced=time.time(), clean=True)


if yaml:
    # read in the default settings from this directory
    fn = os.path.join(os.path.dirname(__file__), "napari.yaml")
    ensure_file(source=fn)

    with open(fn) as f:
        _defaults = yaml.safe_load(f) or {}

    update_defaults(_defaults)
    del fn, _defaults
