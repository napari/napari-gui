from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
from warnings import warn

from napari._pydantic_compat import (
    BaseModel,
    BaseSettings,
    SettingsError,
    ValidationError,
    display_errors,
)
from napari.settings._yaml import PydanticYamlMixin
from napari.utils.events import EmitterGroup, EventedModel
from napari.utils.misc import deep_update
from napari.utils.translations import trans

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from typing import Any, Union

    from napari._pydantic_compat import (
        EnvSettingsSource,
        SettingsSourceCallable,
    )
    from napari.utils.events import Event

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]

Dict = dict  # rename, because EventedSettings has method dict


class EventedSettings(BaseSettings, EventedModel):
    """A variant of EventedModel designed for settings.

    Pydantic's BaseSettings model will attempt to determine the values of any
    fields not passed as keyword arguments by reading from the environment.
    """

    # provide config_path=None to prevent reading from disk.

    class Config(EventedModel.Config):
        pass

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        self.events.add(changed=None)

        # re-emit subfield
        for name, field in self.__fields__.items():
            attr = getattr(self, name)
            if isinstance(getattr(attr, 'events', None), EmitterGroup):
                attr.events.connect(partial(self._on_sub_event, field=name))

            if field.field_info.extra.get('requires_restart'):
                emitter = getattr(self.events, name)

                @emitter.connect
                def _warn_restart(*_):
                    warn(
                        trans._(
                            'Restart required for this change to take effect.',
                            deferred=True,
                        )
                    )

    def _on_sub_event(self, event: Event, field=None):
        """emit the field.attr name and new value"""
        if field:
            field += '.'
        value = getattr(event, 'value', None)
        self.events.changed(key=f'{field}{event._type}', value=value)


_NOT_SET = object()


class EventedConfigFileSettings(EventedSettings, PydanticYamlMixin):
    """This adds config read/write and yaml support to EventedSettings.

    If your settings class *only* needs to read variables from the environment,
    such as environment variables (but not a config file), then subclass from
    EventedSettings.
    """

    _config_path: Path | None = None
    _save_on_change: bool = True
    # this dict stores the data that came specifically from the config file.
    # it's populated in `config_file_settings_source` and
    # used in `_remove_env_settings`
    _config_file_settings: dict

    # provide config_path=None to prevent reading from disk.
    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        _cfg = (
            config_path
            if config_path is not _NOT_SET
            else self.__private_attributes__['_config_path'].get_default()
        )
        # this line is here for usage in the `customise_sources` hook.  It
        # will be overwritten in __init__ by BaseModel._init_private_attributes
        # so we set it again after __init__.
        self._config_path = _cfg
        super().__init__(**values)
        self._config_path = _cfg

    def _maybe_save(self):
        if self._save_on_change and self.config_path:
            self.save()

    def _on_sub_event(self, event, field=None):
        super()._on_sub_event(event, field)
        self._maybe_save()

    @property
    def config_path(self):
        """Return the path to/from which settings be saved/loaded."""
        return self._config_path

    def dict(
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny = None,  # type: ignore
        exclude: AbstractSetIntStr | MappingIntStrAny = None,  # type: ignore
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
    ) -> DictStrAny:
        """Return dict representation of the model.

        May optionally specify which fields to include or exclude.
        """
        data = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        if exclude_env:
            self._remove_env_settings(data)
        return data

    def _save_dict(self, **dict_kwargs: Any) -> DictStrAny:
        """The minimal dict representation that will be persisted to disk.

        By default, this will exclude settings values that match the default
        value, and will exclude values that were provided by environment
        variables.  Empty dicts will also be removed.
        """
        dict_kwargs.setdefault('exclude_defaults', True)
        dict_kwargs.setdefault('exclude_env', True)
        data = self.dict(**dict_kwargs)
        _remove_empty_dicts(data)
        return data

    def save(self, path: str | Path | None = None, **dict_kwargs):
        """Save current settings to path.

        By default, this will exclude settings values that match the default
        value, and will exclude values that were provided by environment
        variables.  (see `_save_dict` method.)
        """
        path = path or self.config_path
        if not path:
            raise ValueError(
                trans._(
                    'No path provided in config or save argument.',
                    deferred=True,
                )
            )

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(exist_ok=True, parents=True)
        self._dump(str(path), self._save_dict(**dict_kwargs))

    def _dump(self, path: str, data: Dict) -> None:
        """Encode and dump `data` to `path` using a path-appropriate encoder."""
        if str(path).endswith(('.yaml', '.yml')):
            _data = self._yaml_dump(data)
        elif str(path).endswith('.json'):
            json_dumps = self.__config__.json_dumps
            _data = json_dumps(data, default=self.__json_encoder__)
        else:
            raise NotImplementedError(
                trans._(
                    'Can only currently dump to `.json` or `.yaml`, not {path!r}',
                    deferred=True,
                    path=path,
                )
            )
        with open(path, 'w') as target:
            target.write(_data)

    def env_settings(self) -> Dict[str, Any]:
        """Get a dict of fields that were provided as environment vars."""
        env_settings = getattr(self.__config__, '_env_settings', {})
        if callable(env_settings):
            env_settings = env_settings(self)
        return env_settings

    def _remove_env_settings(self, data):
        """Remove key:values from `data` that match settings from env vars.

        This is handy when we want to persist settings to disk without
        including settings that were provided by environment variables (which
        are usually more temporary).
        """
        env_data = self.env_settings()
        if env_data:
            _restore_config_data(
                data, env_data, getattr(self, '_config_file_settings', {})
            )

    class Config:
        # If True: validation errors in a config file will raise an exception
        # otherwise they will warn to the logger
        strict_config_check: bool = False
        sources: Sequence[str] = []
        _env_settings: SettingsSourceCallable

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: EnvSettingsSource,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """customise the way data is loaded.

            This does 2 things:
            1) adds the `config_file_settings_source` to the sources, which
               will load data from `settings._config_path` if it exists.
            2) adds support for nested env_vars, such that if a model with an
               env_prefix of "foo_" has a field named `bar`, then you can use
               `FOO_BAR_X=1` to set the x attribute in `foo.bar`.

            Priority is given to sources earlier in the list.  You can resort
            the return list to change the priority of sources.
            """
            cls._env_settings = nested_env_settings(env_settings)
            return (  # type: ignore[return-value]
                init_settings,
                cls._env_settings,
                cls._config_file_settings_source,
                file_secret_settings,
            )
            # Even when EventedConfigFileSettings is a subclass of BaseSettings,
            # mypy do not see this

        @classmethod
        def _config_file_settings_source(
            cls, settings: EventedConfigFileSettings
        ) -> dict[str, Any]:
            return config_file_settings_source(settings)


# Utility functions


def nested_env_settings(
    super_eset: EnvSettingsSource,
) -> SettingsSourceCallable:
    """Wraps the pydantic EnvSettingsSource to support nested env vars.

    currently only supports one level of nesting.

    Examples
    --------
    `NAPARI_APPEARANCE_THEME=light`
    will parse to:
    {'appearance': {'theme': 'light'}}

    If a submodel has a field that explicitly declares an `env`... that will
    also be found.  For example, 'ExperimentalSettings.async_' directly
    declares `env='napari_async'`... so NAPARI_ASYNC is accessible without
    nesting as well.
    """

    def _inner(settings: BaseSettings) -> dict[str, Any]:
        # first call the original implementation
        d = super_eset(settings)

        if settings.__config__.case_sensitive:
            env_vars: Mapping[str, str | None] = os.environ
        else:
            env_vars = {k.lower(): v for k, v in os.environ.items()}

        # now iterate through all subfields looking for nested env vars
        # For example:
        # NapariSettings has a Config.env_prefix of 'napari_'
        # so every field in the NapariSettings.Application subfield will be
        # available at 'napari_application_fieldname'
        for field in settings.__fields__.values():
            if not isinstance(field.type_, type(BaseModel)):
                continue  # pragma: no cover
            field_type = cast(BaseModel, field.type_)
            for env_name in field.field_info.extra['env_names']:
                for subf in field_type.__fields__.values():
                    # first check if subfield directly declares an "env"
                    # (for example: ExperimentalSettings.async_)
                    for e in subf.field_info.extra.get('env_names', []):
                        env_val = env_vars.get(e.lower())
                        if env_val is not None:
                            break
                    # otherwise, look for the standard nested env var
                    else:
                        env_val = env_vars.get(f'{env_name}_{subf.name}')

                    is_complex, all_json_fail = super_eset.field_is_complex(
                        subf
                    )
                    if env_val is not None and is_complex:
                        try:
                            env_val = settings.__config__.json_loads(env_val)
                        except ValueError as e:
                            if not all_json_fail:
                                msg = trans._(
                                    'error parsing JSON for "{env_name}"',
                                    deferred=True,
                                    env_name=env_name,
                                )
                                raise SettingsError(msg) from e

                        if isinstance(env_val, dict):
                            explode = super_eset.explode_env_vars(
                                field, env_vars
                            )
                            env_val = deep_update(env_val, explode)

                    # if we found an env var, store it and return it
                    if env_val is not None:
                        if field.alias not in d:
                            d[field.alias] = {}
                        d[field.alias][subf.name] = env_val
        return d

    return _inner


def config_file_settings_source(
    settings: EventedConfigFileSettings,
) -> dict[str, Any]:
    """Read config files during init of an EventedConfigFileSettings obj.

    The two important values are the `settings._config_path`
    attribute, which is the main config file (if present), and
    `settings.__config__.source`, which is an optional list of additional files
    to read. (files later in the list take precedence and `_config_path` takes
    precedence over all)

    Parameters
    ----------
    settings : EventedConfigFileSettings
        The new model instance (not fully instantiated)

    Returns
    -------
    dict
        *validated* values for the model.
    """
    # _config_path is the primary config file on the model (the one to save to)
    config_path = getattr(settings, '_config_path', None)

    default_cfg = type(settings).__private_attributes__.get('_config_path')
    default_cfg = getattr(default_cfg, 'default', None)

    # if the config has a `sources` list, read those too and merge.
    sources: list[str] = list(getattr(settings.__config__, 'sources', []))
    if config_path:
        sources.append(config_path)
    if not sources:
        return {}

    data: dict = {}
    for path in sources:
        if not path:
            continue  # pragma: no cover
        path_ = Path(path).expanduser().resolve()

        # if the requested config path does not exist, move on to the next
        if not path_.is_file():
            # if it wasn't the `_config_path` stated in the BaseModel itself,
            # we warn, since this would have been user provided.
            if path_ != default_cfg:
                _logger.warning(
                    trans._(
                        'Requested config path is not a file: {path}',
                        path=path_,
                    )
                )
            continue

        # get loader for yaml/json
        if str(path).endswith(('.yaml', '.yml')):
            load = __import__('yaml').safe_load
        elif str(path).endswith('.json'):
            load = __import__('json').load
        else:
            warn(
                trans._(
                    'Unrecognized file extension for config_path: {path}',
                    path=path,
                )
            )
            continue

        try:
            # try to parse the config file into a dict
            new_data = load(path_.read_text()) or {}
        except Exception as err:  # noqa: BLE001
            _logger.warning(
                trans._(
                    'The content of the napari settings file could not be read\n\nThe default settings will be used and the content of the file will be replaced the next time settings are changed.\n\nError:\n{err}',
                    deferred=True,
                    err=err,
                )
            )
            continue
        assert isinstance(new_data, dict), path_.read_text()
        deep_update(data, new_data, copy=False)

    try:
        # validate the data, passing config_path=None so we dont recurse
        # back to this point again.
        type(settings)(config_path=None, **data)
    except ValidationError as err:
        if getattr(settings.__config__, 'strict_config_check', False):
            raise

        # if errors occur, we still want to boot, so we just remove bad keys
        errors = err.errors()
        msg = trans._(
            'Validation errors in config file(s).\nThe following fields have been reset to the default value:\n\n{errors}\n',
            deferred=True,
            errors=display_errors(errors),
        )
        with contextlib.suppress(Exception):
            # we're about to nuke some settings, so just in case... try backup
            backup_path = path_.parent / f'{path_.stem}.BAK{path_.suffix}'
            backup_path.write_text(path_.read_text())

        _logger.warning(msg)
        try:
            _remove_bad_keys(data, [e.get('loc', ()) for e in errors])
        except KeyError:  # pragma: no cover
            _logger.warning(
                trans._(
                    'Failed to remove validation errors from config file. Using defaults.'
                )
            )
            data = {}
    # store data at this state for potential later recovery
    settings._config_file_settings = data
    return data


def _remove_bad_keys(data: dict, keys: list[tuple[int | str, ...]]):
    """Remove list of keys (as string tuples) from dict (in place).

    Parameters
    ----------
    data : dict
        dict to modify (will be modified inplace)
    keys : List[Tuple[str, ...]]
        list of possibly nested keys

    Examples
    --------

    >>> data = {'a': 1, 'b' : {'c': 2, 'd': 3}, 'e': 4}
    >>> keys = [('b', 'd'), ('e',)]
    >>> _remove_bad_keys(data, keys)
    >>> data
    {'a': 1, 'b': {'c': 2}}

    """
    for key in keys:
        if not key:
            continue  # pragma: no cover
        d = data
        while True:
            base, *key = key  # type: ignore
            if not key:
                break
            # since no pydantic fields will be integers, integers usually
            # mean we're indexing into a typed list. So remove the base key
            if isinstance(key[0], int):
                break
            d = d[base]
        del d[base]


def _restore_config_data(dct: dict, delete: dict, defaults: dict) -> dict:
    """delete nested dict keys, restore from defaults."""
    for k, v in delete.items():
        # restore from defaults if present, or just delete the key
        if k in dct:
            if k in defaults:
                dct[k] = defaults[k]
            else:
                del dct[k]
        # recurse
        elif isinstance(v, dict):
            dflt = defaults.get(k)
            if not isinstance(dflt, dict):
                dflt = {}
            _restore_config_data(dct[k], v, dflt)
    return dct


def _remove_empty_dicts(dct: dict, recurse=True) -> dict:
    """Remove all (nested) keys with empty dict values from `dct`"""
    for k, v in list(dct.items()):
        if isinstance(v, Mapping) and recurse:
            _remove_empty_dicts(dct[k])
        if v == {}:
            del dct[k]
    return dct
