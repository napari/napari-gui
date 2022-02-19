import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, validator

from ..utils._base import _DEFAULT_CONFIG_PATH
from ..utils.translations import trans
from ._appearance import AppearanceSettings
from ._application import ApplicationSettings
from ._base import EventedConfigFileSettings, _remove_empty_dicts
from ._experimental import ExperimentalSettings
from ._fields import Version
from ._plugins import PluginsSettings
from ._shortcuts import ShortcutsSettings

_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)

CURRENT_SCHEMA_VERSION = Version(0, 4, 0)


class NapariSettings(EventedConfigFileSettings):
    """Schema for napari settings."""

    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Version = Field(
        CURRENT_SCHEMA_VERSION,
        description=trans._("Napari settings schema version."),
    )

    @validator('schema_version', pre=True)
    def _handle_empty_schema(cls, value):
        return '0.3.0' if not value else value

    application: ApplicationSettings = Field(
        default_factory=ApplicationSettings,
        title=trans._("Application"),
        description=trans._("Main application settings."),
    )
    appearance: AppearanceSettings = Field(
        default_factory=AppearanceSettings,
        title=trans._("Appearance"),
        description=trans._("User interface appearance settings."),
    )
    plugins: PluginsSettings = Field(
        default_factory=PluginsSettings,
        title=trans._("Plugins"),
        description=trans._("Plugins settings."),
    )
    shortcuts: ShortcutsSettings = Field(
        default_factory=ShortcutsSettings,
        title=trans._("Shortcuts"),
        description=trans._("Shortcut settings."),
    )
    experimental: ExperimentalSettings = Field(
        default_factory=ExperimentalSettings,
        title=trans._("Experimental"),
        description=trans._("Experimental settings."),
    )

    # private attributes and ClassVars will not appear in the schema
    _config_path: Optional[Path] = Path(_CFG_PATH) if _CFG_PATH else None

    class Config:
        env_prefix = 'napari_'
        use_enum_values = False
        # all of these fields are evented models, so we don't want to break
        # connections by setting the top-level field itself
        # (you can still mutate attributes in the subfields)
        allow_mutation = False

    def __init__(self, config_path=..., **values: Any) -> None:
        super().__init__(config_path, **values)
        self._maybe_migrate()

    def _save_dict(self):
        # TODO: there must be a better way to always include this
        return {'schema_version': self.schema_version, **super()._save_dict()}

    def __str__(self):
        out = 'NapariSettings (defaults excluded)\n' + 34 * '-' + '\n'
        data = self.dict(exclude_defaults=True)
        out += self._yaml_dump(_remove_empty_dicts(data))
        return out

    def __repr__(self):
        return str(self)

    def _maybe_migrate(self):
        if self.schema_version < CURRENT_SCHEMA_VERSION:
            for migration in Migration.subclasses():
                if self.schema_version == migration.from_:
                    with mutation_allowed(self):
                        migration().migrate(self)
                        self.schema_version = Version.parse(migration.to_)


@contextmanager
def mutation_allowed(obj: BaseModel):
    config = obj.__config__
    prev, config.allow_mutation = config.allow_mutation, True
    try:
        yield
    finally:
        config.allow_mutation = prev


class Migration:
    from_: str
    to_: str

    def migrate(self, model: NapariSettings) -> None:
        ...

    @classmethod
    def subclasses(cls):
        yield from sorted(cls.__subclasses__(), key=lambda x: x.from_)


class Migrate_030_040(Migration):
    from_ = '0.3.0'
    to_ = '0.4.0'

    def migrate(self, model: NapariSettings):
        from importlib.metadata import distributions

        # prior to v0.4.0, npe2 plugins were automatically added to
        # disabled plugins
        for dist in distributions():
            for ep in dist.entry_points:
                if ep.group == "napari.manifest":
                    model.plugins.disabled_plugins.discard(
                        dist.metadata['Name']
                    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        dest = Path(sys.argv[2]).expanduser().absolute()
    else:
        dest = Path(__file__).parent / 'napari.schema.json'
    dest.write_text(NapariSettings.schema_json())
