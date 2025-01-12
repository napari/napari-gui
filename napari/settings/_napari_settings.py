import os
from pathlib import Path
from typing import Any

from napari._pydantic_compat import Field
from napari.settings._appearance import AppearanceSettings
from napari.settings._application import ApplicationSettings
from napari.settings._base import (
    _NOT_SET,
    EventedConfigFileSettings,
    _remove_empty_dicts,
)
from napari.settings._experimental import ExperimentalSettings
from napari.settings._fields import Version
from napari.settings._plugins import PluginsSettings
from napari.settings._shortcuts import ShortcutsSettings
from napari.utils._base import _DEFAULT_CONFIG_PATH
from napari.utils.translations import trans

_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)

CURRENT_SCHEMA_VERSION = Version(0, 6, 0)


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
        description=trans._('Napari settings schema version.'),
    )

    application: ApplicationSettings = Field(
        default_factory=ApplicationSettings,
        title=trans._('Application'),
        description=trans._('Main application settings.'),
    )
    appearance: AppearanceSettings = Field(
        default_factory=AppearanceSettings,
        title=trans._('Appearance'),
        description=trans._('User interface appearance settings.'),
        allow_mutation=False,
    )
    plugins: PluginsSettings = Field(
        default_factory=PluginsSettings,
        title=trans._('Plugins'),
        description=trans._('Plugins settings.'),
        allow_mutation=False,
    )
    shortcuts: ShortcutsSettings = Field(
        default_factory=ShortcutsSettings,
        title=trans._('Shortcuts'),
        description=trans._('Shortcut settings.'),
        allow_mutation=False,
    )
    experimental: ExperimentalSettings = Field(
        default_factory=ExperimentalSettings,
        title=trans._('Experimental'),
        description=trans._('Experimental settings.'),
        allow_mutation=False,
    )

    # private attributes and ClassVars will not appear in the schema
    _config_path: Path | None = Path(_CFG_PATH) if _CFG_PATH else None

    class Config(EventedConfigFileSettings.Config):
        env_prefix = 'napari_'
        use_enum_values = False
        # all of these fields are evented models, so we don't want to break
        # connections by setting the top-level field itself
        # (you can still mutate attributes in the subfields)

        @classmethod
        def _config_file_settings_source(cls, settings) -> dict:
            # before '0.4.0' we didn't write the schema_version in the file
            # written to disk. so if it's missing, add schema_version of 0.3.0
            d = super()._config_file_settings_source(settings)
            d.setdefault('schema_version', '0.3.0')
            return d

    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        super().__init__(config_path, **values)
        self._maybe_migrate()

    def _save_dict(self, **kwargs):
        # we always want schema_version written to the settings.yaml
        # TODO: is there a better way to always include schema version?
        return {
            'schema_version': self.schema_version,
            **super()._save_dict(**kwargs),
        }

    def __str__(self):
        out = 'NapariSettings (defaults excluded)\n' + 34 * '-' + '\n'
        data = self.dict(exclude_defaults=True)
        out += self._yaml_dump(_remove_empty_dicts(data))
        return out

    def __repr__(self):
        return str(self)

    def _maybe_migrate(self):
        if self.schema_version < CURRENT_SCHEMA_VERSION:
            from napari.settings._migrations import do_migrations

            do_migrations(self)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        dest = Path(sys.argv[2]).expanduser().absolute()
    else:
        dest = Path(__file__).parent / 'napari.schema.json'
    dest.write_text(NapariSettings.schema_json())
