from __future__ import annotations

from psutil import virtual_memory

from napari._pydantic_compat import Field, validator
from napari.settings._constants import (
    BrushSizeOnMouseModifiers,
    LabelDTypes,
    LoopMode,
)
from napari.settings._fields import Language
from napari.utils._base import _DEFAULT_LOCALE
from napari.utils.events.custom_types import conint
from napari.utils.events.evented_model import EventedModel
from napari.utils.notifications import NotificationSeverity
from napari.utils.translations import trans

GridStride = conint(ge=-50, le=50, ne=0)
GridWidth = conint(ge=-1, ne=0)
GridHeight = conint(ge=-1, ne=0)

_DEFAULT_MEM_FRACTION = 0.25
MAX_CACHE = virtual_memory().total * 0.5 / 1e9


class DaskSettings(EventedModel):
    enabled: bool = True
    cache: float = Field(
        virtual_memory().total * _DEFAULT_MEM_FRACTION / 1e9,
        ge=0,
        le=MAX_CACHE,
        title='Cache size (GB)',
    )


class ApplicationSettings(EventedModel):
    first_time: bool = Field(
        True,
        title=trans._('First time'),
        description=trans._(
            'Indicate if napari is running for the first time. This setting is managed by the application.'
        ),
    )
    ipy_interactive: bool = Field(
        True,
        title=trans._('IPython interactive'),
        description=trans._(
            r'Toggle the use of interactive `%gui qt` event loop when creating napari Viewers in IPython.'
        ),
    )
    language: Language = Field(
        Language(_DEFAULT_LOCALE),
        title=trans._('Language'),
        description=trans._(
            'Select the display language for the user interface.'
        ),
    )
    # Window state, geometry and position
    save_window_geometry: bool = Field(
        True,
        title=trans._('Save window geometry'),
        description=trans._(
            'Toggle saving the main window size and position.'
        ),
    )
    save_window_state: bool = Field(
        False,  # changed from True to False in schema v0.2.1
        title=trans._('Save window state'),
        description=trans._('Toggle saving the main window state of widgets.'),
    )
    window_position: tuple[int, int] | None = Field(
        None,
        title=trans._('Window position'),
        description=trans._(
            'Last saved x and y coordinates for the main window. This setting is managed by the application.'
        ),
    )
    window_size: tuple[int, int] | None = Field(
        None,
        title=trans._('Window size'),
        description=trans._(
            'Last saved width and height for the main window. This setting is managed by the application.'
        ),
    )
    window_maximized: bool = Field(
        False,
        title=trans._('Window maximized state'),
        description=trans._(
            'Last saved maximized state for the main window. This setting is managed by the application.'
        ),
    )
    window_fullscreen: bool = Field(
        False,
        title=trans._('Window fullscreen'),
        description=trans._(
            'Last saved fullscreen state for the main window. This setting is managed by the application.'
        ),
    )
    window_state: str | None = Field(
        None,
        title=trans._('Window state'),
        description=trans._(
            'Last saved state of dockwidgets and toolbars for the main window. This setting is managed by the application.'
        ),
    )
    window_statusbar: bool = Field(
        True,
        title=trans._('Show status bar'),
        description=trans._(
            'Toggle diplaying the status bar for the main window.'
        ),
    )
    preferences_size: tuple[int, int] | None = Field(
        None,
        title=trans._('Preferences size'),
        description=trans._(
            'Last saved width and height for the preferences dialog. This setting is managed by the application.'
        ),
    )
    gui_notification_level: NotificationSeverity = Field(
        NotificationSeverity.INFO,
        title=trans._('GUI notification level'),
        description=trans._(
            'Select the notification level for the user interface.'
        ),
    )
    console_notification_level: NotificationSeverity = Field(
        NotificationSeverity.NONE,
        title=trans._('Console notification level'),
        description=trans._('Select the notification level for the console.'),
    )
    open_history: list[str] = Field(
        [],
        title=trans._('Opened folders history'),
        description=trans._(
            'Last saved list of opened folders. This setting is managed by the application.'
        ),
    )
    save_history: list[str] = Field(
        [],
        title=trans._('Saved folders history'),
        description=trans._(
            'Last saved list of saved folders. This setting is managed by the application.'
        ),
    )
    playback_fps: int = Field(
        10,
        title=trans._('Playback frames per second'),
        description=trans._('Playback speed in frames per second.'),
    )
    playback_mode: LoopMode = Field(
        LoopMode.LOOP,
        title=trans._('Playback loop mode'),
        description=trans._('Loop mode for playback.'),
    )

    grid_stride: GridStride = Field(  # type: ignore [valid-type]
        default=1,
        title=trans._('Grid Stride'),
        description=trans._('Number of layers to place in each grid square.'),
    )

    grid_width: GridWidth = Field(  # type: ignore [valid-type]
        default=-1,
        title=trans._('Grid Width'),
        description=trans._('Number of columns in the grid.'),
    )

    grid_height: GridHeight = Field(  # type: ignore [valid-type]
        default=-1,
        title=trans._('Grid Height'),
        description=trans._('Number of rows in the grid.'),
    )
    confirm_close_window: bool = Field(
        default=True,
        title=trans._('Confirm window or application closing'),
        description=trans._(
            'Ask for confirmation before closing a napari window or application (all napari windows).',
        ),
    )
    hold_button_delay: float = Field(
        default=0.5,
        title=trans._('Delay to treat button as hold in seconds'),
        description=trans._(
            'This affects certain actions where a short press and a long press have different behaviors, such as changing the mode of a layer permanently or only during the long press.'
        ),
    )

    brush_size_on_mouse_move_modifiers: BrushSizeOnMouseModifiers = Field(
        BrushSizeOnMouseModifiers.ALT,
        title=trans._('Brush size on mouse move modifiers'),
        description=trans._(
            'Modifiers to activate changing the brush size by moving the mouse.'
        ),
    )

    # convert cache (and max cache) from bytes to mb for widget
    dask: DaskSettings = Field(
        default=DaskSettings(),
        title=trans._('Dask cache'),
        description=trans._(
            'Settings for dask cache (does not work with distributed arrays)'
        ),
    )

    new_labels_dtype: LabelDTypes = Field(
        default=LabelDTypes.uint8,
        title=trans._('New labels data type'),
        description=trans._(
            'data type for labels layers created with the "new labels" button.'
        ),
    )

    plugin_widget_positions: dict[str, str] = Field(
        default={},
        title=trans._('Plugin widget positions'),
        description=trans._(
            'Per-widget last saved position of plugin dock widgets. This setting is managed by the application.'
        ),
    )

    @validator('window_state', allow_reuse=True)
    def _validate_qbtye(cls, v: str) -> str:
        if v and (not isinstance(v, str) or not v.startswith('!QBYTE_')):
            raise ValueError(
                trans._("QByte strings must start with '!QBYTE_'")
            )
        return v

    class Config:
        use_enum_values = False  # https://github.com/napari/napari/issues/3062

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = (
            'schema_version',
            'preferences_size',
            'first_time',
            'window_position',
            'window_size',
            'window_maximized',
            'window_fullscreen',
            'window_state',
            'window_statusbar',
            'open_history',
            'save_history',
            'ipy_interactive',
            'plugin_widget_positions',
        )
