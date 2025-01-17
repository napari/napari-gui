from napari._pydantic_compat import Field
from napari.settings._base import EventedSettings
from napari.utils.translations import trans


# this class inherits from EventedSettings instead of EventedModel because
# it uses Field(env=...) for one of its attributes
class ExperimentalSettings(EventedSettings):
    async_: bool = Field(
        False,
        title=trans._('Render Images Asynchronously'),
        description=trans._(
            'Asynchronous loading of image data. \nThis setting partially loads data while viewing.'
        ),
        env='napari_async',
        requires_restart=False,
    )
    autoswap_buffers: bool = Field(
        False,
        title=trans._('Enable autoswapping rendering buffers.'),
        description=trans._(
            'Autoswapping rendering buffers improves quality by reducing tearing artifacts, while sacrificing some performance.'
        ),
        env='napari_autoswap',
        requires_restart=True,
    )

    rdp_epsilon: float = Field(
        0.5,
        title=trans._('Shapes polygon lasso and path RDP epsilon'),
        description=trans._(
            'Setting this higher removes more points from polygons or paths. \nSetting this to 0 keeps all vertices of '
            'a given polygon or path.'
        ),
        type=float,
        ge=0,
    )

    lasso_vertex_distance: int = Field(
        10,
        title=trans._(
            'Minimum distance threshold of shapes lasso and path tool'
        ),
        description=trans._(
            'Value determines how many screen pixels one has to move before another vertex can be added to the polygon'
            'or path.'
        ),
        type=int,
        gt=0,
        lt=50,
    )
    scale_bar_length: int = Field(
        5,
        title=trans._('Scale bar length'),
        description=trans._(
            'The scale bar length as a fraction of the canvas width.'
        ),
        type=int,
        ge=3,
        le=10,
    )
    scale_bar_font_size: int = Field(
        35,
        title=trans._('Scale bar font size'),
        description=trans._(
            'The scale bar font size as a fraction of the canvas height.'
        ),
        type=int,
        ge=30,
        le=50,
    )

    completion_radius: int = Field(
        default=-1,
        title=trans._(
            'Double-click Labels polygon completion radius (-1 to always complete)'
        ),
        description=trans._(
            'Max radius in pixels from first vertex for double-click to complete a polygon; set -1 to always complete.'
        ),
    )

    compiled_triangulation: bool = Field(
        False,
        title=trans._(
            'Use C++ code to speedup creation and updates of Shapes layers'
            '(requires optional dependencies)'
        ),
        description=trans._(
            'When enabled, triangulation (breaking down polygons into '
            "triangles that can be displayed by napari's graphics engine) is "
            'sped up by using C++ code from the optional library '
            'PartSegCore-compiled-backend. C++ code can cause bad crashes '
            'called segmentation faults or access violations. If you '
            'encounter such a crash while using this option please report '
            'it at https://github.com/napari/napari/issues.'
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)
