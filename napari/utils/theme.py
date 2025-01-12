# syntax_style for the console must be one of the supported styles from
# pygments - see here for examples https://help.farbox.com/pygments.html
import logging
import re
import sys
import warnings
from ast import literal_eval
from contextlib import suppress
from typing import Any, Literal, overload

import npe2

from napari._pydantic_compat import Color, validator
from napari.resources._icons import (
    PLUGIN_FILE_NAME,
    _theme_path,
    build_theme_svgs,
)
from napari.utils.events import EventedModel
from napari.utils.events.containers._evented_dict import EventedDict
from napari.utils.translations import trans

try:
    from qtpy import QT_VERSION

    major, minor, *_ = QT_VERSION.split('.')  # type: ignore[attr-defined]
    use_gradients = (int(major) >= 5) and (int(minor) >= 12)
    del major, minor, QT_VERSION
except (ImportError, RuntimeError):
    use_gradients = False


class Theme(EventedModel):
    """Theme model.

    Attributes
    ----------
    id : str
        id of the theme and name of the virtual folder where icons
        will be saved to.
    label : str
        Name of the theme as it should be shown in the ui.
    syntax_style : str
        Name of the console style.
        See for more details: https://pygments.org/docs/styles/
    canvas : Color
        Background color of the canvas.
    background : Color
        Color of the application background.
    foreground : Color
        Color to contrast with the background.
    primary : Color
        Color used to make part of a widget more visible.
    secondary : Color
        Alternative color used to make part of a widget more visible.
    highlight : Color
        Color used to highlight visual element.
    text : Color
        Color used to display text.
    warning : Color
        Color used to indicate something needs attention.
    error : Color
        Color used to indicate something is wrong or could stop functionality.
    current : Color
        Color used to highlight Qt widget.
    font_size : str
        Font size (in points, pt) used in the application.
    """

    id: str
    label: str
    syntax_style: str
    canvas: Color
    console: Color
    background: Color
    foreground: Color
    primary: Color
    secondary: Color
    highlight: Color
    text: Color
    icon: Color
    warning: Color
    error: Color
    current: Color
    font_size: str = '12pt' if sys.platform == 'darwin' else '9pt'

    @validator('syntax_style', pre=True, allow_reuse=True)
    def _ensure_syntax_style(cls, value: str) -> str:
        from pygments.styles import STYLE_MAP

        assert value in STYLE_MAP, trans._(
            'Incorrect `syntax_style` value: {value} provided. Please use one of the following: {syntax_style}',
            deferred=True,
            syntax_style=f" {', '.join(STYLE_MAP)}",
            value=value,
        )
        return value

    @validator('font_size', pre=True)
    def _ensure_font_size(cls, value: str) -> str:
        assert value.endswith('pt'), trans._(
            'Font size must be in points (pt).', deferred=True
        )
        assert int(value[:-2]) > 0, trans._(
            'Font size must be greater than 0.', deferred=True
        )
        return value

    def to_rgb_dict(self) -> dict[str, Any]:
        """
        This differs from baseclass `dict()` by converting colors to rgb.
        """
        th = super().dict()
        return {
            k: v if not isinstance(v, Color) else v.as_rgb()
            for (k, v) in th.items()
        }


increase_pattern = re.compile(r'{{\s?increase\((\w+),?\s?([-\d]+)?\)\s?}}')
decrease_pattern = re.compile(r'{{\s?decrease\((\w+),?\s?([-\d]+)?\)\s?}}')
gradient_pattern = re.compile(r'([vh])gradient\((.+)\)')
darken_pattern = re.compile(r'{{\s?darken\((\w+),?\s?([-\d]+)?\)\s?}}')
lighten_pattern = re.compile(r'{{\s?lighten\((\w+),?\s?([-\d]+)?\)\s?}}')
opacity_pattern = re.compile(r'{{\s?opacity\((\w+),?\s?([-\d]+)?\)\s?}}')


def decrease(font_size: str, pt: int) -> str:
    """Decrease fontsize."""
    return f'{int(font_size[:-2]) - int(pt)}pt'


def increase(font_size: str, pt: int) -> str:
    """Increase fontsize."""
    return f'{int(font_size[:-2]) + int(pt)}pt'


def _parse_color_as_rgb(color: str | Color) -> tuple[int, int, int]:
    if isinstance(color, str):
        if color.startswith('rgb('):
            return literal_eval(color.lstrip('rgb(').rstrip(')'))
        return Color(color).as_rgb_tuple()[:3]
    return color.as_rgb_tuple()[:3]


def darken(color: str | Color, percentage: float = 10) -> str:
    ratio = 1 - float(percentage) / 100
    red, green, blue = _parse_color_as_rgb(color)
    red = min(max(int(red * ratio), 0), 255)
    green = min(max(int(green * ratio), 0), 255)
    blue = min(max(int(blue * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def lighten(color: str | Color, percentage: float = 10) -> str:
    ratio = float(percentage) / 100
    red, green, blue = _parse_color_as_rgb(color)
    red = min(max(int(red + (255 - red) * ratio), 0), 255)
    green = min(max(int(green + (255 - green) * ratio), 0), 255)
    blue = min(max(int(blue + (255 - blue) * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def opacity(color: str | Color, value: int = 255) -> str:
    red, green, blue = _parse_color_as_rgb(color)
    return f'rgba({red}, {green}, {blue}, {max(min(int(value), 255), 0)})'


def gradient(stops, horizontal: bool = True) -> str:
    if not use_gradients:
        return stops[-1]

    if horizontal:
        grad = 'qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, '
    else:
        grad = 'qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, '

    _stops = [f'stop: {n} {stop}' for n, stop in enumerate(stops)]
    grad += ', '.join(_stops) + ')'

    return grad


def template(css: str, **theme):
    def _increase_match(matchobj):
        font_size, to_add = matchobj.groups()
        return increase(theme[font_size], to_add)

    def _decrease_match(matchobj):
        font_size, to_subtract = matchobj.groups()
        return decrease(theme[font_size], to_subtract)

    def darken_match(matchobj):
        color, percentage = matchobj.groups()
        return darken(theme[color], percentage)

    def lighten_match(matchobj):
        color, percentage = matchobj.groups()
        return lighten(theme[color], percentage)

    def opacity_match(matchobj):
        color, percentage = matchobj.groups()
        return opacity(theme[color], percentage)

    def gradient_match(matchobj):
        horizontal = matchobj.groups()[1] == 'h'
        stops = [i.strip() for i in matchobj.groups()[1].split('-')]
        return gradient(stops, horizontal)

    for k, v in theme.items():
        css = increase_pattern.sub(_increase_match, css)
        css = decrease_pattern.sub(_decrease_match, css)
        css = gradient_pattern.sub(gradient_match, css)
        css = darken_pattern.sub(darken_match, css)
        css = lighten_pattern.sub(lighten_match, css)
        css = opacity_pattern.sub(opacity_match, css)
        if isinstance(v, Color):
            v = v.as_rgb()
        css = css.replace(f'{{{{ {k} }}}}', v)
    return css


def get_system_theme() -> str:
    """Return the system default theme, either 'dark', or 'light'."""
    try:
        from napari._vendor import darkdetect
    except ImportError:
        return 'dark'
    try:
        id_ = darkdetect.theme().lower()
    except AttributeError:
        id_ = 'dark'

    return id_


@overload
def get_theme(theme_id: str) -> Theme: ...


@overload
def get_theme(theme_id: str, as_dict: Literal[False]) -> Theme: ...


@overload
def get_theme(theme_id: str, as_dict: Literal[True]) -> dict[str, Any]: ...


def get_theme(theme_id: str, as_dict: bool | None = None):
    """Get a copy of theme based on it's id.

    If you get a copy of the theme, changes to the theme model will not be
    reflected in the UI unless you replace or add the modified theme to
    the `_themes` container.

    Parameters
    ----------
    theme_id : str
        ID of requested theme.
    as_dict : bool
        .. deprecated:: 0.5.0

            Use ``get_theme(...).to_rgb_dict()``

        Flag to indicate that the old-style dictionary
        should be returned. This will emit deprecation warning.

    Returns
    -------
    theme: dict of str: str
        Theme mapping elements to colors. A copy is created
        so that manipulating this theme can be done without
        side effects.
    """
    if theme_id == 'system':
        theme_id = get_system_theme()

    if theme_id not in _themes:
        raise ValueError(
            trans._(
                'Unrecognized theme {id}. Available themes are {themes}',
                deferred=True,
                id=theme_id,
                themes=available_themes(),
            )
        )
    theme = _themes[theme_id].copy()
    if as_dict is not None:
        warnings.warn(
            trans._(
                'The `as_dict` kwarg has been deprecated since Napari 0.5.0 and '
                'will be removed in future version. You can use `get_theme(...).to_rgb_dict()`',
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
    if as_dict:
        return theme.to_rgb_dict()
    return theme


_themes: EventedDict[str, Theme] = EventedDict(basetype=Theme)


def register_theme(theme_id, theme, source):
    """Register a new or updated theme.

    Parameters
    ----------
    theme_id : str
        id of requested theme.
    theme : dict of str: str, Theme
        Theme mapping elements to colors.
    source : str
        Source plugin of theme
    """
    if isinstance(theme, dict):
        theme = Theme(**theme)
    assert isinstance(theme, Theme)
    _themes[theme_id] = theme

    build_theme_svgs(theme_id, source)


def unregister_theme(theme_id):
    """Remove existing theme.

    Parameters
    ----------
    theme_id : str
        id of the theme to be removed.
    """
    _themes.pop(theme_id, None)


def available_themes() -> list[str]:
    """List available themes.

    Returns
    -------
    list of str
        ids of available themes.
    """
    return [*_themes, 'system']


def is_theme_available(theme_id):
    """Check if a theme is available.

    Parameters
    ----------
    theme_id : str
        id of requested theme.

    Returns
    -------
    bool
        True if the theme is available, False otherwise.
    """
    if theme_id == 'system':
        return True
    if theme_id not in _themes and _theme_path(theme_id).exists():
        plugin_name_file = _theme_path(theme_id) / PLUGIN_FILE_NAME
        if not plugin_name_file.exists():
            return False
        plugin_name = plugin_name_file.read_text()
        with suppress(ModuleNotFoundError):
            npe2.PluginManager.instance().register(plugin_name)
        _install_npe2_themes(_themes)

    return theme_id in _themes


def rebuild_theme_settings():
    """update theme information in settings.

    here we simply update the settings to reflect current list of available
    themes.
    """
    from napari.settings import get_settings

    settings = get_settings()
    settings.appearance.refresh_themes()


DARK = Theme(
    id='dark',
    label='Default Dark',
    background='rgb(38, 41, 48)',
    foreground='rgb(65, 72, 81)',
    primary='rgb(90, 98, 108)',
    secondary='rgb(134, 142, 147)',
    highlight='rgb(106, 115, 128)',
    text='rgb(240, 241, 242)',
    icon='rgb(209, 210, 212)',
    warning='rgb(227, 182, 23)',
    error='rgb(153, 18, 31)',
    current='rgb(0, 122, 204)',
    syntax_style='native',
    console='rgb(18, 18, 18)',
    canvas='black',
    font_size='12pt' if sys.platform == 'darwin' else '9pt',
)
LIGHT = Theme(
    id='light',
    label='Default Light',
    background='rgb(239, 235, 233)',
    foreground='rgb(214, 208, 206)',
    primary='rgb(188, 184, 181)',
    secondary='rgb(150, 146, 144)',
    highlight='rgb(163, 158, 156)',
    text='rgb(59, 58, 57)',
    icon='rgb(107, 105, 103)',
    warning='rgb(227, 182, 23)',
    error='rgb(255, 18, 31)',
    current='rgb(253, 240, 148)',
    syntax_style='default',
    console='rgb(255, 255, 255)',
    canvas='white',
    font_size='12pt' if sys.platform == 'darwin' else '9pt',
)

register_theme('dark', DARK, 'builtin')
register_theme('light', LIGHT, 'builtin')


# this function here instead of plugins._npe2 to avoid circular import
def _install_npe2_themes(themes=None):
    if themes is None:
        themes = _themes
    import npe2

    for manifest in npe2.PluginManager.instance().iter_manifests(
        disabled=False
    ):
        for theme in manifest.contributions.themes or ():
            # get fallback values
            theme_dict = themes[theme.type].dict()
            # update available values
            theme_info = theme.dict(exclude={'colors'}, exclude_unset=True)
            theme_colors = theme.colors.dict(exclude_unset=True)
            theme_dict.update(theme_info)
            theme_dict.update(theme_colors)
            try:
                register_theme(theme.id, theme_dict, manifest.name)
            except ValueError:
                logging.exception('Registration theme failed.')


_install_npe2_themes(_themes)
_themes.events.added.connect(rebuild_theme_settings)
_themes.events.removed.connect(rebuild_theme_settings)
