# syntax_style for the console must be one of the supported styles from
# pygments - see here for examples https://help.farbox.com/pygments.html
import re
from ast import literal_eval
from typing import Union

from pydantic.color import Color

try:
    from qtpy import QT_VERSION

    major, minor, *rest = QT_VERSION.split('.')
    use_gradients = (int(major) >= 5) and (int(minor) >= 12)
except Exception:
    use_gradients = False

from ..utils.translations import trans
from .events import EventedModel
from .events.containers._evented_dict import EventedDict

THEME_KEYS = [
    "folder",
    "background",
    "foreground",
    "primary",
    "highlight",
    "text",
    "icon",
    "warning",
    "current",
    "syntax_style",
    "console",
    "canvas",
]


class Theme(EventedModel):
    """Theme model."""

    folder: str
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
    current: Color


_themes = EventedDict(
    {
        'dark': Theme(
            **{
                'folder': 'dark',
                'background': 'rgb(38, 41, 48)',
                'foreground': 'rgb(65, 72, 81)',
                'primary': 'rgb(90, 98, 108)',
                'secondary': 'rgb(134, 142, 147)',
                'highlight': 'rgb(106, 115, 128)',
                'text': 'rgb(240, 241, 242)',
                'icon': 'rgb(209, 210, 212)',
                'warning': 'rgb(153, 18, 31)',
                'current': 'rgb(0, 122, 204)',
                'syntax_style': 'native',
                'console': 'rgb(0, 0, 0)',
                'canvas': 'black',
            }
        ),
        'light': Theme(
            **{
                'folder': 'light',
                'background': 'rgb(239, 235, 233)',
                'foreground': 'rgb(214, 208, 206)',
                'primary': 'rgb(188, 184, 181)',
                'secondary': 'rgb(150, 146, 144)',
                'highlight': 'rgb(163, 158, 156)',
                'text': 'rgb(59, 58, 57)',
                'icon': 'rgb(107, 105, 103)',
                'warning': 'rgb(255, 18, 31)',
                'current': 'rgb(253, 240, 148)',
                'syntax_style': 'default',
                'console': 'rgb(255, 255, 255)',
                'canvas': 'white',
            }
        ),
    },
    basetype=Theme,
)


gradient_pattern = re.compile(r'([vh])gradient\((.+)\)')
darken_pattern = re.compile(r'{{\s?darken\((\w+),?\s?([-\d]+)?\)\s?}}')
lighten_pattern = re.compile(r'{{\s?lighten\((\w+),?\s?([-\d]+)?\)\s?}}')
opacity_pattern = re.compile(r'{{\s?opacity\((\w+),?\s?([-\d]+)?\)\s?}}')


def darken(color: Union[str, Color], percentage=10):
    if isinstance(color, str) and color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    else:
        color = color.as_rgb_tuple()
    ratio = 1 - float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red * ratio), 0), 255)
    green = min(max(int(green * ratio), 0), 255)
    blue = min(max(int(blue * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def lighten(color: Union[str, Color], percentage=10):
    if isinstance(color, str) and color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    else:
        color = color.as_rgb_tuple()
    ratio = float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red + (255 - red) * ratio), 0), 255)
    green = min(max(int(green + (255 - green) * ratio), 0), 255)
    blue = min(max(int(blue + (255 - blue) * ratio), 0), 255)
    return f'rgb({red}, {green}, {blue})'


def opacity(color: Union[str, Color], value=255):
    if isinstance(color, str) and color.startswith('rgb('):
        color = literal_eval(color.lstrip('rgb(').rstrip(')'))
    else:
        color = color.as_rgb_tuple()
    red, green, blue = color
    return f'rgba({red}, {green}, {blue}, {max(min(int(value), 255), 0)})'


def gradient(stops, horizontal=True):
    if not use_gradients:
        return stops[-1]

    if horizontal:
        grad = 'qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, '
    else:
        grad = 'qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, '

    _stops = [f'stop: {n} {stop}' for n, stop in enumerate(stops)]
    grad += ", ".join(_stops) + ")"

    return grad


def template(css: str, **theme):
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
        css = gradient_pattern.sub(gradient_match, css)
        css = darken_pattern.sub(darken_match, css)
        css = lighten_pattern.sub(lighten_match, css)
        css = opacity_pattern.sub(opacity_match, css)
        if isinstance(v, Color):
            v = v.as_rgb()
        css = css.replace('{{ %s }}' % k, v)
    return css


def get_theme(name, as_dict=True):
    """Get a theme based on its name

    Parameters
    ----------
    name : str
        Name of requested theme.
    as_dict : bool
        Flag to indicate that the old-style dictionary
        should be returned. This will emit deprecation warning.

    Returns
    -------
    theme: dict of str: str
        Theme mapping elements to colors. A copy is created
        so that manipulating this theme can be done without
        side effects.
    """
    if name in _themes:
        theme = _themes[name]
        _theme = theme.copy()
        if as_dict:
            # warnings.warn(
            #     trans._(
            #         "themes were recently updated to use evented model with"
            #         " Pydantic's" color type rather than the `rgb(XX, YY, ZZ)`"
            #         " value. You can get the old style color by calling
            #         " `color.as_rgb()`. Please update your codebase to reflect"
            #         " this change",
            #         deferred=True,
            #     ),
            #     category=FutureWarning,
            #     stacklevel=2,
            # )
            _theme = _theme.dict()
            _theme = {
                k: v if not isinstance(v, Color) else v.as_rgb()
                for (k, v) in _theme.items()
            }
            return _theme
        return _theme
    else:
        raise ValueError(
            trans._(
                "Unrecognized theme {name}. Available themes are {themes}",
                deferred=True,
                name=name,
                themes=available_themes(),
            )
        )


def register_theme(name, theme):
    """Register a new or updated theme.

    Parameters
    ----------
    name : str
        Name of requested theme.
    theme : dict of str: str
        Theme mapping elements to colors.
    """
    if not isinstance(theme, Theme):
        theme = Theme(**theme)
    _themes[name] = theme


def unregister_theme(name):
    """Remove existing theme.

    Parameters
    ----------
    name : str
        Name of the theme to be removed.
    """
    if name in _themes:
        _themes.pop(name)


def available_themes():
    """List available themes.

    Returns
    -------
    list of str
        Names of available themes.
    """
    return tuple(_themes)
