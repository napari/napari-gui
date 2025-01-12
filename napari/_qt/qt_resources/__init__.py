from pathlib import Path
from typing import Optional

from napari._qt.qt_resources._svg import QColoredSVGIcon
from napari.settings import get_settings

__all__ = ['QColoredSVGIcon', 'get_stylesheet']


STYLE_PATH = (Path(__file__).parent / 'styles').resolve()
STYLES = {x.stem: str(x) for x in STYLE_PATH.iterdir() if x.suffix == '.qss'}


def get_stylesheet(
    theme_id: str | None = None,
    extra: list[str] | None = None,
    extra_variables: dict[str, str] | None = None,
) -> str:
    """Combine all qss files into single, possibly pre-themed, style string.

    Parameters
    ----------
    theme_id : str, optional
        Theme to apply to the stylesheet. If no theme is provided, the returned
        stylesheet will still have ``{{ template_variables }}`` that need to be
        replaced using the :func:`napari.utils.theme.template` function prior
        to using the stylesheet.
    extra : list of str, optional
        Additional paths to QSS files to include in stylesheet, by default None
    extra_variables : dict, optional
        Dictionary of variables values that replace default theme values.
        For example: `{ 'font_size': '14pt'}`

    Returns
    -------
    css : str
        The combined stylesheet.
    """
    stylesheet = ''
    for key in sorted(STYLES.keys()):
        file = STYLES[key]
        with open(file) as f:
            stylesheet += f.read()
    if extra:
        for file in extra:
            with open(file) as f:
                stylesheet += f.read()

    if theme_id:
        from napari.utils.theme import get_theme, template

        theme_dict = get_theme(theme_id).to_rgb_dict()
        if extra_variables:
            theme_dict.update(extra_variables)

        return template(stylesheet, **theme_dict)

    return stylesheet


def get_current_stylesheet(extra: list[str] | None = None) -> str:
    """
    Return the current stylesheet base on settings. This is wrapper around
    :py:func:`get_stylesheet` that takes the current theme base on settings.

    Parameters
    ----------
    extra : list of str, optional
        Additional paths to QSS files to include in stylesheet, by default None

    Returns
    -------
    css : str
        The combined stylesheet.
    """
    settings = get_settings()
    return get_stylesheet(settings.appearance.theme, extra)
