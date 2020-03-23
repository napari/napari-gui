"""Utility script to generate copies of icons with colors based
on our themes. Neccessary workaround because qt does not allow
for styling svg elements using qss
"""

import os
import re
import shutil
from subprocess import run
from typing import Dict, List, Tuple

from ..utils.theme import palettes as _palettes

RESOURCES_DIR = os.path.abspath(os.path.dirname(__file__))
SVGPATH = os.path.join(RESOURCES_DIR, 'icons')

svg_tag_open = re.compile(r'(<svg[^>]*>)')


def themify_icons(
    dest_dir: str,
    svg_path: str = SVGPATH,
    palettes: Dict[str, Dict[str, str]] = _palettes,
    color_lookup: Dict[str, str] = None,
) -> List[str]:
    """Create a new "themed" SVG file, for every SVG file in ``svg_path``.

    Parameters
    ----------
    dest_dir : str
        The directory in which to write all of the themed icons.  Individual
        themes will be in subdirectories named after each theme.
    svg_path : str, optional
        The folder to look in for SVG files, by default will search in a folder
        named ``icons`` in the same directory as this file.
    palettes : dict, optional
        A mapping of ``theme_name: theme_dict``, where ``theme_dict`` is a
        mapping of color classes to rgb strings. By default will uses palettes
        from :const:`napari.resources.utils.theme.palettes`.
    color_lookup : dict, optional
        A mapping of icon name to color class.  If the icon name is not in the
        color_lookup, it's color class will be ``"icon"``.

    Returns
    -------
    files : list of str
        a list of generaged SVG filepaths (each relative to dest_dir)
    """

    color_lookup = color_lookup or {
        'visibility': 'text',
        'visibility_off': 'highlight',
        'menu': 'highlight',
        'drop_down': 'secondary',
        'plus': 'secondary',
        'minus': 'secondary',
        'properties_contract': 'secondary',
        'properties_expand': 'secondary',
    }
    icon_names = [
        i.replace('.svg', '')
        for i in os.listdir(SVGPATH)
        if i.endswith('.svg')
    ]

    svg_style_insert = r"""<style type="text/css">
    path{fill:{{ color }}}
    polygon{fill:{{ color }}}
    circle{fill:{{ color }}}
    rect{fill:{{ color }}}
    </style>"""

    files = []
    for theme_name, palette in palettes.items():
        palette_dir = os.path.join(dest_dir, theme_name)
        os.makedirs(palette_dir, exist_ok=True)
        for icon_name in icon_names:
            svg_name = icon_name + '.svg'
            new_file = os.path.join(palette_dir, svg_name)
            color = color_lookup.get(icon_name, 'icon')
            css = svg_style_insert.replace('{{ color }}', palette[color])
            with open(os.path.join(SVGPATH, svg_name), 'r') as fr:
                contents = fr.read()
            with open(new_file, 'w') as fw:
                # use regex to find the svg tag and insert css right after
                # (the '\\1' syntax includes the matched tag in the output)
                fw.write(svg_tag_open.sub(f'\\1{css}', contents))
            files.append(os.path.join(theme_name, svg_name))
    return files


def build_resources_qrc(
    dest_dir: str, overwrite: bool = False
) -> Tuple[str, str]:
    """Create a res.qrc file for all icons generated by ``themify_icons``.

    Parameters
    ----------
    dest_dir : str
        The directory in which to write create the output file.  Themed SVG
        icons will also be written to ``dest_dir/themes``.
    overwrite : bool
        Whether to force rebuilding of the icons and res.qrc file, by default
        False

    Returns
    -------
    tuple
        2-tuple of (path-to-qrc.res, path-to-theme-directory).
    """
    qrc_path = os.path.join(dest_dir, 'res.qrc')
    theme_dir = os.path.join(dest_dir, 'themes')
    if os.path.exists(qrc_path) and (not overwrite):
        return qrc_path, theme_dir

    qrc_string = """
    <!DOCTYPE RCC>
    <RCC version="1.0">
    <qresource>
    """
    for filename in themify_icons(theme_dir):
        qrc_string += f'\n    <file>themes/{filename}</file>'

    qrc_string += """
    </qresource>
    </RCC>
    """

    with open(qrc_path, 'w') as f:
        f.write(qrc_string)

    return qrc_path, theme_dir


def build_pyqt_resources(out_path: str, overwrite: bool = False) -> str:
    """Build a res.qrc file from icons and convert for python usage.

    calls :func:`build_resources` and then converts using ``pyside2-rcc`` or
    ``pyrcc5`` depending on which is installed in the environment.
    Finally, cleans up autogenerated icon.svgs and res.qrc file after _qt.py
    file is generated

    References
    ----------
    https://doc.qt.io/qt-5/resources.html

    Parameters
    ----------
    out_path : str
        Path to write the python resource file.
    overwrite : bool, optional
        Whether to force rebuilding of the output file, by default False

    Returns
    -------
    out_path : str
        Path to the python resource file.  Import this file to make the SVGs
        and other resources available to Qt stylesheets.
    """

    if os.path.exists(out_path) and not overwrite:
        return out_path

    # build the resource file to the same path
    qrc_path, theme_dir = build_resources_qrc(
        os.path.dirname(out_path), overwrite=overwrite
    )

    # then convert it to a python file
    try:
        run(['pyrcc5', '-o', out_path, qrc_path])
    except FileNotFoundError:
        run(['pyside2-rcc', '-o', out_path, qrc_path])

    # make sure we import from qtpy
    with open(out_path, "rt") as fin:
        data = fin.read()
        data = data.replace('PySide2', 'qtpy').replace('PyQt5', 'qtpy')
    with open(out_path, "wt") as fin:
        fin.write(data)

    # cleanup.
    # we do this here because pip uninstall napari would not collect these
    # and the final `out_path.py` contains all the necessary bytes info
    shutil.rmtree(theme_dir, ignore_errors=True)
    try:
        os.remove(qrc_path)
    except Exception:
        pass
    return out_path
