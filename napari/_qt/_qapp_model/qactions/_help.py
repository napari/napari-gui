"""Qt 'Help' menu Actions."""

import sys
from functools import partial
from webbrowser import open

from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod
from packaging.version import parse

from napari import __version__
from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.dialogs.log_dialog import LogDialog
from napari._qt.dialogs.qt_about import QtAbout
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans


def _show_about(window: Window):
    QtAbout.showAbout(window._qt_window)


def _show_log(window: Window):
    LogDialog.showLog(window._qt_window)


v = parse(__version__)
VERSION = 'dev' if v.is_devrelease or v.is_prerelease else str(v.base_version)

HELP_URLS: dict[str, str] = {
    'getting_started': f'https://napari.org/{VERSION}/tutorials/start_index.html',
    'tutorials': f'https://napari.org/{VERSION}/tutorials/index.html',
    'layers_guide': f'https://napari.org/{VERSION}/howtos/layers/index.html',
    'examples_gallery': f'https://napari.org/{VERSION}/gallery.html',
    'release_notes': f'https://napari.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html',
    'github_issue': 'https://github.com/napari/napari/issues',
    'homepage': 'https://napari.org',
}

Q_HELP_ACTIONS: list[Action] = [
    Action(
        id='napari.window.help.info',
        title=trans._('‎napari Info'),
        callback=_show_about,
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.RENDER}],
        status_tip=trans._('About napari'),
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.Slash)],
    ),
    Action(
        id='napari.window.help.about_macos',
        title=trans._('About napari'),
        callback=_show_about,
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'group': MenuGroup.RENDER,
                'when': sys.platform == 'darwin',
            }
        ],
        status_tip=trans._('About napari'),
    ),
    Action(
        id='napari.window.help.getting_started',
        title=trans._('Getting started'),
        callback=partial(open, url=HELP_URLS['getting_started']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.tutorials',
        title=trans._('Tutorials'),
        callback=partial(open, url=HELP_URLS['tutorials']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.layers_guide',
        title=trans._('Using Layers Guides'),
        callback=partial(open, url=HELP_URLS['layers_guide']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.examples',
        title=trans._('Examples Gallery'),
        callback=partial(open, url=HELP_URLS['examples_gallery']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='napari.window.help.release_notes',
        title=trans._('Release Notes'),
        callback=partial(open, url=HELP_URLS['release_notes']),
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'when': VERSION != 'dev',
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id='napari.window.help.github_issue',
        title=trans._('Report an issue on GitHub'),
        callback=partial(open, url=HELP_URLS['github_issue']),
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'when': VERSION == 'dev',
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id='napari.window.help.homepage',
        title=trans._('napari homepage'),
        callback=partial(open, url=HELP_URLS['homepage']),
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.NAVIGATION}],
    ),
    Action(
        id='napari.window.help.show_log',
        title=trans._('Show log'),
        callback=_show_log,
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.RENDER}],
        status_tip=trans._('View and filter logs'),
    ),
]
