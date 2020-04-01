import platform

import sys

import napari


def sys_info(as_html=False):
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """
    from napari.plugins import plugin_manager

    sys_version = sys.version.replace('\n', ' ')
    text = (
        f"<b>napari</b>: {napari.__version__}<br>"
        f"<b>Platform</b>: {platform.platform()}<br>"
        f"<b>Python</b>: {sys_version}<br>"
    )

    try:
        from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

        if API_NAME == 'PySide2':
            API_VERSION = PYSIDE_VERSION
        elif API_NAME == 'PyQt5':
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = ''

        text += (
            f"<b>Qt</b>: {QtCore.__version__}<br>"
            f"<b>{API_NAME}</b>: {API_VERSION}<br>"
        )
    except Exception as e:
        text += f"<b>Qt</b>: Import failed ({e})<br>"

    modules = (
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('skimage', 'scikit-image'),
        ('dask', 'Dask'),
        ('vispy', 'VisPy'),
    )

    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:
            text += f"<b>{name}</b>: Import failed ({e})<br>"

    if loaded.get('vispy', False):
        sys_info_text = "<br>".join(
            [
                loaded['vispy'].sys_info().split("\n")[index]
                for index in [-4, -3]
            ]
        ).replace("'", "")
        text += f'<br>{sys_info_text}'

    plugins = []
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        if name == 'builtins':
            continue
        # look for __version__ attr in the plugin module
        version = getattr(plugin, '__version__', '')
        if not version and '.' in plugin.__name__:
            # if no version is found, look in the base module
            root = sys.modules[plugin.__name__.split('.')[0]]
            version = getattr(root, '__version__', '')
        plugins.append(f'  - {name}' + (f': {version}' if version else ''))
    text += '<br><br><b>Plugins</b>:'
    text += ("<br>" + "<br>".join(sorted(plugins))) if plugins else '  None'

    if not as_html:
        text = (
            text.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
        )
    return text


citation_text = (
    'napari contributors (2019). napari: a '
    'multi-dimensional image viewer for python. '
    'doi:10.5281/zenodo.3555620'
)
