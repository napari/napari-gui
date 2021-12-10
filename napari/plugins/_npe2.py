from __future__ import annotations

from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

try:
    import npe2
    from npe2.io_utils import read_get_reader
    from npe2.manifest.schema import PluginManifest
except ImportError:
    npe2 = None

if TYPE_CHECKING:
    from npe2._types import LayerData, WidgetCallable
    from npe2.manifest.contributions import WriterContribution
    from qtpy.QtWidgets import QMenu

    from ..layers import Layer
    from ..types import SampleDict


def npe2_or_return(val=None):
    def decorator(func):
        if npe2 is None:

            def noop(*_, **__) -> None:
                return val

            return noop
        return func

    return decorator


npe2_or_return_none = npe2_or_return(None)


class _FakeHookimpl:
    def __init__(self, name):
        self.plugin_name = name


@npe2_or_return_none
def read(
    path: Union[str, Sequence[str]], plugin: Optional[str] = None
) -> Optional[Tuple[List[LayerData], _FakeHookimpl]]:
    """Try to return data for `path`, from reader plugins using a manifest."""
    with suppress(ValueError):
        layer_data, reader = read_get_reader(path, plugin_name=plugin)
        return layer_data, _FakeHookimpl(reader.plugin_name)
    return None


@npe2_or_return([])
def write_layers(
    path: str,
    layers: List[Layer],
    plugin_name: Optional[str] = None,
    writer: Optional[WriterContribution] = None,
) -> List[str]:
    """
    Write layers to a file using an NPE2 plugin.

    Parameters
    ----------
    path : str
        The path (file, directory, url) to write.
    layer_type : str
        All lower-class name of the layer class to be written.
    plugin_name : str, optional
        Name of the plugin to write data with. If None then all plugins
        corresponding to appropriate hook specification will be looped
        through to find the first one that can write the data.
    command_id : str, optional
        npe2 command identifier that uniquely identifies the command to ivoke
        to save layers. If specified, overrides, the plugin_name.

    Returns
    -------
    list of str
        Empty list when no plugin was found, otherwise a list of file paths,
        if any, that were written.
    """
    layer_data = [layer.as_layer_data_tuple() for layer in layers]

    if writer is None:
        return npe2.write(
            path=path, layer_data=layer_data, plugin_name=plugin_name
        )

    n = sum(ltc.max() for ltc in writer.layer_type_constraints())
    args = (path, *layer_data[0][:2]) if n <= 1 else (path, layer_data)
    res = writer.exec(args=args)
    return [res] if isinstance(res, str) else res or []


@npe2_or_return_none
def get_widget_contribution(
    plugin_name: str, widget_name: str
) -> Optional[WidgetCallable]:
    for contrib in npe2.PluginManager.instance().iter_widgets():
        if (
            contrib.plugin_name == plugin_name
            and contrib.display_name == widget_name
        ):
            return contrib.get_callable()
    return None


@npe2_or_return_none
def populate_qmenu(menu: QMenu, menu_key: str):
    """Populate `menu` from a `menu_key` offering in the manifest."""
    # TODO: declare somewhere what menu_keys are valid.
    pm = npe2.PluginManager.instance()
    for item in pm.iter_menu(menu_key):
        if hasattr(item, 'submenu'):
            subm_contrib = pm.get_submenu(item.submenu)
            subm = menu.addMenu(subm_contrib.label)
            populate_qmenu(subm, subm_contrib.id)
        else:
            cmd = pm.get_command(item.command)
            action = menu.addAction(cmd.title)
            action.triggered.connect(lambda *args: cmd.exec(args=args))


@npe2_or_return((None, []))
def file_extensions_string_for_layers(
    layers: Sequence[Layer],
) -> Tuple[Optional[str], List[WriterContribution]]:
    """Create extensions string using npe2.

    When npe2 can be imported, returns an extension string and the list
    of corresponding writers. Otherwise returns (None,[]).

    The extension string is a ";;" delimeted string of entries. Each entry
    has a brief description of the file type and a list of extensions. For
    example:

        "Images (*.png *.jpg *.tif);;All Files (*.*)"

    The writers, when provided, are the
    `npe2.manifest.io.WriterContribution` objects. There is one writer per
    entry in the extension string.
    """

    pm = npe2.PluginManager.instance()
    layer_types = [layer._type_string for layer in layers]
    writers = list(pm.iter_compatible_writers(layer_types))

    def _items():
        """Lookup the command name and its supported extensions."""
        for writer in writers:
            name = pm.get_manifest(writer.command).display_name
            title = f"{name} {writer.name}" if writer.name else name
            yield title, writer.filename_extensions

    # extension strings are in the format:
    #   "<name> (*<ext1> *<ext2> *<ext3>);;+"

    def _fmt_exts(es):
        return " ".join("*" + e for e in es if e) if es else "*.*"

    return (
        ";;".join(f"{name} ({_fmt_exts(exts)})" for name, exts in _items()),
        writers,
    )


@npe2_or_return(iter([]))
def iter_manifests() -> Iterator[PluginManifest]:
    yield from npe2.PluginManager.instance()._manifests.values()


@npe2_or_return(iter([]))
def widget_iterator() -> Iterator[Tuple[str, Tuple[str, Sequence[str]]]]:
    # eg ('dock', ('my_plugin', {'My widget': MyWidget}))
    wdgs: DefaultDict[str, List[str]] = DefaultDict(list)
    for wdg_contrib in npe2.PluginManager.instance().iter_widgets():
        wdgs[wdg_contrib.plugin_name].append(wdg_contrib.display_name)
    return (('dock', x) for x in wdgs.items())


@npe2_or_return(iter([]))
def sample_iterator() -> Iterator[Tuple[str, Dict[str, SampleDict]]]:
    pm = npe2.PluginManager.instance()
    return (
        (
            plugin_name,
            {
                c.key: {'data': c.open, 'display_name': c.display_name}
                for c in contribs
            },
        )
        for plugin_name, contribs in pm.iter_sample_data()
    )


@npe2_or_return((None, []))
def get_sample_data(
    plugin: str, sample: str
) -> Tuple[Optional[Callable[[], Iterable[LayerData]]], List[Tuple[str, str]]]:
    """Get sample data opener from npe2.

    Parameters
    ----------
    plugin : str
        name of a plugin providing a sample
    sample : str
        name of the sample

    Returns
    -------
    tuple
        - first item is a data "opener": a callable that returns an iterable of
          layer data, or None, if none found.
        - second item is a list of available samples (plugin_name, sample_name)
          if no data opener is found.
    """
    pm = npe2.PluginManager.instance()
    for c in pm._contrib._samples.get(plugin, []):
        if c.key == sample:
            return c.open, []
    return None, [(p, x.key) for p, s in pm.iter_sample_data() for x in s]
