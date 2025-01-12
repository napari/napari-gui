import os
import os.path as osp
import re
from enum import IntFlag
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

from npe2 import PluginManifest

from napari.plugins import _npe2, plugin_manager
from napari.settings import get_settings
from napari.types import PathLike


class MatchFlag(IntFlag):
    NONE = 0
    SET = 1
    ANY = 2
    STAR = 4


@lru_cache
def score_specificity(pattern: str) -> tuple[bool, int, list[MatchFlag]]:
    """Score an fnmatch pattern, with higher specificities having lower scores.

    Absolute paths have highest specificity,
    followed by paths with the most nesting,
    then by path segments with the least ambiguity.

    Parameters
    ----------
    pattern : str
        Pattern to score.

    Returns
    -------
    relpath : boolean
        Whether the path is relative or absolute.
    nestedness : negative int
        Level of nestedness of the path, lower is deeper.
    score : List[MatchFlag]
        Path segments scored by ambiguity, higher score is higher ambiguity.
    """
    pattern = osp.normpath(pattern)

    segments = pattern.split(osp.sep)
    score: list[MatchFlag] = []
    ends_with_star = False

    def add(match_flag):
        score[-1] |= match_flag

    # built-in fnmatch does not allow you to escape meta-characters
    # so we don't need to handle them :)
    for segment in segments:
        # collapse foo/*/*/*.bar or foo*/*.bar but not foo*bar/*.baz
        if segment and not (ends_with_star and segment.startswith('*')):
            score.append(MatchFlag.NONE)

        if '*' in segment:
            add(MatchFlag.STAR)
        if '?' in segment:
            add(MatchFlag.ANY)
        if '[' in segment and ']' in segment[segment.index('[') :]:
            add(MatchFlag.SET)

        ends_with_star = segment.endswith('*')

    return not osp.isabs(pattern), 1 - len(score), score


def _get_preferred_readers(path: PathLike) -> list[tuple[str, str]]:
    """Given filepath, find matching readers from preferences.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    filtered_preferences : List[Tuple[str, str]]
        Filtered patterns and their corresponding readers.
    """
    path = os.path.realpath(str(path))

    if osp.isdir(path) and not path.endswith(os.sep):
        path = path + os.sep

    reader_settings = get_settings().plugins.extension2reader

    def filter_fn(kv: tuple[str, str]) -> bool:
        return fnmatch(path, kv[0])

    ret = list(filter(filter_fn, reader_settings.items()))
    return ret


def get_preferred_reader(path: PathLike) -> str | None:
    """Given filepath, find the best matching reader from the preferences.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    reader : str or None
        Best matching reader, if found.
    """
    readers = sorted(
        _get_preferred_readers(path), key=lambda kv: score_specificity(kv[0])
    )
    if readers:
        preferred = readers[0]
        _, reader = preferred
        return reader

    return None


def get_potential_readers(filename: PathLike) -> dict[str, str]:
    """Given filename, returns all readers that may read the file.

    Original plugin engine readers are checked based on returning
    a function from `napari_get_reader`. Npe2 readers are iterated
    based on file extension and accepting directories.

    Returns
    -------
    Dict[str, str]
        dictionary of registered name to display_name
    """
    readers = {}
    hook_caller = plugin_manager.hook.napari_get_reader
    # lower case file extension
    ext = str(Path(filename).suffix).lower()
    filename = str(Path(filename).with_suffix(ext))
    for impl in hook_caller.get_hookimpls():
        reader = hook_caller._call_plugin(impl.plugin_name, path=filename)
        if callable(reader):
            readers[impl.plugin_name] = impl.plugin_name
    readers.update(_npe2.get_readers(filename))
    return readers


def get_all_readers() -> tuple[dict[str, str], dict[str, str]]:
    """
    Return a dict of all npe2 readers and one of all npe1 readers

    Can be removed once npe2 shim is activated.
    """

    npe2_readers = _npe2.get_readers()

    npe1_readers = {}
    for spec, hook_caller in plugin_manager.hooks.items():
        if spec == 'napari_get_reader':
            potential_readers = hook_caller.get_hookimpls()
            for get_reader in potential_readers:
                npe1_readers[get_reader.plugin_name] = get_reader.plugin_name

    return npe2_readers, npe1_readers


def normalized_name(name: str) -> str:
    """
    Normalize a plugin name by replacing underscores and dots by dashes and
    lower casing it.
    """
    return re.sub(r'[-_.]+', '-', name).lower()


def get_filename_patterns_for_reader(plugin_name: str):
    """Return recognized filename patterns, if any, for a given plugin.

    Where a plugin provides multiple readers it will return a set of
    all recognized filename patterns.

    Parameters
    ----------
    plugin_name : str
        name of plugin to find filename patterns for

    Returns
    -------
    set
        set of filename patterns accepted by all plugin's reader contributions
    """
    all_fn_patterns: set[str] = set()
    current_plugin: PluginManifest | None = None
    for manifest in _npe2.iter_manifests():
        if manifest.name == plugin_name:
            current_plugin = manifest
    if current_plugin:
        readers = current_plugin.contributions.readers or []
        for reader in readers:
            all_fn_patterns = all_fn_patterns.union(
                set(reader.filename_patterns)
            )
    # npe1 plugins
    else:
        _, npe1_readers = get_all_readers()
        if plugin_name in npe1_readers:
            all_fn_patterns = {'*'}

    return all_fn_patterns
