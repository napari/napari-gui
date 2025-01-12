"""Napari Configuration."""

import os
import warnings

from napari.utils.translations import trans


def _set(env_var: str) -> bool:
    """Return True if the env variable is set and non-zero.

    Returns
    -------
    bool
        True if the env var was set to a non-zero value.
    """
    return os.getenv(env_var) not in [None, '0']


"""
Experimental Features

Shared Memory Server
--------------------
Experimental shared memory service. Only enabled if NAPARI_MON is set to
the path of a config file. See this PR for more info:
https://github.com/napari/napari/pull/1909.
"""


# Handle old async/octree deprecated attributes by returning their
# fixed values in the module level __getattr__
# https://peps.python.org/pep-0562/
# Other module attributes are defined as normal.
def __getattr__(name: str) -> bool | None:
    if name == 'octree_config':
        warnings.warn(
            trans._(
                'octree_config is deprecated in version 0.5 and will be removed in a later version.'
                'More generally, the experimental octree feature was removed in napari version 0.5 so this value is always None. '
                'If you need to use that experimental feature, continue to use the latest 0.4 release. '
                'Also look out for announcements regarding similar efforts.'
            ),
            DeprecationWarning,
        )
        return None
    if name == 'async_octree':
        warnings.warn(
            trans._(
                'async_octree is deprecated in version 0.5 and will be removed in a later version.'
                'More generally, the experimental octree feature was removed in version 0.5 so this value is always False. '
                'If you need to use that experimental feature, continue to use the latest 0.4 release. '
                'Also look out for announcements regarding similar future efforts.'
            ),
            DeprecationWarning,
        )
        return False
    if name == 'async_loading':
        # For async_loading, we could get the value of the remaining
        # async setting. We do not because that is dynamic, so will not
        # handle an import of the form
        #
        # `from napari.utils.config import async_loading`
        #
        # consistently. Instead, we let this attribute effectively
        # refer to the old async which is always off in napari now.
        warnings.warn(
            trans._(
                'async_loading is deprecated in version 0.5 and will be removed in a later version. '
                'The old approach to async loading was removed in version 0.5 so this value is always False. '
                'Instead, please use napari.settings.get_settings().experimental.async_ to use a new approach. '
                'If you need to specifically use the old approach, continue to use the latest 0.4 release.'
            ),
            DeprecationWarning,
        )
        return False
    return None


# Shared Memory Server
monitor = _set('NAPARI_MON')
