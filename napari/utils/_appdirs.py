import hashlib
import os
import sys
from collections.abc import Callable
from functools import partial

import appdirs

PREFIX_PATH = os.path.realpath(sys.prefix)

sha_short = f'{os.path.basename(PREFIX_PATH)}_{hashlib.sha1(PREFIX_PATH.encode()).hexdigest()}'

_appname = 'napari'
_appauthor = False


# all of these also take an optional "version" argument ... but if we want
# to be able to update napari while using data (e.g. plugins, settings) from
# an earlier version, we should leave off the version.

user_data_dir: Callable[[], str] = partial(
    appdirs.user_data_dir, _appname, _appauthor
)
user_config_dir: Callable[[], str] = partial(
    appdirs.user_config_dir, _appname, _appauthor, sha_short
)
user_cache_dir: Callable[[], str] = partial(
    appdirs.user_cache_dir, _appname, _appauthor, sha_short
)
user_state_dir: Callable[[], str] = partial(
    appdirs.user_state_dir, _appname, _appauthor
)
user_log_dir: Callable[[], str] = partial(
    appdirs.user_log_dir, _appname, _appauthor
)
