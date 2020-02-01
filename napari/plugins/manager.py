import importlib
import os
import pkgutil
import warnings
from logging import Logger

import pluggy

from . import _builtins, hookspecs, validators

logger = Logger(__name__)


class NapariPluginManager(pluggy.PluginManager):
    def __init__(self, autodiscover=True, greedy_validation=False):
        """pluggy.PluginManager with napari-specific functionality

        In addition to the pluggy functionality, this subclass adds
        autodiscovery using package naming convention.

        Parameters
        ----------
        autodiscover : bool, optional
            Whether to autodiscover plugins by naming convention and setuptools
            entry_points, by default True
        greedy_validation : bool, optional
            Whether to immediately validate hookimpls upon registration,
            by default False
        """
        self._initialized = False
        super().__init__("napari")
        self.greedy_validation = greedy_validation

        # define hook specifications and validators
        self.add_hookspecs(hookspecs)
        self.add_hook_validators(validators)

        # register our own built plugins
        self.register(_builtins, name='builtins')
        # discover external plugins
        if not os.environ.get("NAPARI_DISABLE_PLUGIN_AUTOLOAD"):
            if autodiscover:
                self.discover()

        # validate registered hookimplementations
        if greedy_validation:
            self.validate_hookimpls()
        self._initialized = True

    def register(self, plugin, name=None):
        """Register a plugin and return its canonical name.

        If the name is blocked from registering returns ``None``.
        If the name is already registered, returns ValueError.

        Parameters
        ----------
        plugin : module or class
            The module or class to be registered
        name : str, optional
            If provided, use as name of the plugin. by default None

        Raises
        ------
        ValueError
            if the plugin is already registered
        """
        plugin_name = super().register(plugin, name=name)
        if self.greedy_validation and self._initialized:
            self.validate_hookimpls(plugin_name=plugin_name)

    def discover(self):
        # avoid circular import
        from . import PLUGIN_ENTRYPOINT, PLUGIN_PREFIX

        count = 0
        if not os.environ.get("NAPARI_DISABLE_ENTRYPOINT_PLUGINS"):
            # register modules defining the napari entry_point in setup.py
            count += self.load_setuptools_entrypoints(PLUGIN_ENTRYPOINT)
        if not os.environ.get("NAPARI_DISABLE_NAMEPREFIX_PLUGINS"):
            # register modules using naming convention
            count += self.load_modules_by_prefix(PLUGIN_PREFIX)

        if count:
            msg = f'loaded {count} plugins:\n  '
            msg += "\n  ".join([n for n, m in self.list_name_plugin()])
            logger.info(msg)
        return count

    def add_hook_validators(self, module_or_class):
        """Add hookimpl validators defined in the given ``module_or_class``.

        Functions are recognized if they have been decorated with `@validates`.
        """
        names = []
        for name in dir(module_or_class):
            func = getattr(module_or_class, name)
            specname = getattr(func, 'validates', None)
            if specname is not None:
                hook_caller = getattr(self.hook, specname, None)
                if not hook_caller:
                    raise ValueError("")
                hook_caller.validate = func
                names.append(name)

        if not names:
            warnings.warn(f"No hookimpl validators found in {module_or_class}")

    def _validate_hookimpl(self, hookimpl, specname):
        hook_caller = getattr(self.hook, specname)
        validate = getattr(hook_caller, 'validate', None)
        if validate:
            try:
                validate(hookimpl)
            except validators.HookImplementationError as e:
                if hookimpl.hookwrapper:
                    methods = hook_caller._wrappers
                else:
                    methods = hook_caller._nonwrappers
                methods.pop(methods.index(hookimpl))
                logger.warning(
                    f'"{specname}" hook from plugin "{hookimpl.plugin_name}"'
                    f'was inactivated due to validation error: {e}'
                )

    def validate_hookimpls(
        self, specname: str = None, plugin_name: str = None
    ):
        """Calls registered validator functions on registered hookimpls.

        Parameters
        ----------
        specname : str, optional
            Name of a registered hookspec.
            If provided, only hookimplementations that are registered to
            a hookspec named ``specname`` will be validated.
            by default, **all** hookspecs will be validated.
        plugin_name : str, optional
            Name of a registered plugin.
            If provided, only hookimplementations provided by ``plugin_name``
            will be validated. by default, **all** plugins will be validated.

        Raises
        ------
        ValueError
            if either an invalid ``plugin_name`` or ``specname`` are provided.
        """
        if plugin_name is not None and not self.get_plugin(plugin_name):
            raise ValueError(f"no plugin registered by the name {plugin_name}")

        # by default, validate all hookspecs
        specnames = list(vars(self.hook)) if specname is None else [specname]
        for specname in specnames:
            hook_caller = getattr(self.hook, specname, None)
            if not hook_caller:
                raise ValueError(
                    f"No hook registered for hookspec named '{specname}''"
                )
            for hookimpl in hook_caller.get_hookimpls():
                if plugin_name and plugin_name != hookimpl.plugin_name:
                    continue
                self._validate_hookimpl(hookimpl, specname)

    def load_modules_by_prefix(self, prefix):
        count = 0
        for finder, name, ispkg in pkgutil.iter_modules():
            if (
                not name.startswith(prefix)
                or self.get_plugin(name)
                or self.is_blocked(name)
            ):
                continue
            try:
                mod = importlib.import_module(name)
                self.register(mod, name=name)
                count += 1
            except Exception as e:
                logger.error(f'failed to import plugin: {name}: {str(e)}')
                self.unregister(mod)
        return count


NapariPluginManager.PluginValidationError = (
    pluggy.manager.PluginValidationError
)
