"""
Localization utilities to find available language packs and packages with
localization data.
"""

import gettext
import os
from pathlib import Path
from typing import ClassVar

from yaml import safe_load

from napari.utils._base import _DEFAULT_CONFIG_PATH, _DEFAULT_LOCALE

# Entry points
NAPARI_LANGUAGEPACK_ENTRY = 'napari.languagepack'

# Constants
LOCALE_DIR = 'locale'


def _get_display_name(
    locale: str, display_locale: str = _DEFAULT_LOCALE
) -> str:
    """
    Return the language name to use with a `display_locale` for a given language locale.

    This is used to generate the preferences dialog options.

    Parameters
    ----------
    locale : str
        The language name to use.
    display_locale : str, optional
        The language to display the `locale`.

    Returns
    -------
    str
        Localized `locale` and capitalized language name using `display_locale` as language.
    """
    try:
        # This is a dependency of the language packs to keep out of core
        import babel
    except ModuleNotFoundError:
        display_name = display_locale.capitalize()
    else:
        locale = locale if _is_valid_locale(locale) else _DEFAULT_LOCALE
        display_locale = (
            display_locale
            if _is_valid_locale(display_locale)
            else _DEFAULT_LOCALE
        )
        loc = babel.Locale.parse(locale)
        display_name_ = loc.get_display_name(display_locale)
        if display_name_ is None:
            raise RuntimeError(f'Could not find {display_locale}')
        display_name = display_name_.capitalize()

    return display_name


def _is_valid_locale(locale: str) -> bool:
    """
    Check if a `locale` value is valid.

    Parameters
    ----------
    locale : str
        Language locale code.

    Notes
    -----
    A valid locale is in the form language (See ISO-639 standard) and an
    optional territory (See ISO-3166 standard).

    Examples of valid locales:
    - English: "en"
    - Australian English: "en_AU"
    - Portuguese: "pt"
    - Brazilian Portuguese: "pt_BR"

    Examples of invalid locales:
    - Australian Spanish: "es_AU"
    - Brazilian German: "de_BR"
    """
    valid = False
    try:
        # This is a dependency of the language packs to keep out of core
        import babel

        babel.Locale.parse(locale)
        valid = True
    except ModuleNotFoundError:
        valid = True
    except ValueError:
        pass
    except babel.core.UnknownLocaleError:
        pass

    return valid


def get_language_packs(display_locale: str = _DEFAULT_LOCALE) -> dict:
    """
    Return the available language packs installed in the system.

    The returned information contains the languages displayed in the current
    locale. This can be used to generate the preferences dialog information.

    Parameters
    ----------
    display_locale : str, optional
        Default is _DEFAULT_LOCALE.

    Returns
    -------
    dict
        A dict with the native and display language for all locales found.

    Examples
    --------
    >>> get_language_packs("es_CO")
    {
        'en': {'displayName': 'Inglés', 'nativeName': 'English'},
        'es_CO': {
            'displayName': 'Español (colombia)',
            'nativeName': 'Español (colombia)',
        },
    }
    """
    from napari_plugin_engine.manager import iter_available_plugins

    lang_packs = iter_available_plugins(NAPARI_LANGUAGEPACK_ENTRY)
    found_locales = {k: v for (k, v, _) in lang_packs}

    invalid_locales = []
    valid_locales = []
    for locale in found_locales:
        if _is_valid_locale(locale):
            valid_locales.append(locale)
        else:
            invalid_locales.append(locale)

    display_locale = (
        display_locale if display_locale in valid_locales else _DEFAULT_LOCALE
    )
    locales = {
        _DEFAULT_LOCALE: {
            'displayName': _get_display_name(_DEFAULT_LOCALE, display_locale),
            'nativeName': _get_display_name(_DEFAULT_LOCALE, _DEFAULT_LOCALE),
        }
    }
    for locale in valid_locales:
        locales[locale] = {
            'displayName': _get_display_name(locale, display_locale),
            'nativeName': _get_display_name(locale, locale),
        }

    return locales


# --- Translators
# ----------------------------------------------------------------------------
class TranslationString(str):
    """
    A class that allows to create a deferred translations.

    See https://docs.python.org/3/library/gettext.html for documentation
    of the arguments to __new__ and __init__ in this class.
    """

    __slots__ = (
        '_deferred',
        '_domain',
        '_kwargs',
        '_msgctxt',
        '_msgid',
        '_msgid_plural',
        '_n',
    )

    def __deepcopy__(self, memo):
        from copy import deepcopy

        kwargs = deepcopy(self._kwargs)
        # Remove `n` from `kwargs` added in the initializer
        # See https://github.com/napari/napari/issues/4736
        kwargs.pop('n')
        return TranslationString(
            domain=self._domain,
            msgctxt=self._msgctxt,
            msgid=self._msgid,
            msgid_plural=self._msgid_plural,
            n=self._n,
            deferred=self._deferred,
            **kwargs,
        )

    def __new__(
        cls,
        domain: str | None = None,
        msgctxt: str | None = None,
        msgid: str | None = None,
        msgid_plural: str | None = None,
        n: str | None = None,
        deferred: bool = False,
        **kwargs,
    ):
        if msgid is None:
            raise ValueError(
                trans._('Must provide at least a `msgid` parameter!')
            )

        kwargs['n'] = n

        return str.__new__(
            cls,
            cls._original_value(
                msgid,
                msgid_plural,
                n,
                kwargs,
            ),
        )

    def __init__(
        self,
        domain: str,
        msgid: str,
        msgctxt: str | None = None,
        msgid_plural: str | None = None,
        n: int | None = None,
        deferred: bool = False,
        **kwargs,
    ) -> None:
        self._domain = domain
        self._msgctxt = msgctxt
        self._msgid = msgid
        self._msgid_plural = msgid_plural
        self._n = n
        self._deferred = deferred
        self._kwargs = kwargs

        # Add `n` to `kwargs` to use with `format`
        self._kwargs['n'] = n

    def __repr__(self):
        return repr(self.__str__())

    def __str__(self):
        return self.value() if self._deferred else self.translation()

    @classmethod
    def _original_value(cls, msgid, msgid_plural, n, kwargs):
        """
        Return the original string with interpolated kwargs, if provided.

        Parameters
        ----------
        msgid : str
            The singular string to translate.
        msgid_plural : str
            The plural string to translate.
        n : int
            The number for pluralization.
        kwargs : dict
            Any additional arguments to use when formating the string.
        """
        string = msgid if n is None or n == 1 else msgid_plural
        return string.format(**kwargs)

    def value(self) -> str:
        """
        Return the original string with interpolated kwargs, if provided.
        """
        return self._original_value(
            self._msgid,
            self._msgid_plural,
            self._n,
            self._kwargs,
        )

    def translation(self) -> str:
        """
        Return the translated string with interpolated kwargs, if provided.
        """
        if (
            self._n is not None
            and self._msgid_plural is not None
            and self._msgctxt is not None
        ):
            translation = gettext.dnpgettext(
                self._domain,
                self._msgctxt,
                self._msgid,
                self._msgid_plural,
                self._n,
            )
        elif self._n is not None and self._msgid_plural is not None:
            translation = gettext.dngettext(
                self._domain,
                self._msgid,
                self._msgid_plural,
                self._n,
            )
        elif self._msgctxt is not None:
            translation = gettext.dpgettext(
                self._domain,
                self._msgctxt,
                self._msgid,
            )
        else:
            translation = gettext.dgettext(
                self._domain,
                self._msgid,
            )

        return translation.format(**self._kwargs)


class TranslationBundle:
    """
    Translation bundle providing gettext translation functionality.

    Parameters
    ----------
    domain : str
        The python package/module that this bundle points to. This corresponds
        to the module name of either the core package (``napari``) or any
        extension, for example ``napari_console``. The language packs will
        contain ``*.mo`` files with these names.
    locale : str
        The locale for this bundle. Examples include "en_US", "en_CO".
    """

    def __init__(self, domain: str, locale: str) -> None:
        self._domain = domain
        self._locale = locale

        self._update_locale(locale)

    def _update_locale(self, locale: str):
        """
        Update the locale environment variables.

        Parameters
        ----------
        locale : str
            The language name to use.
        """
        self._locale = locale
        localedir = None
        if locale.split('_')[0] != _DEFAULT_LOCALE:
            from napari_plugin_engine.manager import iter_available_plugins

            lang_packs = iter_available_plugins(NAPARI_LANGUAGEPACK_ENTRY)
            data = {k: v for (k, v, _) in lang_packs}
            if locale not in data:
                import warnings

                trans = self
                warnings.warn(
                    trans._(
                        'Requested locale not available: {locale}',
                        deferred=True,
                        locale=locale,
                    )
                )
            else:
                import importlib

                mod = importlib.import_module(data[locale])
                if mod.__file__ is not None:
                    localedir = Path(mod.__file__).parent / LOCALE_DIR
                else:
                    raise RuntimeError(f'Could not find __file__ for {mod}')

        gettext.bindtextdomain(self._domain, localedir=localedir)

    def _dnpgettext(
        self,
        *,
        msgid: str,
        msgctxt: str | None = None,
        msgid_plural: str | None = None,
        n: int | None = None,
        **kwargs,
    ) -> str:
        """
        Helper to handle all trans methods and delegate to corresponding
        gettext methods.

        Must provide one of the following sets of arguments:
        - msgid
        - msgid, msgctxt
        - msgid, msgid_plural, n
        - msgid, msgid_plural, n, msgctxt

        Parameters
        ----------
        msgctxt : str, optional
            The message context.
        msgid : str, optional
            The singular string to translate.
        msgid_plural : str, optional
            The plural string to translate.
        n : int, optional
            The number for pluralization.
        **kwargs : dict, optional
            Any additional arguments to use when formating the string.
        """
        if msgctxt is not None and n is not None and msgid_plural is not None:
            translation = gettext.dnpgettext(
                self._domain,
                msgctxt,
                msgid,
                msgid_plural,
                n,
            )
        elif n is not None and msgid_plural is not None:
            translation = gettext.dngettext(
                self._domain,
                msgid,
                msgid_plural,
                n,
            )
        elif msgctxt is not None:
            translation = gettext.dpgettext(self._domain, msgctxt, msgid)
        else:
            translation = gettext.dgettext(self._domain, msgid)

        kwargs['n'] = n
        return translation.format(**kwargs)

    def _(
        self, msgid: str, deferred: bool = False, **kwargs
    ) -> TranslationString | str:
        """
        Shorthand for `gettext.gettext` with enhanced functionality.

        Parameters
        ----------
        msgid : str
            The singular string to translate.
        deferred : bool, optional
            Define if the string translation should be deferred or executed
            in place. Default is False.
        **kwargs : dict, optional
            Any additional arguments to use when formatting the string.

        Returns
        -------
        TranslationString or str
            The translation string which might be deferred or translated in
            place.
        """
        return (
            TranslationString(
                domain=self._domain, msgid=msgid, deferred=deferred, **kwargs
            )
            if deferred
            else self._dnpgettext(msgid=msgid, **kwargs)
        )

    def _n(
        self,
        msgid: str,
        msgid_plural: str,
        n: int,
        deferred: bool | None = False,
        **kwargs,
    ) -> TranslationString | str:
        """
        Shorthand for `gettext.ngettext` with enhanced functionality.

        Parameters
        ----------
        msgid : str
            The singular string to translate.
        msgid_plural : str
            The plural string to translate.
        n : int
            The number for pluralization.
        deferred : bool, optional
            Define if the string translation should be deferred or executed
            in place. Default is False.
        **kwargs : dict, optional
            Any additional arguments to use when formating the string.

        Returns
        -------
        TranslationString or str
            The translation string which might be deferred or translated in
            place.
        """
        return (
            TranslationString(
                domain=self._domain,
                msgid=msgid,
                msgid_plural=msgid_plural,
                n=n,
                deferred=deferred,
                **kwargs,
            )
            if deferred
            else self._dnpgettext(
                msgid=msgid, msgid_plural=msgid_plural, n=n, **kwargs
            )
        )

    def _p(
        self,
        msgctxt: str,
        msgid: str,
        deferred: bool | None = False,
        **kwargs,
    ) -> TranslationString | str:
        """
        Shorthand for `gettext.pgettext` with enhanced functionality.

        Parameters
        ----------
        msgctxt : str
            The message context.
        msgid : str
            The singular string to translate.
        deferred : bool, optional
            Define if the string translation should be deferred or executed
            in place. Default is False.
        **kwargs : dict, optional
            Any additional arguments to use when formating the string.

        Returns
        -------
        TranslationString or str
            The translation string which might be deferred or translated in
            place.
        """
        return (
            TranslationString(
                domain=self._domain,
                msgctxt=msgctxt,
                msgid=msgid,
                deferred=deferred,
                **kwargs,
            )
            if deferred
            else self._dnpgettext(msgctxt=msgctxt, msgid=msgid, **kwargs)
        )

    def _np(
        self,
        msgctxt: str,
        msgid: str,
        msgid_plural: str,
        n: int,
        deferred: bool | None = False,
        **kwargs,
    ) -> TranslationString | str:
        """
        Shorthand for `gettext.npgettext` with enhanced functionality.

        Parameters
        ----------
        msgctxt : str
            The message context.
        msgid : str
            The singular string to translate.
        msgid_plural : str
            The plural string to translate.
        n : int
            The number for pluralization.
        deferred : bool, optional
            Define if the string translation should be deferred or executed
            in place. Default is False.
        **kwargs : dict, optional
            Any additional arguments to use when formating the string.

        Returns
        -------
        TranslationString or str
            The translation string which might be deferred or translated in
            place.
        """
        return (
            TranslationString(
                domain=self._domain,
                msgctxt=msgctxt,
                msgid=msgid,
                msgid_plural=msgid_plural,
                n=n,
                deferred=deferred,
                **kwargs,
            )
            if deferred
            else self._dnpgettext(
                msgctxt=msgctxt,
                msgid=msgid,
                msgid_plural=msgid_plural,
                n=n,
                **kwargs,
            )
        )


class _Translator:
    """
    Translations manager.
    """

    _TRANSLATORS: ClassVar[dict[str, TranslationBundle]] = {}
    _LOCALE = _DEFAULT_LOCALE

    @staticmethod
    def _update_env(locale: str):
        """
        Update the locale environment variables based on the settings.

        Parameters
        ----------
        locale : str
            The language name to use.
        """
        for key in ['LANGUAGE', 'LANG']:
            os.environ[key] = f'{locale}.UTF-8'

    @classmethod
    def _set_locale(cls, locale: str):
        """
        Set locale for the translation bundles based on the settings.

        Parameters
        ----------
        locale : str
            The language name to use.
        """
        if _is_valid_locale(locale):
            cls._LOCALE = locale

            if locale.split('_')[0] != _DEFAULT_LOCALE:
                _Translator._update_env(locale)

            for bundle in cls._TRANSLATORS.values():
                bundle._update_locale(locale)

    @classmethod
    def load(cls, domain: str = 'napari') -> TranslationBundle:
        """
        Load translation domain.

        The domain is usually the normalized ``package_name``.

        Parameters
        ----------
        domain : str
            The translations domain. The normalized python package name.

        Returns
        -------
        Translator
            A translator instance bound to the domain.
        """
        if domain in cls._TRANSLATORS:
            trans = cls._TRANSLATORS[domain]
        else:
            trans = TranslationBundle(domain, cls._LOCALE)
            cls._TRANSLATORS[domain] = trans

        return trans


def _load_language(
    default_config_path: str = _DEFAULT_CONFIG_PATH,
    locale: str = _DEFAULT_LOCALE,
) -> str:
    """
    Load language from configuration file directly.

    Parameters
    ----------
    default_config_path : str or Path
        The default configuration path, optional
    locale : str
        The default locale used to display options, optional

    Returns
    -------
    str
        The language locale set by napari.
    """
    if (config_path := Path(default_config_path)).exists():
        with config_path.open() as fh:
            try:
                data = safe_load(fh) or {}
            except Exception as err:  # noqa BLE001
                import warnings

                warnings.warn(
                    'The `language` setting defined in the napari '
                    'configuration file could not be read.\n\n'
                    'The default language will be used.\n\n'
                    f'Error:\n{err}'
                )
                data = {}

        locale = data.get('application', {}).get('language', locale)

    return os.environ.get('NAPARI_LANGUAGE', locale)


# Default translator
trans = _Translator.load('napari')

# Update Translator locale before any other import uses it
_Translator._set_locale(_load_language())

translator = _Translator
