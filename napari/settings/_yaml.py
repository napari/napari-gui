from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from app_model.types import KeyBinding
from yaml import SafeDumper, dump_all

from napari._pydantic_compat import BaseModel
from napari.settings._fields import Version

if TYPE_CHECKING:
    from collections.abc import Mapping, Set as AbstractSet
    from typing import Any, TypeVar, Union

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]
    Model = TypeVar('Model', bound=BaseModel)


class YamlDumper(SafeDumper):
    """The default YAML serializer for our pydantic models.

    Add support for custom types by using `YamlDumper.add_representer` or
    `YamlDumper.add_multi_representer` below.
    """


# add_representer requires a strict type match
# add_multi_representer also works for all subclasses of the provided type.
YamlDumper.add_multi_representer(str, YamlDumper.represent_str)
YamlDumper.add_multi_representer(
    Enum, lambda dumper, data: dumper.represent_str(data.value)
)
# the default set representer is ugly:
# disabled_plugins: !!set
#   bioformats: null
# and pydantic will make sure that incoming sets are converted to sets
YamlDumper.add_representer(
    set, lambda dumper, data: dumper.represent_list(data)
)
YamlDumper.add_representer(
    Version, lambda dumper, data: dumper.represent_str(str(data))
)
YamlDumper.add_representer(
    KeyBinding, lambda dumper, data: dumper.represent_str(str(data))
)


class PydanticYamlMixin(BaseModel):
    """Mixin that provides yaml dumping capability to pydantic BaseModel.

    To provide a custom yaml Dumper on a subclass, provide a `yaml_dumper`
    on the Config:

        class Config:
            yaml_dumper = MyDumper
    """

    def yaml(
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny = None,  # type: ignore
        exclude: AbstractSetIntStr | MappingIntStrAny = None,  # type: ignore
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        dumper: type[SafeDumper] | None = None,
        **dumps_kwargs: Any,
    ) -> str:
        """Serialize model to yaml."""
        data = self.dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        if self.__custom_root_type__:
            from napari._pydantic_compat import ROOT_KEY

            data = data[ROOT_KEY]
        return self._yaml_dump(data, dumper, **dumps_kwargs)

    def _yaml_dump(
        self, data, dumper: type[SafeDumper] | None = None, **kw
    ) -> str:
        kw.setdefault('sort_keys', False)
        dumper = dumper or getattr(self.__config__, 'yaml_dumper', YamlDumper)
        return dump_all([data], Dumper=dumper, **kw)
