import copy
import dataclasses
import inspect
import itertools
import json
import os.path
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, TypedDict, TypeGuard, TypeVar

import flax
import flax.linen
import flax.struct
import hydra_zen
import pydantic
import pydantic.schema
from hydra.core.object_type import ObjectType
from hydra.core.plugins import Plugins
from hydra.plugins.config_source import ConfigResult, ConfigSource
from omegaconf import DictConfig
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from simple_parsing.docstring import get_attribute_docstring, inspect_getdoc
from simple_parsing.helpers.serialization.serializable import dump_yaml
from simple_parsing.utils import Dataclass, PossiblyNestedDict, is_dataclass_type

from project.utils.env_vars import REPO_ROOTDIR

logger = get_logger(__name__)


class AutoSchemaPlugin(ConfigSource):
    # todo: Perhaps we can make a hydra plugin with the auto-schema stuff?
    def __init__(self, provider: str, path: str) -> None:
        super().__init__(provider=provider, path=path)
        logger.info(f"{provider=}, {path=}")

    @staticmethod
    def scheme() -> str:
        return "auto_schema"

    def load_config(self, config_path: str) -> ConfigResult:
        _name = self._normalize_file_name(config_path)
        raise NotImplementedError(config_path)

    def is_group(self, config_path: str) -> bool:
        raise NotImplementedError(config_path)

    def is_config(self, config_path: str) -> bool:
        raise NotImplementedError(config_path)

    def available(self) -> bool:
        """
        :return: True is this config source is pointing to a valid location
        """
        return True
        raise NotImplementedError()

    def list(self, config_path: str, results_filter: ObjectType | None) -> list[str]:
        """List items under the specified config path.

        :param config_path: config path to list items in, examples: "", "foo", "foo/bar"
        :param results_filter: None for all, GROUP for groups only and CONFIG for configs only
        :return: a list of config or group identifiers (sorted and unique)
        """
        raise NotImplementedError(config_path, results_filter)


# def register_auto_schema_plugin() -> None:
Plugins.instance().register(AutoSchemaPlugin)


def add_or_update_shemas_for_yaml_configs(
    configs_dir: Path = REPO_ROOTDIR / "project" / "configs",
    schemas_dir: Path = REPO_ROOTDIR / "project" / "configs" / ".schemas",
):
    schemas_dir.mkdir(exist_ok=True)
    (schemas_dir / ".gitkeep").touch()
    for config_file in itertools.chain(configs_dir.rglob("*.yaml"), configs_dir.rglob("*.yml")):
        if "_target_" not in config_file.read_text():
            continue

        schema_path = schemas_dir / f"{config_file.stem}_schema.json"
        try:
            add_schema_to_hydra_config_file(
                input_file=config_file, output_file=config_file, schema_file=schema_path
            )
        except Exception as e:
            logger.info(f"Unable to update the schema for yaml config file {config_file}: {e}")
        else:
            logger.info(f"Updated schema for {config_file}.")


def add_schema_header(config_file: Path, schema_path: Path) -> None:
    input_lines = config_file.read_text().splitlines(keepends=True)
    relative_path_to_schema = os.path.relpath(schema_path, start=config_file.parent)
    new_first_line = f"# yaml-language-server: $schema={relative_path_to_schema}\n"
    # todo; remove leading empty lines.
    if input_lines[0].startswith("# yaml-language-server: $schema="):
        output_lines = [new_first_line, *input_lines[1:]]
    else:
        output_lines = [new_first_line, *input_lines]

    with config_file.open("w") as f:
        f.writelines(output_lines)


def add_schema_to_hydra_config_file(
    input_file: Path, output_file: Path, schema_file: Path
) -> bool:
    schema = get_schema(input_file)
    schema_file.write_text(json.dumps(schema, indent=2) + "\n")
    add_schema_header(input_file, schema_path=schema_file)
    return True


CONFIGS_DIR = REPO_ROOTDIR / "project/configs"


# todo: read https://stackoverflow.com/questions/70639556/is-it-possible-to-use-pydantic-instead-of-dataclasses-in-structured-configs-in-h
def get_schema(input_file: Path):
    # todo: instead of using `load_from_yaml`, we should use something like `get_config_loader()`

    # *config_groups, config_name = input_file.relative_to(CONFIGS_DIR).with_suffix("").parts
    # config_group = "/".join(config_groups)
    # config = get_config_loader().load_configuration(None, overrides=[f"{config_group}={config_name}"])

    config = hydra_zen.load_from_yaml(input_file)
    assert isinstance(config, DictConfig)
    logger.debug(f"Config: {config}")
    # todo: maybe support the case where there's a single entry in the dictionary, which itself has a _target_ key
    if len(config) == 1 and "_target_" in (only_value := next(iter(config.values()))):
        logger.debug(f"Only value: {only_value}")
        raise NotImplementedError("TODO?")

    # todo: this doesn't take `defaults` into account.
    target = hydra_zen.get_target(config)  # type: ignore

    if inspect.isclass(target) and issubclass(target, flax.linen.Module):
        object_type = hydra_zen.builds(
            target,
            populate_full_signature=True,
            hydra_recursive=False,
            hydra_convert="all",
            zen_exclude=["parent"],
            dataclass_name=f"{target.__name__}Config",
        )
    elif dataclasses.is_dataclass(target):
        # The target is a dataclass, so the schema is just the schema of the dataclass.
        object_type = target
    else:
        # The target is a type or callable.
        assert callable(target)
        object_type = hydra_zen.builds(
            target,
            populate_full_signature=True,
            hydra_defaults=config.get("defaults", None),
            hydra_recursive=False,
            hydra_convert="all",
            dataclass_name=f"{target.__name__}Config",
            # zen_wrappers=pydantic_parser,  # unsure if this is how it works?
        )

    json_schema = pydantic.TypeAdapter(object_type).json_schema(
        mode="serialization", schema_generator=MyGenerateJsonSchema
    )

    # Add field docstrings as descriptions in the schema!
    json_schema = _update_schema_with_descriptions(object_type, json_schema=json_schema)

    schema = adapt_schema_for_hydra(input_file, config, json_schema)

    return schema


class PropertySchema(TypedDict, total=False):
    title: str
    type: str
    description: str
    default: Any
    examples: list[str]
    deprecated: bool
    readOnly: bool
    writeOnly: bool


class Schema(TypedDict):
    properties: dict[str, PropertySchema]


def adapt_schema_for_hydra(
    input_file: Path, config: DictConfig, schema_from_pydantic: dict[str, Any]
):
    """TODO: Adapt the schema to be better adapted for Hydra configs.

    TODOs:
    - [ ] defaults should always be accepted as a field.
    - [ ] _partial_ should make it so there are no mandatory fields
    - [ ] Unexpected extra fields should not be allowed
    """
    # TODO: This generated schema does not seem that well-adapted for Hydra, actually.
    schema = copy.deepcopy(schema_from_pydantic)
    if hydra_zen.is_partial_builds(config):
        # todo: add a special marker that allows extra fields?
        schema["required"] = []
    return schema


class MyGenerateJsonSchema(GenerateJsonSchema):
    # def handle_invalid_for_json_schema(
    #     self, schema: core_schema.CoreSchema, error_info: str
    # ) -> JsonSchemaValue:
    #     raise PydanticOmit

    def enum_schema(self, schema: "core_schema.EnumSchema") -> JsonSchemaValue:
        """Generates a JSON schema that matches an Enum value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        enum_type = schema["cls"]
        logger.debug(f"Enum of type {enum_type}")
        import torchvision.models.resnet

        if issubclass(enum_type, torchvision.models.WeightsEnum):

            @dataclasses.dataclass
            class Dummy:
                value: str

            slightly_changed_schema = schema | {
                "members": [Dummy(v.name) for v in schema["members"]]
            }
            return super().enum_schema(slightly_changed_schema)
        return super().enum_schema(schema)


def save_yaml_with_schema_in_vscode_settings(
    dc: Dataclass,
    path: Path,
    repo_root: Path = Path.cwd(),
    generated_schemas_dir: Path | None = None,
    gitignore_schemas: bool = True,
):
    try:
        import pydantic
    except ModuleNotFoundError:
        logger.error("pydantic is required for this feature.")
        raise

    json_schema = pydantic.TypeAdapter(type(dc)).json_schema(mode="serialization")
    # Add field docstrings as descriptions in the schema!
    json_schema = _update_schema_with_descriptions(dc, json_schema=json_schema)

    dc_schema_filename = f"{type(dc).__qualname__}_schema.json"

    if generated_schemas_dir is None:
        # Defaults to saving in a .schemas folder next to the config yaml file.
        generated_schemas_dir = path.parent / ".schemas"
    generated_schemas_dir.mkdir(exist_ok=True, parents=True)

    repo_root, _ = _try_make_relative(repo_root, relative_to=Path.cwd())
    generated_schemas_dir, _ = _try_make_relative(generated_schemas_dir, relative_to=repo_root)

    if gitignore_schemas:
        # Add a .gitignore in the schemas dir so the schema files aren't tracked by git.
        _write_gitignore_file_for_schemas(generated_schemas_dir)

    schema_file = generated_schemas_dir / dc_schema_filename
    schema_file.write_text(json.dumps(json_schema, indent=2) + "\n")

    # We can use a setting in the VsCode editor to associate a schema file with
    # a list of config files.

    vscode_dir = repo_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True, parents=False)
    vscode_settings_file = vscode_dir / "settings.json"
    vscode_settings_file.touch()

    try:
        vscode_settings: dict[str, Any] = json.loads(vscode_settings_file.read_text())
    except json.decoder.JSONDecodeError:
        logger.error("Unable to load the vscode settings file!")
        raise

    yaml_schemas_setting: dict[str, str | list[str]] = vscode_settings.setdefault(
        "yaml.schemas", {}
    )

    schema_key = str(schema_file.relative_to(repo_root))
    try:
        path_to_add = str(path.relative_to(repo_root))
    except ValueError:
        path_to_add = str(path)

    files_associated_with_schema: str | list[str] = yaml_schemas_setting.get(schema_key, [])
    if isinstance(files_associated_with_schema, str):
        existing_value = files_associated_with_schema
        files_associated_with_schema = sorted(set([existing_value, path_to_add]))
    else:
        files_associated_with_schema = sorted(set(files_associated_with_schema + [path_to_add]))
    yaml_schemas_setting[schema_key] = files_associated_with_schema

    vscode_settings_file.write_text(json.dumps(vscode_settings, indent=2))
    return schema_file


def _write_yaml_with_schema_header(dc: Dataclass, path: Path, schema_path: Path):
    with path.open("w") as f:
        f.write(f"# yaml-language-server: $schema={schema_path}\n")
        dump_yaml(dc, f)


def _try_make_relative(p: Path, relative_to: Path) -> tuple[Path, bool]:
    try:
        return p.relative_to(relative_to), True
    except ValueError:
        return p, False


def _write_gitignore_file_for_schemas(generated_schemas_dir: Path):
    gitignore_file = generated_schemas_dir / ".gitignore"
    if gitignore_file.exists():
        gitignore_entries = [
            stripped_line
            for line in gitignore_file.read_text().splitlines()
            if (stripped_line := line.strip())
        ]
    else:
        gitignore_entries = []
    schema_filename_pattern = "*_schema.json"
    if schema_filename_pattern not in gitignore_entries:
        gitignore_entries.append(schema_filename_pattern)
    gitignore_file.write_text("\n".join(gitignore_entries) + "\n")


def _has_default_dataclass_docstring(dc_type: type[Dataclass]) -> bool:
    docstring: str | None = inspect_getdoc(dc_type)
    return bool(docstring) and docstring.startswith(f"{dc_type.__name__}(")


def _get_dc_type_with_name(dataclass_name: str) -> type[Dataclass] | None:
    # Get the dataclass type has this classname.
    frame = inspect.currentframe()
    assert frame
    for frame_info in inspect.getouterframes(frame):
        if is_dataclass_type(definition_dc_type := frame_info.frame.f_globals.get(dataclass_name)):
            return definition_dc_type
    return None


def _update_schema_with_descriptions(
    object_type: type,
    json_schema: PossiblyNestedDict[str, str | list[str]],
    inplace: bool = True,
):
    if not inplace:
        json_schema = copy.deepcopy(json_schema)

    if "$defs" in json_schema:
        definitions = json_schema["$defs"]
        assert isinstance(definitions, dict)
        for classname, definition in definitions.items():
            if classname == object_type.__name__:
                definition_dc_type = object_type
            else:
                # Get the dataclass type has this classname.
                frame = inspect.currentframe()
                assert frame
                definition_dc_type = _get_dc_type_with_name(classname)
                if not definition_dc_type:
                    logger.debug(
                        f"Unable to find the dataclass type for {classname} in the caller globals."
                        f"Not adding descriptions for this dataclass."
                    )
                    continue

            assert isinstance(definition, dict)
            _update_definition_in_schema_using_dc(definition, dc_type=definition_dc_type)

    if "properties" in json_schema:
        _update_definition_in_schema_using_dc(json_schema, dc_type=object_type)

    return json_schema


K = TypeVar("K")
V = TypeVar("V")


def is_possibly_nested_dict(
    some_dict: Any, k_type: type[K], v_type: type[V]
) -> TypeGuard[PossiblyNestedDict[K, V]]:
    return isinstance(some_dict, dict) and all(
        isinstance(k, k_type)
        and (isinstance(v, v_type) or is_possibly_nested_dict(v, k_type, v_type))
        for k, v in some_dict.items()
    )


def _update_definition_in_schema_using_dc(definition: dict[str, Any], dc_type: type[Dataclass]):
    # If the class has a docstring that isn't the default one generated by dataclasses, add a
    # description.
    docstring = inspect_getdoc(dc_type)
    if docstring is not None and not _has_default_dataclass_docstring(dc_type):
        definition.setdefault("description", docstring)

    if "properties" not in definition:
        # Maybe a dataclass without any fields?
        return

    assert isinstance(definition["properties"], dict)
    dc_fields = {field.name: field for field in dataclasses.fields(dc_type)}

    for property_name, property_values in definition["properties"].items():
        assert isinstance(property_values, dict)
        # note: here `property_name` is supposed to be a field of the dataclass.
        # double-check just to be sure.
        if property_name not in dc_fields:
            logger.warning(
                RuntimeWarning(
                    "assuming that properties are dataclass fields, but encountered"
                    f"property {property_name} which isn't a field of the dataclass {dc_type}"
                )
            )
            continue
        field_docstring = get_attribute_docstring(dc_type, property_name)
        field_desc = field_docstring.help_string.strip()
        if field_desc:
            property_values.setdefault("description", field_desc)


if __name__ == "__main__":
    import logging

    import rich.logging

    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )

    root_logger = logging.getLogger("project")

    add_or_update_shemas_for_yaml_configs()
