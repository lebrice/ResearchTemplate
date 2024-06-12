from pathlib import Path

from project.configs.datamodule import REPO_ROOTDIR
from project.utils.auto_schema import add_schema_to_hydra_config_file


# @dataclass
class Foo:
    # some_integer: int
    # optional_str: str = "bob"
    def __init__(self, some_integer: int, optional_str: str = "bob"):
        self.some_integer = some_integer
        self.optional_str = optional_str


expected_schema = {
    "title": "Foo",
    "type": "object",
    "description": "Some docstring.",
    "properties": {
        "some_integer": {
            "title": "some_integer",
            "type": "integer",
            #   "description": "A very important field."
        },
        "optional_str": {
            "title": "optional_str",
            "type": "string",
            "default": "bob",
        },
    },
}

initial_yaml_content = """\
_target_: project.utils.auto_schema_test.Foo
some_integer: 42
"""

expected_yaml_content = """\
# yaml-language-server: $schema=SCHEMA_PATH
_target_: project.utils.auto_schema_test.Foo
some_integer: 42
"""


def test_get_schema(tmp_path: Path):
    input_file = tmp_path / "input.yaml"
    input_file.write_text(initial_yaml_content)

    schema_file = tmp_path / ".input_schema.yaml"

    output_file = tmp_path / "output.yaml"
    success = add_schema_to_hydra_config_file(
        input_file=input_file, output_file=output_file, schema_file=schema_file
    )
    assert success
    assert output_file.read_text() == expected_yaml_content.replace(
        "SCHEMA_PATH", schema_file.as_posix()
    )


def test_on_actual_configs():
    config_file = REPO_ROOTDIR / "project" / "configs" / "network" / "resnet50.yaml"
    schema_file = REPO_ROOTDIR / "project" / "configs" / "network" / ".resnet50_schema.json"
    add_schema_to_hydra_config_file(
        config_file, config_file.with_stem("resnet50_with_schema"), schema_file=schema_file
    )

    config_file = REPO_ROOTDIR / "project" / "configs" / "network" / "foo.yaml"
    schema_file = REPO_ROOTDIR / "project" / "configs" / "network" / ".foo_schema.json"
    add_schema_to_hydra_config_file(config_file, config_file, schema_file=schema_file)
