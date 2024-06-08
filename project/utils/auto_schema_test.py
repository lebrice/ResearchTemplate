class Foo:
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

initial_yaml_contet = """\
__target__: project.utils.auto_schema_test.Foo
some_integer: 42
"""

expected_yaml_content = """\
# yaml-language-server: $schema=.schemas/Foo_schema.json
__target__: project.utils.auto_schema_test.Foo
some_integer: 42
"""


def test_get_schema():
    from project.utils.auto_schema import get_schema

    schema = get_schema(Foo)
    assert schema == expected_schema
