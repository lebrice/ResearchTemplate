from collections.abc import Callable
from typing import Any, TypedDict

from pydantic import TypeAdapter


class PropertySchema(TypedDict):
    title: str
    type: str
    default: Any


class Schema(TypedDict):
    title: str
    type: str
    description: str
    properties: dict[str, PropertySchema]


def get_schema(callable: type | Callable) -> dict:
    schema = TypeAdapter(callable).json_schema(mode="serialziable")
    assert False, schema
