"""Minimal JSON-Schema subset validation for tool arguments.

Tool inputs arrive from language models; they are validated and defaulted
before any permission check or execution. Supported: object/properties/required,
type (string, integer, number, boolean, array, object), enum, default, items,
minimum/maximum.
"""

from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    pass


_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def validate_args(schema: dict[str, Any], args: Any) -> dict[str, Any]:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise SchemaError("arguments must be an object")
    props: dict[str, Any] = schema.get("properties", {})
    out: dict[str, Any] = {}
    for name, value in args.items():
        if name not in props:
            if schema.get("additionalProperties", False):
                out[name] = value
                continue
            raise SchemaError(f"unexpected argument {name!r}")
        out[name] = _check(props[name], value, name)
    for name, prop in props.items():
        if name not in out and "default" in prop:
            out[name] = prop["default"]
    missing = [r for r in schema.get("required", []) if r not in out]
    if missing:
        raise SchemaError(f"missing required argument(s): {', '.join(missing)}")
    return out


def _check(prop: dict[str, Any], value: Any, where: str) -> Any:
    expected = prop.get("type")
    if expected:
        types = _TYPES[expected]
        if expected in ("integer", "number") and isinstance(value, bool):
            raise SchemaError(f"{where}: expected {expected}")
        if expected == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if expected == "number" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if expected == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if not isinstance(value, types):
            raise SchemaError(f"{where}: expected {expected}, got {type(value).__name__}")
    if "enum" in prop and value not in prop["enum"]:
        raise SchemaError(f"{where}: must be one of {prop['enum']}")
    if "minimum" in prop and value < prop["minimum"]:
        raise SchemaError(f"{where}: must be >= {prop['minimum']}")
    if "maximum" in prop and value > prop["maximum"]:
        raise SchemaError(f"{where}: must be <= {prop['maximum']}")
    if expected == "array" and "items" in prop:
        value = [_check(prop["items"], v, f"{where}[]") for v in value]
    return value
