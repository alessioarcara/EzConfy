from enum import Enum
from types import UnionType
from typing import Any, TypeGuard, Union, get_args, get_origin

from pydantic import BaseModel


def is_union(t: object) -> bool:
    # Handles both `Union[A, B]` (typing) and `A | B` (PEP 604 / UnionType)
    origin = get_origin(t)
    return origin is Union or isinstance(t, UnionType)


def is_pydantic_model(annotation: Any) -> TypeGuard[type[BaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel) and annotation is not BaseModel


def is_enum(annotation: Any) -> TypeGuard[type[Enum]]:
    return isinstance(annotation, type) and issubclass(annotation, Enum)


def is_builtin(annotation: Any) -> bool:
    return getattr(annotation, "__module__", None) == "builtins"


def resolve_type(annotation: type[Any]) -> tuple[str, set[tuple[str, str]]]:
    imports: set[tuple[str, str]] = set()
    origin = get_origin(annotation)

    if origin is not None:
        args = get_args(annotation)
        if is_union(annotation):
            non_none_args = [a for a in args if a is not type(None)]
            has_none = type(None) in args
            resolved = []
            for arg in non_none_args:
                arg_str, arg_imports = resolve_type(arg)
                resolved.append(arg_str)
                imports.update(arg_imports)
            type_str = " | ".join(resolved)
            if has_none:
                type_str += " | None"
        else:
            resolved = []
            for arg in args:
                arg_str, arg_imports = resolve_type(arg)
                resolved.append(arg_str)
                imports.update(arg_imports)
            origin_name = getattr(origin, "__name__", str(origin))
            type_str = f"{origin_name}[{', '.join(resolved)}]"
        return type_str, imports

    type_str = annotation.__name__
    if not (is_pydantic_model(annotation) or is_enum(annotation) or is_builtin(annotation)):
        imports.add((annotation.__module__, annotation.__name__))
    return type_str, imports
