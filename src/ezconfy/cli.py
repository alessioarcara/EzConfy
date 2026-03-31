from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import typer
from loguru import logger
from pydantic import BaseModel

from ezconfy.schema_parser import SchemaParser

app = typer.Typer()


def _is_union(t: object) -> bool:
    # Handles both `Union[A, B]` (typing) and `A | B` (PEP 604 / UnionType)
    origin = get_origin(t)
    return origin is Union or isinstance(t, UnionType)


def resolve_type(annotation: type[Any]) -> tuple[str, set[tuple[str, str]]]:
    origin = get_origin(annotation)
    if origin is not None:
        # Generic type: List[X], Dict[K, V], Optional[X], Union[A, B], etc.
        args = get_args(annotation)
        all_imports: set[tuple[str, str]] = set()

        if _is_union(annotation):
            # Emit as `A | B` (PEP 604 style), keeping `None` at the end
            non_none_args = [a for a in args if a is not type(None)]
            has_none = type(None) in args
            resolved = []
            for arg in non_none_args:
                arg_str, arg_imports = resolve_type(arg)
                resolved.append(arg_str)
                all_imports.update(arg_imports)
            result = " | ".join(resolved)
            if has_none:
                result += " | None"
            return (result, all_imports)

        # Non-union generic: recurse into each type argument
        resolved = []
        for arg in args:
            arg_str, arg_imports = resolve_type(arg)
            resolved.append(arg_str)
            all_imports.update(arg_imports)

        origin_name = getattr(origin, "__name__", str(origin))
        result = f"{origin_name}[{', '.join(resolved)}]"
        return (result, all_imports)

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        # Nested Pydantic model — defined in the same generated file, no import needed
        return (annotation.__name__, set())

    if annotation.__module__ == "builtins":
        # Built-in types (int, str, bool, …) need no import
        return (annotation.__name__, set())

    # Any other type lives in a third-party or user module and needs an explicit import
    return (annotation.__name__, {(annotation.__module__, annotation.__name__)})


def collect_nested_models(model: type[BaseModel]) -> list[type[BaseModel]]:
    """
    DFS post-order traversal to collect all nested BaseModel classes used in the given model.
    Leaf models are collected first, then their parents, ensuring correct order for code generation.
    """
    result: list[type[BaseModel]] = []
    seen: set[int] = set()

    def _visit_type(t: Any) -> None:
        origin = get_origin(t)
        # list[X], Optional[X], ... -> unwrap and recurse into args
        if origin is not None:
            for arg in get_args(t):
                _visit_type(arg)
            return
        if isinstance(t, type) and issubclass(t, BaseModel) and t is not BaseModel:
            if id(t) not in seen:
                seen.add(id(t))
                for fi in t.model_fields.values():
                    if fi.annotation is not None:
                        _visit_type(fi.annotation)
                result.append(t)

    for fi in model.model_fields.values():
        if fi.annotation is not None:
            _visit_type(fi.annotation)

    return result


def _emit_class_block(model: type[BaseModel]) -> tuple[list[str], set[tuple[str, str]]]:
    """Return (field_lines, imports) for a single model class."""
    all_imports: set[tuple[str, str]] = set()
    field_lines: list[str] = []

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        if annotation is None:
            logger.error(f"Field '{field_name}' has no type annotation.")
            raise ValueError(f"Field '{field_name}' has no type annotation.")

        type_str, type_imports = resolve_type(annotation)
        all_imports.update(type_imports)

        if field_info.is_required():
            default_str = "..."
        else:
            default_str = repr(field_info.default)

        field_lines.append(f"    {field_name}: {type_str} = Field({default_str})")

    if not field_lines:
        field_lines.append("    pass")

    return field_lines, all_imports


def run_generation(
    schema_path: Path,
    output_path: Path,
    parser: SchemaParser,
) -> None:
    schema_str = schema_path.read_text(encoding="utf-8")
    model = parser.parse(schema_str)

    nested_models = collect_nested_models(model)
    all_models = [*nested_models, model]

    all_imports: set[tuple[str, str]] = set()
    class_blocks: list[tuple[str, list[str]]] = []

    for m in all_models:
        field_lines, imports = _emit_class_block(m)
        all_imports.update(imports)
        class_blocks.append((m.__name__, field_lines))

    # Build import lines
    lines: list[str] = []
    for module, name in sorted(all_imports):
        lines.append(f"from {module} import {name}")
    lines.append("from pydantic import BaseModel, Field")

    # Build class blocks
    for class_name, field_lines in class_blocks:
        lines.append("")
        lines.append("")
        lines.append(f"class {class_name}(BaseModel):")
        lines.extend(field_lines)

    generated_code = "\n".join(lines) + "\n"
    output_path.write_text(generated_code, encoding="utf-8")
    logger.info(f"Code generated successfully at {output_path}")


@app.command()
def generate(
    schema_path: Path = typer.Argument(..., help="Path to the YAML/JSON schema file."),
    output_path: Path = typer.Option(Path("generated.py"), "--output", "-o", help="Target output file."),
) -> None:
    if not schema_path.exists():
        logger.error(f"File not found: {schema_path}")
        raise typer.Exit(1)
    try:
        run_generation(schema_path, output_path, SchemaParser())
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
