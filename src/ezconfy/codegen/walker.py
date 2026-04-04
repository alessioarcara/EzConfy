from itertools import chain
from typing import Any, get_args, get_origin

from ezconfy.codegen.extractors import Extractor


def walk_schema(annotation: Any, extractors: list[Extractor]) -> None:
    seen: set[int] = set()

    def _visit(current: Any) -> None:
        origin = get_origin(current)
        if origin is not None:
            for arg in get_args(current):
                _visit(arg)
            return

        if not isinstance(current, type):
            return

        current_id = id(current)
        if current_id in seen:
            return
        seen.add(current_id)

        for child in chain.from_iterable(e.children(current) for e in extractors):
            _visit(child)

        for extractor in extractors:
            extractor.extract(current)

    _visit(annotation)
