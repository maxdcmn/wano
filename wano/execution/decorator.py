import inspect
import re
from collections.abc import Callable
from functools import wraps

_function_registry: dict[str, dict] = {}


def _strip_decorator(source: str) -> str:
    return re.sub(r"^\s*@wano\.function\(.*?\)\n", "", source, count=1, flags=re.MULTILINE)


def function(compute: str, gpus: int | None = None):
    def decorator(func: Callable):
        try:
            source_code = inspect.getsource(func)
        except OSError as err:
            raise ValueError(
                f"Cannot serialize function {func.__name__}: source code not available. "
                "Functions must be defined in a file, not interactively."
            ) from err
        clean_source = _strip_decorator(source_code)
        _function_registry[func.__name__] = {
            "function": func,
            "compute": compute,
            "gpus": gpus,
            "code": clean_source.encode("utf-8"),
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_function_code(func_name: str) -> bytes | None:
    code = _function_registry.get(func_name, {}).get("code")
    return code if isinstance(code, bytes) else None


def _clear_registry():
    _function_registry.clear()
