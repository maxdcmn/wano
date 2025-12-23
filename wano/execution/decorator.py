import pickle
from collections.abc import Callable
from functools import wraps

_function_registry = {}


def function(compute: str, gpus: int | None = None):
    def decorator(func: Callable):
        _function_registry[func.__name__] = {
            "function": func,
            "compute": compute,
            "gpus": gpus,
            "code": pickle.dumps(func),
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_function_code(func_name: str) -> bytes | None:
    code = _function_registry.get(func_name, {}).get("code")
    return code if isinstance(code, bytes) else None
