import pytest

import wano
from wano.execution.decorator import _clear_registry, _strip_decorator, get_function_code


@pytest.fixture(autouse=True)
def clean_registry():
    _clear_registry()
    yield
    _clear_registry()


def test_registers_function():
    @wano.function(compute="cpu")
    def test_func():
        return 1806

    function_code = get_function_code("test_func")

    assert function_code is not None
    assert b"def test_func" in function_code


def test_get_unknown_function_returns_none():
    assert get_function_code("nonexistent") is None


def test_strip_decorator_removes_wano_line():
    source = '@wano.function(compute="gpu", gpus=2)\ndef train():\n    pass\n'
    assert _strip_decorator(source) == "def train():\n    pass\n"


def test_strip_decorator_no_decorator():
    source = "def train():\n    pass\n"
    assert _strip_decorator(source) == source


def test_registered_code_excludes_decorator():
    @wano.function(compute="gpu", gpus=4)
    def my_train():
        return 42

    code = get_function_code("my_train")
    assert code is not None
    assert b"@wano.function" not in code
    assert b"def my_train" in code


def test_registry_isolation():
    @wano.function(compute="cpu")
    def func_a():
        pass

    assert get_function_code("func_a") is not None
    _clear_registry()
    assert get_function_code("func_a") is None


def test_gpu_metadata_stored():
    @wano.function(compute="gpu", gpus=2)
    def gpu_func():
        pass

    from wano.execution.decorator import _function_registry

    entry = _function_registry.get("gpu_func")
    assert entry is not None
    assert entry["compute"] == "gpu"
    assert entry["gpus"] == 2
