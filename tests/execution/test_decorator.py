import wano
from wano.execution.decorator import get_function_code


def test_registers_function():
    @wano.function(compute="cpu")
    def test_func():
        return 1806

    function_code = get_function_code("test_func")

    assert function_code is not None
    assert b"def test_func" in function_code
