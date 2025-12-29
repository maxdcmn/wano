import re
import textwrap

import ray

import wano
from wano.execution.decorator import get_function_code


def test_executes_registered_function(ray_cluster):
    @wano.function(compute="cpu")
    def task():
        return 1806

    code = get_function_code("task")
    assert code is not None
    source = textwrap.dedent(code.decode("utf-8"))
    source = "\n".join([line for line in source.split("\n") if not line.strip().startswith("@")])
    namespace: dict[str, object] = {}
    exec(compile(source, "<string>", "exec"), namespace)
    match = re.search(r"def\s+(\w+)\s*\(", source)
    assert match is not None
    function = namespace[match.group(1)]
    assert callable(function)

    result = ray.get(ray.remote(function).remote())

    assert result == 1806
