
import os
import inspect

def filter_inputs(func, inputs):
    func_args = inspect.signature(func).parameters.keys()
    filtered = {key:val for key, val in inputs.items() if key in func_args}
    return filtered
