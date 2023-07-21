
import os
from dataclasses import dataclass
import sys, importlib.util
import pickle
import json
import random
from types import ModuleType

def import_config(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    config_name = f"tmp_config_{random.randrange(1000000)}"
    spec = importlib.util.spec_from_loader(config_name, loader = None)
    module = importlib.util.module_from_spec(spec)
    with open(path, "r") as f:
        source = f.read()
    source = source.replace("__file__", f"\"{path}\"")
    exec(source, module.__dict__)
    sys.modules[config_name] = module
    globals()[config_name] = module

    config = Args()
    for key, val in eval(config_name).__dict__.items():
        if not key.startswith("__") and not isinstance(val, type) and not isinstance(val, ModuleType):
            setattr(config, key, val)
    return config

def import_from_string(string):
    temp = string.split(".")
    location = ".".join(temp[:-1])
    name = temp[-1]
    exec(f"import {location}")
    return eval(f"{location}.{name}")

@dataclass
class Options:
    def __init__(self, options):
        self.options = options

    def to_list(self):
        return self.options

    def __str__(self):
        return f"Options({self.options})"

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        res = ["*" * 80] + self.get_list_strs() + ["*" * 80]
        return "\n".join(res)

    def get_list_strs(self, prefix = ""):
        d = self.to_dict()
        res = []
        tmp_prefix = prefix
        for key in d:
            if isinstance(d[key], Args):
                res.extend(d[key].get_list_strs(f"{prefix}{key}."))
            else:
                res.append(f"{tmp_prefix}{key} = {d[key]}")
                tmp_prefix = " " * len(tmp_prefix)
        return res

    def __repr__(self):
        return str(self)

    def to_dict(self):
        return self.__dict__
