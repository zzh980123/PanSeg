import importlib
import sys

import sys


def get_caller(level=1):
    def show(func):
        def wrapper(*args, **kwargs):
            mod = '__main__'
            kwargs['caller_information_'] = mod
            return func(*args, **kwargs)
        return wrapper

    return show


@get_caller(level=1)
def get_module(name=None, **kwargs):
    if name is None:
        name = kwargs.pop('caller_information_')
    mod = sys.modules[name]
    return mod


def import_module(name, package=...):
    return importlib.import_module(name=name, package=package)


def create_instance(class_name, module=None, *args, **kwargs):
    try:
        clazz = getattr(module, class_name)
        instance = clazz(*args, **kwargs)
    except ...:
        return None

    return instance
