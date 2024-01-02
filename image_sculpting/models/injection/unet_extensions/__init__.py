__modules__ = {}

def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls
    return decorator

def find(name):
    return __modules__[name]

from . import (
    attention,
    resnet_block_2d,
)