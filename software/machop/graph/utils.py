import inspect
from typing import Tuple


def check_func_type(node, my_func):
    return type(node.target) == type(my_func)


def isinstance_but_not_subclass(my_object, my_class):
    return my_object.__class__ is my_class


def get_parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def get_module_by_name(model, request_name):
    for name, layer in model.named_modules():
        if name == request_name:
            return layer
    return None


def get_node_by_name(graph, request_name):
    for node in graph.nodes:
        if node.name == request_name:
            return node
    raise RuntimeError(f"No node named {request_name} found in graph")


# Verilog format
# Format to a compatible verilog name
def vf(string):
    return string.replace(".", "_").replace(" ", "_")