# mypackage/__init__.py

import importlib

# Mapping of public names to their source submodules
_lazy_attributes = {
    "RegionTree": ".region_tree",
    "TreeNode": ".tree_node",
    "Region": ".region",
    "Tree": ".tree",
}

# Dynamically generate __all__ from lazy attributes
__all__ = list(_lazy_attributes.keys())


def __getattr__(name):
    """
    Lazily import attributes from submodules on access.
    """
    if name in _lazy_attributes:
        module = importlib.import_module(
            _lazy_attributes[name],
            package=__name__
        )
        value = getattr(module, name)
        globals()[name] = value  # cache in module namespace
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """
    Include lazily-loaded attributes in dir() calls.
    """
    return sorted(set(globals()) | set(_lazy_attributes))
