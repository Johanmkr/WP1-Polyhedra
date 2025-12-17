# __all__ = ["RegionTree", "TreeNode", "EstimateQuantities1Run", "AveragedEstimates"]

# from .region_tree import RegionTree
# from .tree_node import TreeNode
# from .mi_estimate import EstimateQuantities1Run, AveragedEstimates


# mypackage/__init__.py

# Mapping of public names to their source submodules
_lazy_attributes = {
    "RegionTree": ".region_tree",
    "TreeNode": ".tree_node",
    "EstimateQuantities1Run": ".mi_estimate",
    "AveragedEstimates": ".mi_estimate",
}

# Dynamically generate __all__ from lazy attributes
__all__ = list(_lazy_attributes.keys())


def __getattr__(name):
    """
    Lazily import attributes from submodules on access.
    """
    if name in _lazy_attributes:
        module = __import__(_lazy_attributes[name], globals(), locals(), [name])
        value = getattr(module, name)
        globals()[name] = value  # cache in module namespace
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """
    Include lazily-loaded attributes in dir() calls.
    """
    return sorted(list(globals().keys()) + list(_lazy_attributes.keys()))
