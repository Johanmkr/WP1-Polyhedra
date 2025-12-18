# mypackage/__init__.py

# Mapping of public names to their source modules
_lazy_attributes = {
    # from latent_representation.py
    "VisualisationOfLatenRepresentation": "latent_representations.py",
}

# Dynamically define __all__ based on lazy attributes
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
