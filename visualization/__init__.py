# mypackage/__init__.py
import importlib
# Mapping of public names to their source modules
_lazy_attributes = {
    # from latent_representation.py
    "VisualisationOfLatenRepresentation": ".latent_representations",
    "plot_all_quantities": ".plot_tools",
    "plot_training": ".plot_tools",
    "plot_training_sweep": ".plot_tools",
    "plot_average_KL": ".plot_tools",
    "plot_epoch_layer_grid": ".polygon_plotting",
}

# Dynamically define __all__ based on lazy attributes
__all__ = list(_lazy_attributes.keys())


def __getattr__(name):
    """
    Lazily import attributes from submodules on access.
    """
    if name in _lazy_attributes:
        module = importlib.import_module(_lazy_attributes[name], package=__name__)
        value = getattr(module, name)
        globals()[name] = value  # cache in module namespace
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """
    Include lazily-loaded attributes in dir() calls.
    """
    return sorted(list(globals().keys()) + list(_lazy_attributes.keys()))
