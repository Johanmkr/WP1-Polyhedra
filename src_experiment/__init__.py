# __all__ = [
#     "get_args", 
#     "NeuralNet", "moons_models", "paths", "createfolders", "train_model", "get_specific_moon_state_dict", "get_df_of_run_summary", "get_df_of_convergence_summary", "get_moon_dataloaders", "get_new_moons_data", "get_new_moons_data_for_all_noises", "get_model", "VisualisationOfLatenRepresentation"]

# from .arg_parser import get_args
# from .utils import NeuralNet, createfolders, get_specific_moon_state_dict, get_df_of_run_summary, get_df_of_convergence_summary
# from .paths import *
# from .dataset import get_new_moons_data, get_new_moons_data_for_all_noises

# from .train_models import train_model  
# from .models import moons_models, get_model
# from .visualize_data_representation import VisualisationOfLatenRepresentation



# mypackage/__init__.py

# Mapping of public names to their source modules
_lazy_attributes = {
    # Argument parser
    "get_args": ".arg_parser",

    # Models
    "moons_models": ".models",
    "get_model": ".models",
    "NeuralNet": ".utils",

    # Dataset
    "get_new_moons_data": ".dataset",
    "get_new_moons_data_for_all_noises": ".dataset",

    # Training
    "train_model": ".train_models",

    # Utilities
    "createfolders": ".utils",
    "get_specific_moon_state_dict": ".utils",
    "get_df_of_run_summary": ".utils",
    "get_df_of_convergence_summary": ".utils",

    # Visualization
    "VisualisationOfLatenRepresentation": ".visualize_data_representation",
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
