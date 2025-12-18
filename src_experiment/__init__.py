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
    # from dataset.py
    "get_data": ".dataset.py",
    
    # from divergence_engine.py
    "QUANTITIES_TO_ESTIMATE": ".divergence_engine.py",
    "DivergenceEngine": ".divergence_engine.py",
    
    # from estimate_quanties.py
    "EstimateQuantities1Run": ".estimate_quantities.py",
    "AverageEstimates": ".estimate_quantities.py",
    
    # from models.py
    "get_model": ".models.py",
    
    # from paths.py
    "get_storage_pat": ".paths.py",
    
    # from train_models.py
    "train_model": ".train_models.py",
    
    # from utils.py
    "NeuralNet": ".utils.py",
    "createfolders": ".utils.py",
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
