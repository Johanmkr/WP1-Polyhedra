import numpy as np
from geobin_py.region import Region
from geobin_py.utils import find_hyperplanes
from geobin_py.construction import construct_tree

class Tree:
    def __init__(self, state_dict):
        # Find the hyperplanes from the state dict
        self.W, self.b = find_hyperplanes(state_dict)
        self.input_dim = self.W[0].shape[1]  # Assuming all hyperplanes have the same input dimension
        self.L = len(self.W)  # Number of layers
        
        # Initialize the root region
        self.root = Region(input_dim = self.input_dim)

    # ----------------------------------------------------------------------
    # External methods
    # ----------------------------------------------------------------------

    # Tree construction
    def construct_tree(self, verbose=False):
        construct_tree(self, verbose=verbose)
    
    def get_regions_at_layer(self, layer: int):
        regions = []
        queue = [self.root]
        
        while queue:
            current_region = queue.pop(0)
            
            if current_region.layer_number == layer:
                regions.append(current_region)
            elif current_region.layer_number < layer:
                queue.extend(current_region.get_children())
        return regions
