# Region class as dataclass
import numpy as np


class Region:
    def __init__(self, activation=None, input_dim=None):
        # Region attributes
        self.qlw = activation # Activation pattern
        self.qlw_tilde = None # Active bits in activation pattern (indices)
        self.bounded = None
        
        # Inequalities and projections
        
        
        if input_dim is not None:
            self.Alw = np.eye(input_dim) # Slope projection matrix
            self.clw = np.zeros((input_dim, 1)) # Intercept projection matrix
            
            
            # Generate hypercube inequalities
            # Each dimension has 2 constraints (upper and lower bound)
            # D will have 2n rows and n columns
            self.Dlw  = np.zeros((2 * input_dim, input_dim))
            # g will be a vector of all 1s since |x_i| <= 1
            self.glw  = np.ones((2 * input_dim, 1))
            
            for i in range(input_dim):
                # Constraint for x_i <= 1
                self.Dlw[2 * i, i] = 1
                # Constraint for -x_i <= 1 (which is x_i >= -1)
                self.Dlw[2 * i + 1, i] = -1
            
            
            self.Dlw_active = self.Dlw # Active slopes
            self.glw_active = self.glw # Active intercepts
        else:
            self.Alw = None # Slope projection matrix
            self.clw = None # Intercept projection matrix
            
            self.Dlw = None # Slopes of inequalities
            self.glw = None # Intercept of inequalities 
            
        self.Dlw_active = None # Active slopes
        self.glw_active = None # Active intercepts
        
        # Tree attributes
        self.parent = None # Parent Region object
        self.children = [] # List of children (Region objects)
        
        # Utility attributed
        self.layer_number = 0 # Layer to which this region belongs
        self.region_index = 0 # Index to identify regions #TODO necessary?

        # Estimation attributes
        self.volume_estimate = None
        self.vertices = None
        self.sample_points = {}
        
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def get_children(self):
        return self.children
    
    def get_depth(self):
        depth = 0
        node = self
        while node.parent is not None:
            node = node.parent
            depth += 1
        assert depth == self.layer_number, "Depth and layer number are not equal"
        return depth
    
    def get_ancestors(self):
        ancestors = []
        node = self
        
        while node.parent is not None:
            ancestors.append(node.parent)
            node = node.parent
        
        return ancestors[::-1]
    
    def get_path_inequalities(self):
        D_list = []
        g_list = []
        node = self
        
        while node.parent is not None:
            D_list.append(node.Dlw_active)
            g_list.append(node.glw_active)
            node = node.parent
        
        D_path = np.vstack(D_list[::-1])
        g_path = np.vstack(g_list[::-1])
        
        return D_path, g_path
    
    
    def __str__(self):
        returnstring = f"\n--------------\nRegion info:\n--------------\nLayer: {self.layer_number}\nIndex: {self.region_index}\nLocal activation: {self.qlw}\nInput dim: {self.Alw.shape[1] if self.Alw is not None else 'N/A'}\nChildren: {len(self.children)}"
        if self.is_root():
            returnstring += "\nThis is the root region."
        elif self.is_leaf():
            returnstring += "\nThis is a leaf region."
            returnstring += f"\nAncestors: {len(self.get_ancestors())}"
        return returnstring  
    
if __name__=="__main__":
    pass