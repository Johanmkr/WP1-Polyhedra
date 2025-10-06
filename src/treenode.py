# Tree class

class Tree:
    def __init__(self, root=None):
        self.root = root
        
        
class TreeNode:
    def __init__(self, activation=None):
        self.activation = activation  # Region
        self.projection_matrix = None # For affine transformation to next layer
        self.intercept_vector = None # For affine transformation to next layer
        self.parent = None # Parent TreeNode
        self.inequalities = None # Inequalities added at this node
        self.children = []
        
    
        