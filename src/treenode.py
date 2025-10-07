import numpy as np

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
        self.layer_number = 0 if self.parent == None else None # Layer index in the network
        self.children = []
        
    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
        
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
            
        assert depth == self.layer_number, "Layer number mismatch"
        return depth
    
    def get_ancestors(self):
        ancestors = []
        node = self
        while node.parent is not None:
            ancestors.append(node.parent)
            node = node.parent
        return ancestors[::-1]  # Return in root-to-leaf order
    
    def get_path_inequalities(self):
        inequalities = []
        node = self
        while node.parent is not None:
            if node.inequalities is not None:
                inequalities.append(node.inequalities)
            node = node.parent
        return np.vstack(inequalities[::-1]) if inequalities else None  # Return in root-to-leaf order