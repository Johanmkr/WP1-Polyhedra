from platform import node
import numpy as np
import itertools
from tqdm import tqdm
class RegionTree:
    def __init__(self, state_dict=None, automatic_build=False):
        self.state_dict = state_dict
        self.root = TreeNode()
        self.root.layer_number = 0
        self.hyperplanes = []
        self.size = []

        if automatic_build:
            self.build_tree()

    def _find_hyperplanes(self):
        weights = []
        biases = []
        for key in self.state_dict:
            if "weight" in key:
                weights.append(self.state_dict[key])
            elif "bias" in key:
                biases.append(self.state_dict[key])
        for W, b in zip(weights, biases):
            self.hyperplanes.append(np.hstack((W, b.reshape(-1,1))))

    def _calculate_projections_for_node(self, node, hp):
        # Find projections D and C
        # Assume node is not root for now
        q_lw = node.activation
        s_lw = self._get_signs_from_activation(q_lw)

        W_l = hp[:,:-1]
        b_l = hp[:,-1]
        
        A_lw_prev = node.parent.projection_matrix
        b_lw_prev = node.parent.intercept_vector
        
        # Find projected slopes
        D = np.diag(s_lw) @ W_l @ A_lw_prev
        # Find projected intercepts
        c = -np.diag(s_lw) @ (W_l @ b_lw_prev + b_l)

        # Store projected hyperplanes
        node.inequalities = np.hstack((D, c.reshape(-1,1)))

        # Recursively find projection matrices for the current node based on the previous node.
        A_lw_current = np.diag(q_lw) @ W_l @ A_lw_prev 
        b_lw_current = np.diag(q_lw) @ (W_l @ b_lw_prev + b_l)
        
        # Set the projection matrix and intercept vector for the current node
        node.projection_matrix = A_lw_current
        node.intercept_vector = b_lw_current
        
    def build_tree(self):
        # Check if state_dict is provided
        if self.state_dict is None:
            raise ValueError("State dictionary is not provided.")
        
        # Find the hyperplanes from the state_dict
        self._find_hyperplanes()
        
        # Defining the root as the input to the neural network. 
        # Input dimension is determined by the first layer's weight matrix
        m = self.hyperplanes[0].shape[1] - 1 # Input dimension
        self.root.projection_matrix = np.eye(m) # Identity matrix of size m for root projection
        self.root.intercept_vector = np.zeros(m) # Zero vector of size m for root intercept
        self.size.append(1)
        
        # Initialize the current layer nodes with the root
        current_layer_nodes = [self.root]
        # Iterate through each layer's hyperplanes
        for layer_index, hp in enumerate(self.hyperplanes):
            self.size.append(0)
            next_layer_nodes = []
            num_neurons = hp.shape[0]
            # Generate all possible activation patterns for the current layer
            for activation_pattern in tqdm(itertools.product([0, 1], repeat=num_neurons), desc=f"Processing Layer {layer_index + 1}/{len(self.hyperplanes)}"):
                activation_pattern = np.array(activation_pattern)
                # Create a new node for each activation pattern
                for parent_node in current_layer_nodes:
                    # Create new node with the current activation pattern
                    new_node = TreeNode(activation=activation_pattern)
                    new_node.layer_number = layer_index + 1
                    self.size[-1] += 1
                    new_node.parent = parent_node
                    # Calculate and set the projection matrix and intercept vector for the new node
                    self._calculate_projections_for_node(new_node, hp)
                    # Add the new node as a child to the parent node
                    parent_node.add_child(new_node)
                    next_layer_nodes.append(new_node)

            # Move to the next layer
            current_layer_nodes = next_layer_nodes
        
    def pass_input_through_tree(self, x, reset=False):
        # Pass input x through the tree to find the corresponding leaf node
        if self.root.projection_matrix is None or self.root.intercept_vector is None:
            raise ValueError("Tree has not been built. Please build the tree before passing input.")
        

        current_node = self.root
            
        while not current_node.is_leaf():
            found_child = False
            for child in current_node.get_children():
                # Get inequalities for the child node
                inequalities = child.inequalities
                if inequalities is None:
                    continue
                A = inequalities[:,:-1]
                b = inequalities[:,-1]
                # Check if x satisfies the inequalities
                if np.all(A @ x <= b):
                    current_node = child
                    found_child = True
                    current_node.counter += 1
                    break
            if not found_child:
                raise ValueError("Input does not belong to any region in the tree.")
    
    def reset_counters(self):
        # Reset counters for all nodes in the tree
        def dfs(node):
            node.counter = 0
            for child in node.get_children():
                dfs(child)
        dfs(self.root)

    def _get_nodes_at_layer(self, layer):
        # Get all nodes at a specific layer
        nodes = []
        def dfs(node):
            if node.layer_number == layer:
                nodes.append(node)
            for child in node.get_children():
                dfs(child)
        dfs(self.root)
        return nodes
    
    def read_off_counters(self):
        # Each node has a counter, I need to read them off for each layer in order to estimate the probability density using these number counts.
        for layer in range(len(self.size)):
            print(f"Layer {layer} has {self.size[layer]} nodes.")
            nodes_in_layer = self._get_nodes_at_layer(layer)
            counters = [node.counter for node in nodes_in_layer]
            print(f"Counters: {counters}")

    # Small utility functions
    def _get_activation_from_signs(self, signs):
        return -(signs - 1) / 2

    def _get_signs_from_activation(self, activation):
        return -2 * activation + 1

    # Get functions
    def get_root(self):
        return self.root
    def get_state_dict(self):
        return self.state_dict
    def get_hyperplanes(self):
        return self.hyperplanes

class TreeNode:
    def __init__(self, activation=None):
        self.activation = activation  # Region
        self.projection_matrix = None # For affine transformation to next layer
        self.intercept_vector = None # For affine transformation to next layer
        self.parent = None # Parent TreeNode
        self.inequalities = None # Inequalities added at this node
        self.layer_number = 0 if self.parent == None else None # Layer index in the network
        self.children = []
        self.counter = 0
        
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
    
    
if __name__=="__main__":
    pass