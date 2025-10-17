import numpy as np
import itertools
from tqdm import tqdm

from geobin import tree_node as tn

class RegionTree:
    def __init__(self, state_dict=None, automatic_build=False):
        self.state_dict = state_dict
        self.root = tn.TreeNode()
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
        
        print("Building tree...")
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
            # possible_activation_patterns = itertools.product([0, 1], repeat=num_neurons)
            # print(possible_activation_patterns)
            for activation_pattern in tqdm(itertools.product([0, 1], repeat=num_neurons),total=2 ** num_neurons, desc=f"Layer {layer_index+1} / {len(self.hyperplanes)}"):
                activation_pattern = np.array(activation_pattern)
                # Create a new node for each activation pattern
                
                for parent_node in tqdm(current_layer_nodes, desc=f"Node", leave=False):
                    # Create new node with the current activation pattern
                    new_node = tn.TreeNode(activation=activation_pattern)
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
        
    def pass_input_through_tree(self, x, return_path=False, reset=False):
        # Pass input x through the tree to find the corresponding leaf node
        if self.root.projection_matrix is None or self.root.intercept_vector is None:
            raise ValueError("Tree has not been built. Please build the tree before passing input.")
        
        if reset:
            self.reset_counters()
        
        current_node = self.root
        
        path = [current_node]

        while not current_node.is_leaf():
            found_child = False
            # print(f"At layer {current_node.layer_number}, checking children...")
            for child in current_node.get_children():
                # Get inequalities for the child node
                inequalities = child.inequalities
                if inequalities is None:
                    continue
                A = inequalities[:,:-1]
                b = inequalities[:,-1]
                # print(f"Checking child at layer {child.layer_number}")
                # Check if x satisfies the inequalities
                if np.all(A @ x <= b):
                    current_node = child
                    found_child = True
                    path.append(current_node)
                    break
            if not found_child:
                raise ValueError("Input does not belong to any region in the tree.")
        for path_node in path:
            path_node.counter += 1
        if return_path:
            return path

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
        counters = self.get_counters()
        for layer_idx, layer_counters in enumerate(counters):
            print(f"Layer {layer_idx}: {layer_counters}\nTotal: {sum(layer_counters)}\n")
            
    def get_counters(self):
        counters_per_layer = []
        for layer in range(len(self.size)):
            nodes_in_layer = self._get_nodes_at_layer(layer)
            counters = [node.counter for node in nodes_in_layer]
            counters_per_layer.append(counters)
        return counters_per_layer

    def get_nonzero_counter_nodes(self):
        nonzero_counter_nodes = {}
        for layer in range(len(self.size)):
            nodes_in_layer = self._get_nodes_at_layer(layer)
            nonzero_nodes = []
            for node in nodes_in_layer:
                if np.abs(node.counter) < 1e-5:
                    continue
                else:
                    nonzero_nodes.append(node)
            nonzero_counter_nodes[layer] = nonzero_nodes
        return nonzero_counter_nodes
    
    def store_counters(self, reset=True):
        for layer in range(len(self.size)):
            for node in self._get_nodes_at_layer(layer):
                node.number_counts.append(node.counter)
                if reset:
                    node.counter = 0
        
    def get_number_counts(self):
        all_number_counts = {}
        for layer in range(len(self.size)):
            nodes_in_layer = self._get_nodes_at_layer(layer)
            layer_counts = []
            for node in nodes_in_layer:
                if sum(node.number_counts) > 1e-5:
                    layer_counts.append(node.number_counts)
                else:
                    continue
            all_number_counts[layer] = layer_counts
        return all_number_counts
            
        
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
