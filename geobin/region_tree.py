from __future__ import annotations
import numpy as np
import itertools
from tqdm import tqdm
from .tree_node import TreeNode
import pandas as pd
from typing import Dict, List, Optional, Tuple, Iterable


class RegionTree:
    """
    A tree structure representing all linear regions of a piecewise-linear 
    (ReLU) neural network. Each node corresponds to a unique activation pattern 
    at a particular layer, and contains:
    
    - Activation pattern
    - Hyperplane inequalities defining its region
    - Affine projection matrices mapping input → layer space
    - Children representing the next-layer regions
    
    Parameters
    ----------
    state_dict : dict, optional
        State dictionary of a fully-connected feedforward PyTorch model. 
        Expected to contain weight and bias matrices in layer order.
    build : bool, optional
        If True, automatically build the region tree during initialization.
    """

    def __init__(self, state_dict: Optional[Dict[str, np.ndarray]] = None, build: bool = False):
        self.state_dict = state_dict
        self.root: TreeNode = TreeNode()
        self.root.layer_number = 0

        self.hyperplanes: List[np.ndarray] = []
        """List of stacked hyperplane matrices for each layer."""

        self.size: List[int] = []
        """Number of regions at each depth of the network."""

        self.number_count_dict: Optional[pd.DataFrame] = None
        """DataFrame summarizing region-wise class counts."""

        if build:
            self.build_tree()

    # ----------------------------------------------------------------------
    # Hyperplane Extraction Utilities
    # ----------------------------------------------------------------------

    def _find_hyperplanes(self) -> None:
        """
        Extract weight and bias hyperplanes from the model's state dictionary.
        
        Notes
        -----
        Produces a list ``self.hyperplanes`` where each entry is a matrix
        shaped (n_neurons, input_dim + 1) concatenating each neuron's:
            [W | b]
        """
        weights, biases = [], []

        for key, val in self.state_dict.items():
            if "weight" in key:
                weights.append(val)
            elif "bias" in key:
                biases.append(val)

        for W, b in zip(weights, biases):
            self.hyperplanes.append(np.hstack((W, b.reshape(-1, 1))))

    # ----------------------------------------------------------------------
    # Node Projection Computation
    # ----------------------------------------------------------------------

    def _calculate_projections_for_node(self, node: TreeNode, hp: np.ndarray) -> None:
        """
        Compute the hyperplane inequalities and updated projection matrices 
        for a given node.

        Parameters
        ----------
        node : TreeNode
            Node whose projection/inequalities will be computed.
        hp : np.ndarray
            Layer hyperplanes of shape (num_neurons, prev_dim + 1).

        Notes
        -----
        Implements the ReLU linear-region projection equations:

            D = diag(signs) @ W @ A_prev
            c = -diag(signs) @ (W @ b_prev + b)

            A_curr = diag(activation) @ W @ A_prev
            b_curr = diag(activation) @ (W @ b_prev + b)
        """
        q_lw = node.activation
        s_lw = self._get_signs_from_activation(q_lw)

        W_l = hp[:, :-1]
        b_l = hp[:, -1]

        A_prev = node.parent.projection_matrix
        b_prev = node.parent.intercept_vector

        # Hyperplane slopes/intercepts for this region
        D = np.diag(s_lw) @ W_l @ A_prev
        c = -np.diag(s_lw) @ (W_l @ b_prev + b_l)
        node.inequalities = np.hstack((D, c.reshape(-1, 1)))

        # Updated projection parameters for this region
        node.projection_matrix = np.diag(q_lw) @ W_l @ A_prev
        node.intercept_vector = np.diag(q_lw) @ (W_l @ b_prev + b_l)

    # ----------------------------------------------------------------------
    # Tree Construction
    # ----------------------------------------------------------------------

    def build_tree(self, verbose: bool = False, check_feasibility=True) -> None:
        """
        Build the full region tree from the given `state_dict`.

        Parameters
        ----------
        verbose : bool, optional
            If True, display tqdm progress bars for region enumeration.

        Raises
        ------
        ValueError
            If no state dictionary is provided.

        Notes
        -----
        For each layer, all 2^n activation patterns (q-vectors) are enumerated:
            q ∈ {0, 1}^n

        For each activation pattern and each parent node, a new child node is 
        constructed and its projection matrices are computed.
        """
        if self.state_dict is None:
            raise ValueError("State dictionary is not provided.")

        if verbose:
            print("Building tree...")

        self._find_hyperplanes()

        # Root projections
        m = self.hyperplanes[0].shape[1] - 1
        self.root.projection_matrix = np.eye(m)
        self.root.intercept_vector = np.zeros(m)
        self.size.append(1)

        current_layer_nodes = [self.root]

        for layer_index, hp in enumerate(self.hyperplanes):
            self.size.append(0)
            next_layer_nodes = []
            num_neurons = hp.shape[0]

            q_vectors = itertools.product([0, 1], repeat=num_neurons)
            iterator = tqdm(q_vectors, total=2**num_neurons, desc=f"Layer {layer_index+1}") if verbose else q_vectors

            region_index = 0

            for activation_pattern in iterator:
                activation_pattern = np.array(activation_pattern)

                parents = tqdm(current_layer_nodes, leave=False, desc="Node") if verbose else current_layer_nodes
                for parent_node in parents:
                    new_node = TreeNode(activation=activation_pattern)
                    new_node.layer_number = layer_index + 1
                    new_node.region_index = region_index
                    region_index += 1

                    self.size[-1] += 1

                    new_node.parent = parent_node
                    self._calculate_projections_for_node(new_node, hp)
                    
                    # Feasibility check should go here
                    if check_feasibility:
                        if not new_node.is_feasible():
                            continue
                    
                    parent_node.add_child(new_node)
                    next_layer_nodes.append(new_node)

            current_layer_nodes = next_layer_nodes

    # ----------------------------------------------------------------------
    # Point / Dataloader Traversal
    # ----------------------------------------------------------------------

    def pass_single_point_through_tree(self, point: Tuple[np.ndarray, int]) -> None:
        """
        Pass a single datapoint through the region tree and increment 
        region-wise label counts.

        Parameters
        ----------
        point : tuple
            (x, y) where x is input vector and y is integer class label.

        Notes
        -----
        At each layer, the method selects the unique child whose 
        inequalities satisfy ``A @ x <= b``.
        """
        x, y = point
        current_node = self.root

        while not current_node.is_leaf():
            for child in current_node.get_children():
                ineqs = child.inequalities
                if ineqs is None:
                    continue

                A = ineqs[:, :-1]
                b = ineqs[:, -1]

                if np.all(A @ x <= b):
                    current_node = child
                    current_node.number_counts[str(y)] = current_node.number_counts.get(str(y), 0) + 1
                    break

    def pass_dataloader_through_tree(self, dl: Iterable) -> None:
        """
        Pass an entire PyTorch dataloader through the tree.

        Parameters
        ----------
        dl : iterable
            PyTorch DataLoader producing batches of (inputs, labels).

        Raises
        ------
        ValueError
            If the tree has not been built.

        Notes
        -----
        Converts tensors to numpy arrays before passing to the tree.
        """
        if self.root.projection_matrix is None or self.root.intercept_vector is None:
            raise ValueError("Tree has not been built.")

        for inputs, labels in dl:
            for x, y in zip(inputs, labels):
                self.pass_single_point_through_tree((x.numpy(), int(y.numpy())))


    def pass_dataloader_through_tree_batchwise(self, dl: Iterable) -> None:
        """
        Pass an entire PyTorch dataloader through the region tree in a
        batch-wise, vectorized manner.

        Parameters
        ----------
        dl : iterable
            PyTorch DataLoader producing batches of (inputs, labels).

        Notes
        -----
        At each node, the batch is split among children according to
        which inequalities are satisfied.
        """
        assert True, "This function is not working properly yet"
        
        #FIXME
        
        if self.root.projection_matrix is None:
            raise ValueError("Tree has not been built.")

        for inputs, labels in dl:
            X = inputs.numpy()      # shape: (B, d)
            Y = labels.numpy()      # shape: (B,)

            # Each entry: (node, X_subset, Y_subset)
            stack = [(self.root, X, Y)]

            while stack:
                node, X_node, Y_node = stack.pop()

                if X_node.shape[0] == 0:
                    continue

                # If leaf, accumulate counts and stop
                if node.is_leaf():
                    for y in Y_node:
                        key = str(int(y))
                        node.number_counts[key] = node.number_counts.get(key, 0) + 1
                    continue

                # Otherwise split batch among children
                for child in node.get_children():
                    if child.inequalities is None:
                        continue

                    A = child.inequalities[:, :-1]   # (m, d)
                    b = child.inequalities[:, -1]    # (m,)

                    # Vectorized inequality check
                    mask = np.all(X_node @ A.T < b, axis=1)

                    if not np.any(mask):
                        continue

                    X_child = X_node[mask]
                    Y_child = Y_node[mask]

                    # Update counts immediately if desired
                    for y in Y_child:
                        key = str(int(y))
                        child.number_counts[key] = child.number_counts.get(key, 0) + 1

                    stack.append((child, X_child, Y_child))






    # ----------------------------------------------------------------------
    # Counter Reset / Aggregation
    # ----------------------------------------------------------------------

    def reset_counters(self) -> None:
        """
        Reset class counters on all nodes.
        """
        self.number_count_dict = None

        def reset(node: TreeNode):
            node.number_counts = {}
            for child in node.get_children():
                reset(child)

        reset(self.root)

    def collect_number_counts(self) -> None:
        """
        Traverse the entire tree and collect all region counter statistics.

        Produces a pandas DataFrame with columns:
            [layer_idx, region_idx, <class_counts...>, total]

        Only regions with nonzero counts are retained.
        """
        rows = []

        def _collect(node: TreeNode):
            rows.append({
                "layer_idx": node.layer_number,
                "region_idx": node.region_index,
                **node.number_counts,
            })
            for child in node.get_children():
                _collect(child)

        _collect(self.root)

        frame = pd.DataFrame(rows)
        classes = frame.columns[2:]  # All class labels

        for c in classes:
            frame[c] = np.nan_to_num(frame[c], nan=0)

        # Create "total" column
        frame["total"] = frame[classes].sum(axis=1)
        
        # Remove layer 0 - corresponds to input layer
        frame = frame[frame["layer_idx"] != 0]
        
        # self.number_count_dict = frame[frame["total"] > 0] # Cancel empty regions
        self.number_count_dict = frame # Include empty regions, but remove layer 0 (input)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_activation_from_signs(self, signs: np.ndarray) -> np.ndarray:
        """Convert sign vector {-1, 1} to activation vector {0, 1}."""
        return -(signs - 1) / 2

    def _get_signs_from_activation(self, activation: np.ndarray) -> np.ndarray:
        """Convert activation vector {0, 1} to sign vector {-1, 1}."""
        return -2 * activation + 1

    # ----------------------------------------------------------------------
    # Getters
    # ----------------------------------------------------------------------

    def get_number_counts(self) -> Optional[pd.DataFrame]:
        """Return collected class counts for all regions."""
        return self.number_count_dict

    def get_root(self) -> TreeNode:
        """Return the tree root."""
        return self.root

    def get_state_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the model's state dictionary."""
        return self.state_dict

    def get_hyperplanes(self) -> List[np.ndarray]:
        """Return hyperplanes defining the neural network layers."""
        return self.hyperplanes


if __name__ == "__main__":
    pass
