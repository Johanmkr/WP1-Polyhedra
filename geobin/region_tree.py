from __future__ import annotations
import numpy as np
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from .tree_node import TreeNode
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
        
        self.test_x = None

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

    # def build_tree(self, verbose: bool = False, check_feasibility=True) -> None:
    #     """
    #     Build the full region tree from the given `state_dict`.

    #     Parameters
    #     ----------
    #     verbose : bool, optional
    #         If True, display tqdm progress bars for region enumeration.

    #     Raises
    #     ------
    #     ValueError
    #         If no state dictionary is provided.

    #     Notes
    #     -----
    #     For each layer, all 2^n activation patterns (q-vectors) are enumerated:
    #         q ∈ {0, 1}^n

    #     For each activation pattern and each parent node, a new child node is 
    #     constructed and its projection matrices are computed.
    #     """
    #     if self.state_dict is None:
    #         raise ValueError("State dictionary is not provided.")

    #     if verbose:
    #         print("Building tree...")

    #     self._find_hyperplanes()

    #     # Root projections
    #     m = self.hyperplanes[0].shape[1] - 1
    #     self.root.projection_matrix = np.eye(m)
    #     self.root.intercept_vector = np.zeros(m)
    #     self.size.append(1)

    #     current_layer_nodes = [self.root]

    #     for layer_index, hp in enumerate(self.hyperplanes):
    #         self.size.append(0)
    #         next_layer_nodes = []
    #         num_neurons = hp.shape[0]

    #         q_vectors = itertools.product([0, 1], repeat=num_neurons)
    #         iterator = tqdm(q_vectors, total=2**num_neurons, desc=f"Layer {layer_index+1}") if verbose else q_vectors

    #         region_index = 0


    #         for activation_pattern in iterator:
    #             parents = tqdm(current_layer_nodes, leave=False, desc="Node") if verbose else current_layer_nodes
    #             for parent_node in parents:
                    
                      
    #         # parents = tqdm(current_layer_nodes, leave=False, desc="Node") if verbose else current_layer_nodes
    #         # for parent_node in parents:
    #         #     for activation_pattern in iterator:
                    
                    
    #                 activation_pattern = np.array(activation_pattern)

    #                 new_node = TreeNode(activation=activation_pattern)
    #                 new_node.layer_number = layer_index + 1
    #                 new_node.region_index = region_index

    #                 new_node.parent = parent_node
    #                 self._calculate_projections_for_node(new_node, hp)
                    
    #                 new_node.accumulate_inequalities()
    #                 # Feasibility check should go here
    #                 if check_feasibility:
    #                     if not new_node.is_feasible():
    #                         continue
                    
    #                 region_index += 1
    #                 self.size[-1] += 1
    #                 parent_node.add_child(new_node)
    #                 next_layer_nodes.append(new_node)

    #         current_layer_nodes = next_layer_nodes


#-------------------------------------------------------------------------------
#       BELOW: Tree builder that loops over parents first, then activations
#-------------------------------------------------------------------------------

    # def build_tree(self, verbose: bool = False, check_feasibility: bool = True) -> None:
    #     """
    #     Build the full region tree from the given `state_dict`.

    #     Parent-first loop order: iterate over each parent, then over all activation patterns.

    #     Parameters
    #     ----------
    #     verbose : bool, optional
    #         If True, display tqdm progress bars.
    #     check_feasibility : bool
    #         Whether to skip infeasible regions.
    #     """
    #     if self.state_dict is None:
    #         raise ValueError("State dictionary is not provided.")

    #     if verbose:
    #         print("Building tree...")

    #     self._find_hyperplanes()

    #     # Root projections
    #     m = self.hyperplanes[0].shape[1] - 1
    #     self.root.projection_matrix = np.eye(m)
    #     self.root.intercept_vector = np.zeros(m)
    #     self.size.append(1)

    #     current_layer_nodes = [self.root]

    #     for layer_index, hp in enumerate(self.hyperplanes):
    #         self.size.append(0)
    #         next_layer_nodes = []
    #         num_neurons = hp.shape[0]

    #         # Generate all activation patterns for this layer
    #         q_vectors = list(itertools.product([0, 1], repeat=num_neurons))
    #         if verbose:
    #             q_iter = tqdm(q_vectors, total=2**num_neurons, desc=f"Layer {layer_index+1} activations")
    #         else:
    #             q_iter = q_vectors

    #         # Parent-first loop
    #         for parent_node in tqdm(current_layer_nodes, desc=f"Layer {layer_index+1} parents", leave=False) if verbose else current_layer_nodes:
    #             feasible_children = []

    #             for activation_pattern in q_iter:
    #                 activation_pattern = np.array(activation_pattern)

    #                 # Create new child node
    #                 new_node = TreeNode(activation=activation_pattern)
    #                 new_node.layer_number = layer_index + 1
    #                 new_node.parent = parent_node

    #                 # Compute projection for this node
    #                 self._calculate_projections_for_node(new_node, hp)

    #                 # Accumulate inequalities (parent-first)
    #                 new_node.accumulate_inequalities()

    #                 # Feasibility check (independent of other children)
    #                 if check_feasibility and not new_node.is_feasible():
    #                     continue

    #                 feasible_children.append(new_node)

    #             # Assign region indices after collecting feasible children
    #             for idx, child in enumerate(feasible_children):
    #                 child.region_index = self.size[-1] + idx
    #                 parent_node.add_child(child)

    #             # Update layer size
    #             self.size[-1] += len(feasible_children)
    #             next_layer_nodes.extend(feasible_children)

    #         current_layer_nodes = next_layer_nodes

#-------------------------------------------------------------------------------
#       BELOW: New method with pruning algorithm. 
#-------------------------------------------------------------------------------

    def _activation_from_point(
        self,
        parent: TreeNode,
        hp: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute activation pattern at this layer for a given input x.
        """
        W = hp[:, :-1]
        b = hp[:, -1]


        z = W @ (parent.projection_matrix @ x + parent.intercept_vector) + b
        return (z > 0).astype(int)


    def _traverse_activation_graph(
        self,
        parent: TreeNode,
        hp: np.ndarray,
    ) -> list[TreeNode]:
        """
        Enumerate all feasible activation patterns reachable from
        an initial pattern by flipping active bits.
        """
        num_neurons = hp.shape[0]

        # Fixed starting point
        x0 = np.random.randint(parent.lower_bounds.max(),parent.upper_bounds.min(), size=self.hyperplanes[1]) # Random sample from input space that is within parent region.

        q0 = self._activation_from_point(parent, hp, x0)
        q0_t = tuple(q0.tolist())

        visited = {q0_t}
        stack = [q0]

        feasible_children = []

        while stack:
            q = stack.pop()

            node = TreeNode(activation=q)
            node.parent = parent

            self._calculate_projections_for_node(node, hp)
            node.accumulate_inequalities()
            node.propagate_bounds()

            # if not node.is_feasible():
            #     continue

            # feasible_children.append(node)

            # Explore neighbors
            active_bits = self.get_active_bits_lp(node, len(q))
            for i in active_bits:
                q_new = q.copy()
                q_new[i] ^= 1
                q_new_t = tuple(q_new.tolist())

                if q_new_t not in visited:
                    visited.add(q_new_t)
                    stack.append(q_new)

        return feasible_children

    def get_active_bits_lp(self, node, num_current_neurons: int, tol: float = 1e-7):
        """
        Return indices (0-based) of active bits for the CURRENT layer only.
        """
        if node.inequalities is None:
            return []

        A = node.inequalities[:, :-1]
        b = node.inequalities[:, -1]

        n_constraints, d = A.shape

        # Indices of current-layer constraints in the accumulated system
        start = n_constraints - num_current_neurons
        end = n_constraints

        active_bits = []

        bounds = [(None, None)] * d

        for global_i in range(start, end):
            res = linprog(
                c=-A[global_i],
                A_ub=A,
                b_ub=b,
                bounds=bounds,
                method="highs",
            )

            if not res.success:
                continue

            max_val = -res.fun

            if abs(max_val - b[global_i]) <= tol:
                # Map global constraint index → local neuron index
                local_i = global_i - start
                active_bits.append(local_i)

        return active_bits


    def build_tree(self, verbose: bool = False, check_feasibility: bool = True) -> None:
        """
        Build the region tree using traversal-and-pruning (Algorithm 2).
        """
        if self.state_dict is None:
            raise ValueError("State dictionary is not provided.")

        if verbose:
            print("Building tree (traversal-and-pruning)...")

        self._find_hyperplanes()

        # Root projections
        m = self.hyperplanes[0].shape[1] - 1
        self.root.projection_matrix = np.eye(m)
        self.root.intercept_vector = np.zeros(m)
        self.root.propagate_bounds()
        self.size = [1]
        # self.test_x = np.ones(self.root.projection_matrix.shape[1])
        current_layer_nodes = [self.root]

        for layer_index, hp in enumerate(self.hyperplanes):
            if verbose:
                print(f"Layer {layer_index + 1}")

            self.size.append(0)
            next_layer_nodes = []

            parents = tqdm(current_layer_nodes, leave=False) if verbose else current_layer_nodes


            for parent in parents:
                # Algorithm 2 happens here
                children = self._traverse_activation_graph(parent, hp)

                for child in children:
                    child.layer_number = layer_index + 1
                    child.region_index = self.size[-1]
                    parent.add_child(child)

                    self.size[-1] += 1
                    next_layer_nodes.append(child)

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
