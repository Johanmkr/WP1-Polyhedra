from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Optional


class TreeNode:
    """
    A node representing a region in a piecewise-linear tree structure derived
    from a neural network's activation patterns. Each node may store linear
    inequalities, activation patterns, affine transformation parameters, and 
    child nodes corresponding to deeper regions.

    Parameters
    ----------
    activation : np.ndarray, optional
        Activation pattern at this node. Typically a binary vector indicating
        which ReLUs are active. Default is None.
    """

    def __init__(self, activation: Optional[np.ndarray] = None):
        self.activation: Optional[np.ndarray] = activation
        """Activation pattern at this node."""

        self.projection_matrix: Optional[np.ndarray] = None
        """Affine projection matrix for transition to the next layer."""

        self.intercept_vector: Optional[np.ndarray] = None
        """Affine intercept vector for transition to the next layer."""

        self.parent: Optional[TreeNode] = None
        """Parent node in the tree; None if this node is the root."""

        self.inequalities: Optional[np.ndarray] = None
        """
        Linear inequalities defining this region.
        Expected shape: (m, n+1) where A = [:, :-1] and b = [:, -1].
        """

        self.layer_number: int = 0
        """Depth level in the network; 0 for root."""

        self.region_index: int = 0
        """Index of the region at this layer."""

        self.children: List[TreeNode] = []
        """List of child nodes refining this region."""

        self.number_counts: Dict[str, int] = {}
        """Counts of labels (or other identifiers) passing through this node."""
        
        self._feasible: Optional[bool] = None

    # ---------------------------------------------------------

    def add_child(self, child_node: TreeNode) -> None:
        """
        Add a child node and set its parent pointer.

        Parameters
        ----------
        child_node : TreeNode
            The node to add as a child.
        """
        child_node.parent = self
        self.children.append(child_node)

    # ---------------------------------------------------------

    def is_leaf(self) -> bool:
        """
        Check whether this node is a leaf (i.e., has no children).

        Returns
        -------
        bool
            True if the node has no children; False otherwise.
        """
        return len(self.children) == 0

    # ---------------------------------------------------------

    def is_root(self) -> bool:
        """
        Check whether this node is the root of the tree.

        Returns
        -------
        bool
            True if this node has no parent; False otherwise.
        """
        return self.parent is None

    # ---------------------------------------------------------

    def get_children(self) -> List[TreeNode]:
        """
        Return the list of children of this node.

        Returns
        -------
        list of TreeNode
            Child nodes of this node.
        """
        return self.children

    # ---------------------------------------------------------

    def get_depth(self) -> int:
        """
        Compute the depth of this node in the tree.

        Returns
        -------
        int
            Depth measured as the number of edges from the root.

        Raises
        ------
        AssertionError
            If the computed depth does not match `self.layer_number`.
        """
        depth = 0
        node = self

        while node.parent is not None:
            node = node.parent
            depth += 1

        assert depth == self.layer_number, "Layer number mismatch"
        return depth

    # ---------------------------------------------------------

    def get_ancestors(self) -> List[TreeNode]:
        """
        Return the list of ancestor nodes in root-to-leaf order.

        Returns
        -------
        list of TreeNode
            All ancestors of this node, starting from the root.
        """
        ancestors = []
        node = self

        while node.parent is not None:
            ancestors.append(node.parent)
            node = node.parent

        return ancestors[::-1]

    # ---------------------------------------------------------

    def get_path_inequalities(self) -> Optional[np.ndarray]:
        """
        Retrieve all inequalities accumulated along the path from the root
        to this node (including this node).

        The inequalities are stacked in root-to-leaf order.

        Returns
        -------
        np.ndarray or None
            A (k, n+1) stacked inequality matrix if inequalities exist, 
            otherwise None.
        """
        inequalities = []
        node = self

        while node.parent is not None:
            if node.inequalities is not None:
                inequalities.append(node.inequalities)
            node = node.parent

        return np.vstack(inequalities[::-1]) if inequalities else None

    def _lp_feasible(self, tol: float = 1e-9) -> bool:
        """
        Check whether the polyhedron defined by the accumulated inequalities
        is non-empty using linear programming.

        Returns
        -------
        bool
            True if feasible, False otherwise.
        """
        ineqs = self.get_path_inequalities()
        if ineqs is None:
            return True  # no constraints ⇒ whole space

        A = ineqs[:, :-1]
        b = ineqs[:, -1]

        n = A.shape[1]

        # Dummy objective (we only care about feasibility)
        c = np.zeros(n)

        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=[(None, None)] * n,
            method="highs",
        )

        return res.success
    
    def is_feasible(self, tol: float = 1e-9) -> bool:
        return self._lp_feasible()




if __name__ == "__main__":
    pass