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
        
        self.cumulative_inequalities = None


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
    
    
    def accumulate_inequalitites(self):
        if self.parent.cumulative_inequalities is None:
            self.cumulative_inequalities = self.inequalities
        else:
            self.cumulative_inequalities = np.vstack(
                [self.parent.cumulative_inequalities, self.inequalities]
            )

    
    def _single_variable_bound_contradiction(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-9,
    ) -> bool:
        """
        Detect infeasibility from single-variable implied bounds.

        Returns
        -------
        bool
            True if infeasible, False otherwise.
        """
        m, n = A.shape

        for i in range(n):
            lower = -np.inf
            upper = np.inf

            ai = A[:, i]

            pos = ai > tol
            neg = ai < -tol

            # Upper bounds from a_i > 0
            if np.any(pos):
                upper = min(upper, np.min(b[pos] / ai[pos]))

            # Lower bounds from a_i < 0
            if np.any(neg):
                lower = max(lower, np.max(b[neg] / ai[neg]))

            # Contradiction
            if lower > upper + tol:
                return True

        return False
    
    def _opposing_halfspace_contradiction(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-9,
    ) -> bool:
        """
        Detect constraints of the form:
            a^T x <= b
        -a^T x <= -b - eps
        """
        m = A.shape[0]

        for i in range(m):
            for j in range(i + 1, m):
                if np.allclose(A[i], -A[j], atol=tol):
                    if b[i] + b[j] < -tol:
                        return True

        return False
    
    def cheap_infeasibility_check(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-9,
    ) -> bool:
        """
        Return True if infeasible is CERTAIN.
        """
        if self._single_variable_bound_contradiction(A, b, tol):
            return True

        if self._opposing_halfspace_contradiction(A, b, tol):
            return True

        return False


    def _lp_feasible(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-9,
    ) -> bool:
        """
        Check whether the polyhedron defined by the accumulated inequalities
        is non-empty using linear programming.

        Returns
        -------
        bool
            True if feasible, False otherwise.
        """

        n = A.shape[1]

        # Dummy objective (we only care about feasibility)
        c = np.zeros(n)

        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            # bounds=[(None, None)] * n,
            method="highs",
        )

        return res.success
    
    def is_feasible(self, tol: float = 1e-9) -> bool:
        if self._feasible is not None:
            return self._feasible

        # Parent pruning
        if self.parent is not None and not self.parent.is_feasible(tol):
            self._feasible = False
            return False
        
        A = self.cumulative_inequalities[:,:-1]
        b = self.cumulative_inequalities[:,-1]

        # Cheap contradiction test
        if self._cheap_infeasibility_check(A,b,tol):
            self._feasible = False
            return False

        # Exact LP
        self._feasible = self._lp_feasible(A,b,tol)
        return self._feasible






if __name__ == "__main__":
    pass