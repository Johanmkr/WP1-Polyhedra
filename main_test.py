from dataclasses import dataclass
from sklearn import tree
import torch
import numpy as np
import torch.nn as nn
import itertools
import cvxpy as cp
from tqdm import tqdm
import bigtree
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull


# Define a simple neural network using PyTorch
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, classification=False):
        super(NeuralNet, self).__init__()
        self.hidden_sizes = hidden_sizes

        for i in range(len(hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            if i == 0:
                setattr(self, layer_name, nn.Linear(input_size, hidden_sizes[i]))
                setattr(self, relu_name, nn.ReLU())
            else:
                setattr(
                    self, layer_name, nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
                )
                setattr(self, relu_name, nn.ReLU())

        output_layer_name = f"l{len(hidden_sizes) + 1}"

        setattr(self, output_layer_name, nn.Linear(hidden_sizes[-1], num_classes))
        # setattr(self, "output_activation", nn.Softmax(dim=num_classes - 1) if classification else nn.Identity())
        
    def forward(self, x):
        out = x
        for i in range(len(self.hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            out = getattr(self, layer_name)(out)
            out = getattr(self, relu_name)(out)

        output_layer_name = f"l{len(self.hidden_sizes) + 1}"
        out = getattr(self, output_layer_name)(out)
        # out = getattr(self, "output_activation")(out)
        # if len(out.shape) == 1:
        #     out = out.unsqueeze(0)
        # elif len(out.shape) == 2 and out.shape[0] == 1:
        #     out = out.squeeze(0)
        # elif len(out.shape) == 3 and out.shape[0] == 1:
        #     out = out.squeeze(0)
        return out



# Extract hyperplanes from the neural network through its state_dict
def _get_hyperplanes(state_dict:dict):
    weights = []
    biases = []
    for key in state_dict:
        if "weight" in key:
            weights.append(state_dict[key])
        elif "bias" in key:
            biases.append(state_dict[key].unsqueeze(1))

    hyperplanes = []
    for i in range(len(weights) - 1):
        W = weights[i]
        b = biases[i]
        hp = torch.hstack((W, b))
        hyperplanes.append(hp.numpy())
    return hyperplanes


# Get activation or sign given the other
def _get_activation_from_signs(signs:torch.tensor):
    return -(signs - 1) / 2
def _get_signs_from_activation(activation:torch.tensor):
    return -2 * activation + 1

# Checking feasibility of a given activation pattern. Solves Ax <= c using gradient descent.
def check_feasibility_torch(A, c, max_iters=200, tol=1e-4, input_bound=None, device="cpu"):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.float32)

    m = A.shape[1]  # input dimension
    x = torch.randn((m, 1), requires_grad=True, device=device) # Random initialization that can be optimized
    if device is not None:
        A = A.to(device)
        c = c.to(device)
    optimizer = torch.optim.Adam([x], lr=0.05)

    for _ in range(max_iters):
        # Original inequality Ax<=c, such that any positive contribution of the following breaks the inequalities. 
        constraint_violations = A @ x - c
        
        if input_bound is not None:
            # Bounding box constraints: -bound <= x <= bound
            lower_bounds = torch.relu(-x - input_bound)
            upper_bounds = torch.relu(x - input_bound)
            
            loss = constraint_violations.sum() + lower_bounds.sum() + upper_bounds.sum()
        else:
            loss = torch.relu(constraint_violations).sum()
            
        if loss.item() < tol:
            return True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return False



# Find feasible activations for each layer
def _find_feasible_activations(hyperplanes:list):
    feasible_activations = {}
    for i, hp in enumerate(hyperplanes):
        layer_name = f"l_{i+1}"
        num_neurons = hp.shape[0]
        for activation_pattern in itertools.product([0, 1], repeat=num_neurons):
            signs = _get_signs_from_activation(torch.tensor(activation_pattern)).unsqueeze(1).numpy()
            A = signs * hp[:, :-1]
            c = -signs * hp[:, -1:]
            if check_feasibility_torch(A, c):
                if layer_name not in feasible_activations:
                    feasible_activations[layer_name] = []
                feasible_activations[layer_name].append(activation_pattern)
    return feasible_activations


# Construct tree of feasible activations
def _construct_feasible_activation_tree(feasible_activations: dict):
    root = bigtree.Node("root", activation_patterns=None, slope_projection=None, intercept_projection=None, region_inequalities=None)
    
    # Start with the root as the only node in the current layer
    current_layer_nodes = [root]
    
    # Sort layers by key to maintain order
    for layer in sorted(feasible_activations.keys()):
        activations = feasible_activations[layer]
        next_layer_nodes = []
        
        # For each node in the current layer, attach all activations as children
        for parent_node in current_layer_nodes:
            for activation in activations:
                node = bigtree.Node(str(activation), parent=parent_node, activation_patterns=activation, slope_projection=None, intercept_projection=None, region_inequalities=None)
                next_layer_nodes.append(node)
        
        # Move to the next layer
        current_layer_nodes = next_layer_nodes
    
    return bigtree.Tree(root)


def _find_halfspace_intersection(inequalities):
    # inequalities is of shape (num_inequalities, dimension + 1), where the last column is the offset
    
    # Remove zero rows
    inequalities = inequalities[~np.all(inequalities == 0, axis=1)]
    if inequalities.shape[0] == 0:
        raise ValueError("No inequalities provided.")
    
    halfspaces = np.hstack((inequalities[:, :-1], -inequalities[:, -1:]))  # Convert Ax <= b to [A | -b] for HalfspaceIntersection
    
    # Find a feasible point using linear programming
    # Maximize t subject to A x + t <= b
    # This pushes x away from the boundary
    A_ub = np.hstack([halfspaces[:, :-1], np.ones((halfspaces.shape[0], 1))])
    b_ub = -halfspaces[:, -1]
    c = np.zeros(A_ub.shape[1])
    c[-1] = -1  # maximize t (minimize -t)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    # feasible_point = res.x[:-1]
    if not res.success:
        raise ValueError("No feasible point found for the given inequalities.")
    feasible_point = res.x[:-1]

    # Print a lot of info
    print("Feasible point found:", feasible_point)
    print("Halfspaces shape:", halfspaces.shape)
    print("Halfspaces:", halfspaces)
    
    
    
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    return hs

def _fill_tree_with_projection_matrices(tree, hyperplanes, feasible_activations):
    # Initialize root (same as before)
    root = [node for node in tree.levelorder_iter()][0]  # root node
    root.slope_projection = np.eye(hyperplanes[0].shape[1] - 1)
    root.intercept_projection = np.zeros((hyperplanes[0].shape[1] - 1, 1))
    root.region_inequalities = np.zeros((1, hyperplanes[0].shape[1] + 1))  # trivial

    # Start with the root as the only node to process at layer 0
    current_nodes = [root]

    # Loop through each layer of the network (correspond to hyperplanes)
    for layer_index, hp in enumerate(hyperplanes):
        layer_idx = layer_index + 1
        next_nodes = []

        for node in tqdm(current_nodes, desc=f"Processing layer {layer_idx}"):
            # Iterate over the parent's children (snapshot list so we can remove safely)
            node = tree[node_name].node
            # activation_pattern stored in node_name as string; convert back
            activation_pattern = np.array(eval(node.node_name)).reshape(-1, 1)

            # Compute inequalities / projections using node.parent projections
            signs = _get_signs_from_activation(activation_pattern).flatten()
            D = np.diag(signs) @ hp[:, :-1] @ node.parent.slope_projection
            c = -np.diag(signs) @ (hp[:, :-1] @ node.parent.intercept_projection + hp[:, -1:])

            # Combine with parent's region inequalities
            node.region_inequalities = np.vstack((np.hstack((D, c)), node.parent.region_inequalities))

            # Slope and intercept projections for this child
            node.slope_projection = np.diag(activation_pattern.flatten()) @ hp[:, :-1] @ node.parent.slope_projection
            node.intercept_projection = np.diag(activation_pattern.flatten()) @ (hp[:, :-1] @ node.parent.intercept_projection + hp[:, -1:])

            # Test feasibility. If infeasible, remove the child subtree by removing this child.
            try:
                # skip the parent's trivial inequality (index 0) if needed; your original used [1:]
                _find_halfspace_intersection(node.region_inequalities[1:])
            except ValueError:
                # Remove the infeasible child (and thus its entire subtree)
                node.parent.remove_child(node)
                continue

            # Only append to next_nodes if child is still present (feasible)
            next_nodes.append(node)

        # Move down one layer
        current_nodes = next_nodes



# def _fill_tree_with_projection_matrices(tree, hyperplanes, feasible_activations):
#     # Initialize projection matrices for the root node
#     # tree["root"].slope_projection = np.eye(hyperplanes[0].shape[1] - 1)  # Identity matrix for input layer
#     # tree["root"].intercept_projection = np.zeros((hyperplanes[0].shape[1] - 1, 1))  # Zero vector for input layer

#     # Initialize root (same as before)
#     root = [node for node in tree.levelorder_iter()][0]  # root node
#     root.slope_projection = np.eye(hyperplanes[0].shape[1] - 1)
#     root.intercept_projection = np.zeros((hyperplanes[0].shape[1] - 1, 1))
#     root.region_inequalities = np.zeros((1, hyperplanes[0].shape[1] + 1))  # trivial



#     # Loop through each layer of the network (correspond to tree levels)
#     for layer_index, hp in enumerate(hyperplanes):
#         layer_idx = layer_index + 1  # Layer index starts from 1 in the tree, and 0 in hyperplanes (since 0 in tree is the root node). 
#         layer_name = f"l_{layer_idx}"
        
#         # Loop through each node in the current layer
#         for node in tqdm(node_groups_by_level[layer_idx], desc=f"Processing layer {layer_idx}"):
#             # node = tree[node_name].node
#             activation_pattern = np.array(eval(node.node_name)).reshape(-1, 1)  # Convert string back to tuple and then to numpy array
            
#             # Get inequalities for the current region
#             signs = _get_signs_from_activation(activation_pattern).flatten()
#             D = np.diag(signs) @ hp[:, :-1] @ node.parent.slope_projection  # D = diag(s) W^l A_\omega^{l-1}
#             c = -np.diag(signs) @ (hp[:, :-1] @ node.parent.intercept_projection + hp[:, -1:])  # c = diag(s) (W^l b_\omega^{l-1} + b^l)
            
#             # Combine with parent's region inequalities to get full region inequalities
#             node.region_inequalities = np.vstack((np.hstack((D, c)), node.parent.region_inequalities))

#             # Calculate slope projection matrix
#             node.slope_projection = np.diag(activation_pattern.flatten()) @ hp[:, :-1] @ node.parent.slope_projection # A_\omega^l = diag(q_\omega^l) W^l A_\omega^{l-1}
            
#             # Calculate intercept projection matrix
#             node.intercept_projection = np.diag(activation_pattern.flatten()) @ (hp[:,:-1] @ node.parent.intercept_projection + hp[:, -1:]) # b_\omega^l = diag(q_\omega^l) (W^l b_\omega^{l-1} + b^l)
            
#             try:
#                 _find_halfspace_intersection(node.region_inequalities[1:])  # Skip the first trivial inequality and the last one that comes from the parent node.
#             except ValueError:
#                 # Remove current node from the tree if no feasible region
#                 node.parent.remove_child(node)
#                 continue
            


if __name__ == "__main__":
    # Make model
    model = NeuralNet(input_size=2, num_classes=1, hidden_sizes=[4,6,5])
    # Get hyperplanes
    hyperplanes = _get_hyperplanes(model.state_dict())
    for i, hp in enumerate(hyperplanes):
        print(f"Layer {i+1} hyperplanes shape: {hp.shape}")
    # Find feasible activations
    feasible_activations = _find_feasible_activations(hyperplanes)
    i = 0
    for layer, activations in feasible_activations.items():
        # from IPython import embed; embed()
        print(f"{layer}: {len(activations)}/{2**(len(hyperplanes[i]))} feasible activation patterns")
        i += 1
    mytree = _construct_feasible_activation_tree(feasible_activations)
    # tree.vshow()
    
    # for group in [[node.node_name for node in node_group] for node_group in tree.levelordergroup_iter()]:
    #     print(f"Level with {len(group)} nodes: {group}")
        
    _fill_tree_with_projection_matrices(mytree, hyperplanes, feasible_activations)

        
    # Find projections
    # projections = _find_projections(hyperplanes, feasible_activations)
