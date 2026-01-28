import numpy as np
from numba import njit
from scipy.optimize import linprog
from tqdm import tqdm
from .region import Region
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor



class Tree:
    def __init__(self, state_dict):
        # Find the hyperplanes from the state dict
        self.hp = find_hyperplanes(state_dict)
        self.input_dim = self.hp[0].shape[1] - 1  # Assuming all hyperplanes have the same input dimension
        self.L = len(self.hp)  # Number of layers
        
        # Initialize the root region
        self.root = Region(input_dim = self.input_dim)

    # ----------------------------------------------------------------------
    # External methods
    # ----------------------------------------------------------------------

    # Tree construction
    def construct_tree(self, verbose=False):
        current_layer_nodes = [self.root]
        
        for i, local_hp in enumerate(self.hp):
            
            # Get local layer parameters
            Wl = local_hp[:,:-1] # Local weights
            bl = local_hp[:,-1].reshape(-1,1) # Local biases
            layer = i+1 # local layer number
            
            next_layer_nodes = []
            for parent in tqdm(current_layer_nodes, desc=f"Processing layer {layer}", leave=False) if verbose else current_layer_nodes:
                # print(parent.layer_number, layer)
                assert parent.layer_number == layer - 1
                
                new_nodes_info = find_next_layer_region_info(parent.Dlw_active, parent.glw_active, parent.Alw, parent.clw, Wl, bl, layer)
                
                for act, info in new_nodes_info.items():
                    child = Region(np.array(act))
                    child.q_tilde = info["q_tilde"]
                    child.bounded = info["bounded"]
                    child.Dlw = info["Dlw"]
                    child.glw = info["glw"]
                    child.Dlw_active = info["Dlw"][list(info["q_tilde"])]
                    child.glw_active = info["glw"][list(info["q_tilde"])]
                    child.Alw = info["Alw"]
                    child.clw = info["clw"]
                    child.layer_number = layer
                    parent.add_child(child)
                    next_layer_nodes.append(child)
                          
            current_layer_nodes = next_layer_nodes
            # Optional: Break if there are no more children to process
            if not current_layer_nodes:
                break
    
    def get_regions_at_layer(self, layer: int):
        regions = []
        queue = [self.root]
        
        while queue:
            current_region = queue.pop(0)
            
            if current_region.layer_number == layer:
                regions.append(current_region)
            elif current_region.layer_number < layer:
                queue.extend(current_region.get_children())
        return regions




# Utility functions
def get_interior_point_adaptive(A, b, initial_slack=0.1, min_threshold=1e-10, verbose=False):
    """
    Finds a point inside Ax <= b. If min_slack is too large, 
    it recursively tries with smaller slack.
    """
    m, n = A.shape
    
    # Ensure inputs are flat 1D arrays to avoid the Dimension/Shape ValueError
    b = np.asarray(b).flatten()
    norms = np.linalg.norm(A, axis=1).flatten()
    c = np.zeros(n)
    
    current_slack = initial_slack
    print(f"Searching for point...") if verbose else None
    while current_slack >= min_threshold:
        # Apply the shrinking logic
        b_shrunk = b - (current_slack * norms)
        
        # Try to solve the LP
        res = linprog(c, A_ub=A, b_ub=b_shrunk, 
                    bounds=[(None, None)] * n, 
                    method='highs')
        
        if res.success:
            print(f"Success found with slack: {current_slack}") if verbose else None
            return res.x
        
        # If infeasible, reduce slack and try again
        print(f"Slack {current_slack} too large, trying {current_slack/10}...") if verbose else None
        current_slack /= 10
        
    # Final attempt: try with zero slack (finds any point on/in the boundary)
    res_final = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method='highs')
    if res_final.success:
        return res_final.x
        
    raise ValueError("Polytope is empty or infeasible even with zero slack.")

def find_active_indices(D_local, g_local, D_global=None, g_global=None, tol=1e-7):
    """
    Identifies which local constraints are active relative to a global domain.
    Combines constraint preparation and LP solving into a single sequential pass.
    """
    
    # TODO Move this into the solver, and have this function accept only D and g
    # --------------------------------------------------------------------------
    D_local = np.asarray(D_local)
    g_local = np.asarray(g_local).flatten()
    n_local, dim = D_local.shape
    
    # Merge local and global constraints into a unified set
    if D_global is not None and g_global is not None:
        D_all = np.vstack([D_local, np.asarray(D_global)])
        g_all = np.concatenate([g_local, np.asarray(g_global).flatten()])
    else:
        D_all, g_all = D_local, g_local
    # --------------------------------------------------------------------------
    
    bounds = [(None, None)] * dim
    active_local_bits = []
    is_bounded = True

    # Iterate through local constraints ONLY
    for i in range(n_local):
        d_i = D_all[i]
        
        # Define the feasible region using all constraints EXCEPT the one we are testing
        D_tilde = np.delete(D_all, i, axis=0)
        g_tilde = np.delete(g_all, i, axis=0)

        try:
            res = linprog(
                c=-d_i, # Maximize distance along the normal of constraint i
                A_ub=D_tilde,
                b_ub=g_tilde,
                bounds=bounds,
                method="highs",
            )
            
            if res.success:
                # Calculate the objective value manually if res.fun is ambiguous
                val = -res.fun if res.fun is not None else np.dot(d_i, res.x)
                
                # If we can push past the boundary, it's a "redundant" or reachable bit
                if val > g_all[i] + tol:
                    active_local_bits.append(i)
            
            elif res.status == 3: # Unbounded: we can go to infinity past this boundary
                active_local_bits.append(i)
                is_bounded = False

        except ValueError as e:
            # Handle specific HiGHS internal status 15 if encountered
            if "HiGHS Status 15" in str(e):
                active_local_bits.append(i)
                is_bounded = False
            else:
                raise e

    return np.array(sorted(active_local_bits)), is_bounded

def find_activation_pattern(x, Wl, bl, Alw_prev, clw_prev):
    z = Wl @ Alw_prev @ x + Wl @ clw_prev + bl
    return (z>0).astype(int)

def to_tuple(array):
    return tuple(array.ravel())

# Hyperplane extraction from state_dict
def find_hyperplanes(state_dict):
    weights = []
    biases = []
    hyperplanes = [] 
    for key, val in state_dict.items():
        if "weight" in key:
            weights.append(val)
        elif "bias" in key:
            biases.append(val)
    for W, b in zip(weights, biases):
        hyperplanes.append(np.hstack((W, b.reshape(-1,1))))
    return hyperplanes

# def calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev):
#     #Old way
#     Slw = np.diag((-2*qlw +1).ravel())
#     Qlw = np.diag(qlw.ravel())
    
#     Wl_hat = np.matmul(Wl, Alw_prev)
#     bl_hat = np.matmul(Wl, clw_prev) + bl
    
#     Dlw = np.matmul(Slw, Wl_hat)
#     glw = -np.matmul(Slw, bl_hat)
    
#     Alw = np.matmul(Qlw, Wl_hat)
#     clw = np.matmul(Qlw, bl_hat)

#     return Dlw, glw, Alw, clw



@njit(cache=True, fastmath=True)
def _jit_core_logic(Wl, bl, qlw, Alw_prev, clw_prev):
    # We cast to float64 inside JIT to ensure mathematical consistency
    qlw_f = qlw.astype(np.float64).reshape(-1, 1)
    bl_2d = bl.reshape(-1, 1).astype(np.float64)
    clw_prev_2d = clw_prev.reshape(-1, 1).astype(np.float64)

    # Combined matmul for speed
    combined_prev = np.column_stack((Alw_prev, clw_prev_2d))
    combined_hat = Wl @ combined_prev
    
    Wl_hat = combined_hat[:, :-1]
    bl_hat = combined_hat[:, -1:] + bl_2d

    s_vec = -2.0 * qlw_f + 1.0
    
    # This is where your shape error happens: 
    # s_vec (from qlw) must match Wl_hat (from Wl rows)
    Dlw = s_vec * Wl_hat
    glw = -(s_vec * bl_hat)
    
    Alw = qlw_f * Wl_hat
    clw = (qlw_f * bl_hat)

    return Dlw, glw, Alw, clw

def calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev):
    # --- DIMENSION SAFETY CHECK ---
    # qlw (bitstring) length must match Wl output (rows)
    if len(qlw) != Wl.shape[0]:
        raise ValueError(f"Dimension Mismatch! Activation qlw has {len(qlw)} bits, "
                         f"but Wl produces {Wl.shape[0]} outputs. "
                         f"Are you using the Wl from the wrong layer?")
    
    # Ensure inputs are numpy arrays
    return _jit_core_logic(
        np.asarray(Wl, dtype=np.float64), 
        np.asarray(bl, dtype=np.float64), 
        np.asarray(qlw, dtype=np.float64), 
        np.asarray(Alw_prev, dtype=np.float64), 
        np.asarray(clw_prev, dtype=np.float64)
    )

def find_next_layer_region_info(Dlw_active_prev, glw_active_prev, Alw_prev, clw_prev, Wl, bl, layer_nr, verbose=False):    
    
    # Random point within region
    if layer_nr != 1:
        x = get_interior_point_adaptive(Dlw_active_prev, glw_active_prev).reshape(-1,1)
    else:
        x = np.random.random((2,1)).reshape(-1,1)
    
    # print(f"{Wl.shape} @ {Alw_prev.shape} @ {x.shape} + {Wl.shape} @ {clw_prev.shape} + {bl.shape}")
    # Get activation of random point
    z = Wl @ Alw_prev @ x + Wl @ clw_prev + bl
    q0 = (z>0).astype(int)
    
  
    traversed = {}
    queue = [tuple(q0.ravel())]
    while queue:
        q = queue.pop(0) # Get next activation
        qlw = np.array(q)
        
        Dlw, glw, Alw, clw = calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev)
        
        qi_act, is_bounded = find_active_indices(Dlw, glw, Dlw_active_prev, glw_active_prev)
        
        
        for i_act in qi_act:
            q_new = qlw.copy()
            q_new[i_act] ^= 1
            new_key = tuple(q_new.ravel())
            
            if new_key not in traversed:
                traversed[new_key] = False
                queue.append(new_key)
                
        traversed[q] = {
            "q_tilde": qi_act,
            "bounded": is_bounded,
            "Dlw": Dlw,
            "glw": glw,
            "Alw": Alw,
            "clw": clw,
        }
    return traversed
    
    