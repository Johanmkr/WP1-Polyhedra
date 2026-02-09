import numpy as np
from tqdm import tqdm
from geobin_py.region import Region
from geobin_py.geometry import get_interior_point_adaptive, find_active_indices, calculate_next_layer_quantities
from geobin_py.utils import to_tuple

def construct_tree(tree, verbose=False):
    current_layer_nodes = [tree.root]

    for i, local_hp in enumerate(tree.hp):

        # Get local layer parameters
        Wl = local_hp[:,:-1] # Local weights
        bl = local_hp[:,-1].reshape(-1,1) # Local biases
        layer = i+1 # local layer number

        next_layer_nodes = []
        iterator = tqdm(current_layer_nodes, desc=f"Processing layer {layer}", leave=False) if verbose else current_layer_nodes

        for parent in iterator:
            # print(parent.layer_number, layer)
            assert parent.layer_number == layer - 1

            new_nodes_info = find_next_layer_region_info(parent.Dlw_active, parent.glw_active, parent.Alw, parent.clw, Wl, bl, layer)

            for act, info in new_nodes_info.items():
                child = Region(np.array(act))
                child.q_tilde = info["q_tilde"]
                child.bounded = info["bounded"]
                child.Dlw = info["Dlw"]
                child.glw = info["glw"]
                # Use list indexing for active constraints
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
    queue = [to_tuple(q0)]
    while queue:
        q = queue.pop(0) # Get next activation
        qlw = np.array(q)

        Dlw, glw, Alw, clw = calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev)

        qi_act, is_bounded = find_active_indices(Dlw, glw, Dlw_active_prev, glw_active_prev)


        for i_act in qi_act:
            q_new = qlw.copy()
            q_new[i_act] ^= 1
            new_key = to_tuple(q_new)

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
