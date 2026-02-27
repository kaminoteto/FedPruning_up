import torch

import torch
import torch.nn as nn
import torch.optim as optim

def compute_topological_mask(model, target_density, lambda_etr=0.0):
    """
    Computes a pruning mask based on FedHML with Elastic Topological Rectification (ETR).
    
    Args:
        model: PyTorch model
        target_density: Fraction of weights to retain (e.g., 0.1 = 90% pruning)
        lambda_etr: ETR hyperparameter (0 = pure topology, 1 = pure magnitude)
    
    Returns:
        mask_dict: Dictionary of binary masks for each weight parameter
    """
    mask_dict = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            # Flatten Conv layers to 2D bipartite matrix
            if len(param.shape) == 4:
                W_flat = param.data.view(param.shape[0], -1)
            else:
                W_flat = param.data
            
            W_abs = W_flat.abs()
            if W_abs.max() == 0:
                mask_dict[name] = torch.ones_like(param.data)
                continue
            
            # --------------------------
            # 1. Compute Topological Score (P_topo)
            # --------------------------
            # Normalize weights for topological calculation
            W_norm_topo = (W_abs - W_abs.min()) / (W_abs.max() - W_abs.min() + 1e-8)
            
            # Input Similarity & Cycle Support
            input_similarity = torch.matmul(W_norm_topo.T, W_norm_topo)
            input_similarity.fill_diagonal_(0)
            cycle_support = torch.matmul(W_norm_topo, input_similarity)
            P_topo = W_norm_topo * cycle_support
            
            # --------------------------
            # 2. Compute Magnitude Score (P_mag)
            # --------------------------
            P_mag = W_abs
            
            # --------------------------
            # 3. Elastic Topological Rectification (ETR)
            # --------------------------
            # Normalize both scores to [0, 1]
            def min_max_normalize(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)
            
            P_topo_norm = min_max_normalize(P_topo)
            P_mag_norm = min_max_normalize(P_mag)
            
            # Elastic fusion
            P_elastic = (1 - lambda_etr) * P_topo_norm + lambda_etr * P_mag_norm
            
            # --------------------------
            # 4. Generate Top-K Mask
            # --------------------------
            k = int(P_elastic.numel() * target_density)
            if k == 0:
                threshold = P_elastic.max() + 1.0
            else:
                threshold = torch.kthvalue(P_elastic.view(-1), P_elastic.numel() - k + 1).values
            
            mask = (P_elastic >= threshold).float()
            mask_dict[name] = mask.view(param.shape)
            
    return mask_dict

# def compute_topological_mask(model, target_density):
#     """
#     Computes a pruning mask based on the Persistent Homology (H1) 
#     of the layer's bipartite graph.
    
#     Instead of using simple weight magnitude, this method scores an edge based on 
#     its role in forming '4-cycles' (Algebraic Cycles).
    
#     A 4-cycle in a bipartite graph (Input->Output) represents a robust 
#     logical circuit: Input_A -> Out_1 -> Input_B -> Out_2 -> Input_A.
#     """
#     mask_dict = {}
    
#     for name, param in model.named_parameters():
#         # We only process weight parameters (skip biases)
#         if 'weight' in name and len(param.shape) >= 2:
            
#             # Handle Convolutional Layers:
#             # Flatten to 2D Bipartite Matrix: (Out_Channels, In_Channels * Kernel_H * Kernel_W)
#             if len(param.shape) == 4:
#                 W_flat = param.data.view(param.shape[0], -1)
#             else:
#                 W_flat = param.data
            
#             # 1. Normalize weights to [0, 1]
#             # We treat the normalized weight as the "probability" or "strength" of connection.
#             # W_abs represents the 'Birth' time in Persistence (stronger = born earlier).
#             W_abs = W_flat.abs()
#             if W_abs.max() == 0:
#                 mask_dict[name] = torch.ones_like(param.data)
#                 continue
                
#             W_norm = (W_abs - W_abs.min()) / (W_abs.max() - W_abs.min() + 1e-8)
            
#             # --- Algebraic Cycle Counting (The "Backbone" Score) ---
            
#             # 2. Compute Input Similarity (Shared Outputs)
#             # This calculates how many outputs two inputs share.
#             # Mathematically: Similarity(Input_i, Input_j) = sum(W[:, i] * W[:, j])
#             # Operation: (In x Out) @ (Out x In) -> (In x In)
#             # Note: We use W_norm.T @ W_norm. 
#             # This effectively walks: Input_i -> Output -> Input_j
#             input_similarity = torch.matmul(W_norm.T, W_norm)
            
#             # Remove self-loops (Input_i -> Output -> Input_i is not a 4-cycle)
#             input_similarity.fill_diagonal_(0)
            
#             # 3. Compute Topological Support for each Edge
#             # We want to score edge (Output_u, Input_v).
#             # This edge is supported if there exists a path: 
#             # Input_v --(similar to)--> Input_k --(connects to)--> Output_u
#             #
#             # Score(u, v) = sum over k of ( Weight(u, k) * Similarity(k, v) )
#             # Operation: (Out x In) @ (In x In) -> (Out x In)
#             cycle_support = torch.matmul(W_norm, input_similarity)
            
#             # 4. Final Persistence Score
#             # In Persistent Homology, a feature's importance is: 
#             # Lifetime = Death (Cycle breaks) - Birth (Edge appears).
#             # Here:
#             # - Birth: Determined by the edge weight itself (W_norm).
#             # - Death: Determined by the existence of the alternative path (cycle_support).
#             # We combine them: An edge is important if it is strong AND part of a strong cycle.
#             persistence_score = W_norm * cycle_support
            
#             # --- Global Thresholding ---
            
#             # Determine the threshold to keep the top 'target_density' percent
#             k = int(persistence_score.numel() * target_density)
            
#             if k == 0:
#                 # Extreme case: prune everything
#                 threshold = persistence_score.max() + 1.0
#             else:
#                 # Get the k-th largest value
#                 # We flatten the score matrix to find the global threshold for this layer
#                 threshold = torch.kthvalue(persistence_score.view(-1), 
#                                            persistence_score.numel() - k + 1).values
            
#             # Generate binary mask
#             mask = (persistence_score >= threshold).float()
            
#             # Reshape back to original dimensions (e.g., for Conv layers)
#             mask_dict[name] = mask.view(param.shape)
            
#     return mask_dict

# def compute_topological_mask(model, target_density, lam=0.5): # Added lam (lambda)
#     mask_dict = {}
#     for name, param in model.named_parameters():
#         if 'weight' in name and len(param.shape) >= 2:
#             W_abs = param.data.abs()
#             W_norm = (W_abs - W_abs.min()) / (W_abs.max() - W_abs.min() + 1e-8)
            
#             # 1. Topological Score (from Innovation 2)
#             row_sum = W_norm.sum(dim=1, keepdim=True)
#             col_sum = W_norm.sum(dim=0, keepdim=True)
#             persistence_score = W_norm * (row_sum * col_sum)
            
#             # 2. Magnitude Score (Standard)
#             magnitude_score = W_norm
            
#             # 3. ETR Fusion (Innovation 4)
#             final_score = lam * persistence_score + (1 - lam) * magnitude_score
            
#             # 4. Pruning
#             threshold = torch.quantile(final_score.view(-1), 1 - target_density)
#             mask = (final_score >= threshold).float()
#             mask_dict[name] = mask
#     return mask_dict

def aggregate_topological_masks(mask_list):
    """
    Topological Consensus: Performs a majority vote on masks 
    received from clients to find the stable global backbone.
    """
    if not mask_list:
        return None
    
    consensus_mask = {}
    # Use the first mask to determine device/keys
    key_sample = list(mask_list[0].keys())[0]
    device = mask_list[0][key_sample].device

    for key in mask_list[0].keys():
        # Stack masks from all clients: Shape (Num_Clients, Layer_Shape...)
        stacked_masks = torch.stack([m[key].to(device) for m in mask_list])
        
        # Calculate vote sum
        vote_sum = torch.sum(stacked_masks, dim=0)
        
        # Majority Vote Rule:
        # If more than half of the clients think this edge is topologically important, keep it.
        # This filters out client-specific noise (local data bias).
        consensus_mask[key] = (vote_sum >= (len(mask_list) / 2)).float()
        
    return consensus_mask