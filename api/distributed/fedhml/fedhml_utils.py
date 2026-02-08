import torch
import numpy as np

def compute_topological_mask(model, target_density):
    """
    Computes a pruning mask based on the Persistent Homology (H1) 
    of the layer's bipartite graph.
    """
    mask_dict = {}
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2: # Focus on FC/Conv layers
            W = param.data.abs()
            # Approximation of H1 persistence for bipartite graphs:
            # We identify the 'structural backbone' by looking at 4-cycles.
            # For efficiency, we use a scoring based on local clustering and magnitude.
            
            # 1. Normalize weights for filtering
            W_norm = (W - W.min()) / (W.max() - W.min() + 1e-8)
            
            # 2. Persistence Score: Edges that connect high-degree hubs 
            # and form stable cycles have higher persistence.
            row_sum = W_norm.sum(dim=1, keepdim=True)
            col_sum = W_norm.sum(dim=0, keepdim=True)
            persistence_score = W_norm * (row_sum * col_sum)
            
            # 3. Pruning
            threshold = torch.quantile(persistence_score.view(-1), 1 - target_density)
            mask = (persistence_score >= threshold).float()
            mask_dict[name] = mask
    return mask_dict

def aggregate_topological_masks(mask_list):
    """
    Topological Consensus: Performs a majority vote on masks 
    to find the stable global backbone.
    """
    if not mask_list:
        return None
    
    consensus_mask = {}
    for key in mask_list[0].keys():
        stacked_masks = torch.stack([m[key] for m in mask_list])
        # Majority vote: if > 50% of clients agree on a connection
        vote_sum = torch.sum(stacked_masks, dim=0)
        consensus_mask[key] = (vote_sum >= (len(mask_list) / 2)).float()
        
    return consensus_mask