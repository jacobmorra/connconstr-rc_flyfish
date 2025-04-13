import numpy as np
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import networkx as nx

from tqdm import tqdm, trange

TESTING = 0  # true if using a tiny connectome for testing
THRESHOLD = 1  # if true, use thresholded connectome from file
TOP_300 = 0 # if true, use top 300 connectome as original

def save_connectome_npy(connectome, filename):
    """Save the connectome as a .npy file."""
    np.save(filename, connectome)
def show_connectome(connectome, title_prefix, directed=False):
    # Heatmap of the connectivity matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(connectome, cmap='viridis')
    plt.title(f"{title_prefix} - Connectivity Matrix")
    plt.show()
def rewire_connectome(connectome):
    """
    Rewires the connectome by randomizing the positions of its edges
    without preserving the degree distribution.

    This function extracts the edge weights from the upper triangular portion
    of the connectome (assuming an undirected network) and randomly reassigns
    them to new positions in the upper triangle.

    Parameters:
        connectome (np.ndarray): Square adjacency matrix of the connectome.

    Returns:
        np.ndarray: Rewired connectome with randomized edge positions.
    """
    assert connectome.shape[0] == connectome.shape[1], "Connectome must be a square matrix."

    n = connectome.shape[0]
    # Create an empty matrix for the rewired connectome
    rewired_connectome = np.zeros_like(connectome)

    # Identify the upper triangular indices (excluding the diagonal)
    upper_indices = np.transpose(np.where(np.triu(np.ones_like(connectome), k=1)))
    # Extract the edge weights from the upper triangle that are nonzero
    upper_values = connectome[np.triu_indices(n, k=1)]
    edge_mask = upper_values > 0
    num_edges = np.sum(edge_mask)
    edge_weights = upper_values[edge_mask]

    print(f"Original connectome has {num_edges} edges in the upper triangle.")

    # Randomly select new positions for the existing edges in the upper triangle.
    # We shuffle all possible indices and then select the first 'num_edges' positions.
    shuffled_indices = upper_indices.copy()
    np.random.shuffle(shuffled_indices)
    new_edge_positions = shuffled_indices[:num_edges]

    # Shuffle the extracted edge weights for additional randomness
    np.random.shuffle(edge_weights)

    # Assign the shuffled edge weights to the new positions in the upper triangular region
    for (i, j), weight in tqdm(zip(new_edge_positions, edge_weights), total=num_edges, desc="Rewiring Connectome"):
        rewired_connectome[i, j] = weight
        rewired_connectome[j, i] = weight  # maintain symmetry

    print("Rewiring complete.")
    return rewired_connectome
def threshold_connectome(connectome, percentile=95, binary=False):
    """
    Thresholds the connectome by preserving only the top X% of weights by absolute value.
    The threshold is calculated from the absolute values of all off-diagonal entries.

    Parameters:
        connectome (np.ndarray): Square connectivity matrix.
        percentile (float): Percentile to use for thresholding.
                            (e.g., 95 to keep the top 5% highest magnitude entries)
        binary (bool): If True, the function returns a binary adjacency matrix indicating
                       the presence or absence of a connection. If False, the original weights
                       of the connections are preserved.

    Returns:
        np.ndarray: Thresholded connectome.
    """
    n = connectome.shape[0]
    # Create a mask for off-diagonal entries
    off_diag_mask = ~np.eye(n, dtype=bool)

    # Extract the absolute values of the off-diagonals
    off_diag_values = np.abs(connectome[off_diag_mask])

    # Compute the threshold: keep the top (100 - percentile)% of entries
    threshold_value = np.percentile(off_diag_values, percentile)
    print(f"Threshold value (absolute) at {percentile}th percentile: {threshold_value:.4f}")

    # Create a copy of the connectome to threshold
    thresholded_connectome = np.zeros_like(connectome)

    # Preserve entries that have an absolute value greater than or equal to threshold_value
    strong_connection_mask = (np.abs(connectome) >= threshold_value) & off_diag_mask
    if binary:
        thresholded_connectome[strong_connection_mask] = 1
    else:
        thresholded_connectome[strong_connection_mask] = connectome[strong_connection_mask]

    # Optionally, you might want to preserve diagonal entries.
    # Currently, they are set to zero.

    return thresholded_connectome
def adjust_connectome_density(connectome, target_density=0.2, tol=0.005, max_iter=1000):
    """
    Adjusts the symmetric connectome so that the density (proportion of nonzero off-diagonal entries)
    is close to the target_density by adding or removing connections.

    Density is computed over the full matrix excluding the diagonal.

    Parameters:
        connectome (np.ndarray): Square symmetric connectivity matrix.
        target_density (float): Desired density (e.g., 0.2 for 20%).
        tol (float): Acceptable tolerance from target density.
        max_iter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Adjusted connectome with density close to target_density.
    """
    assert connectome.shape[0] == connectome.shape[1], "Connectome must be a square matrix."
    num_nodes = connectome.shape[0]

    # Total off-diagonal elements in a square matrix (both upper and lower parts)
    total_off_diag = num_nodes * (num_nodes - 1)

    def current_density(mat):
        nonzeros = np.count_nonzero(mat) - np.count_nonzero(np.diag(mat))
        return nonzeros / total_off_diag

    adjusted_connectome = connectome.copy()
    cur_density = current_density(adjusted_connectome)
    print(f"Initial density: {cur_density:.4f}")

    # Calculate target number of nonzero off-diagonals
    target_nonzero = int(round(target_density * total_off_diag))

    off_diag_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    for iter_count in trange(max_iter, desc="Adjusting Density"):
        cur_density = current_density(adjusted_connectome)
        if abs(cur_density - target_density) <= tol:
            print(f"Converged to density {cur_density:.4f} after {iter_count} iterations.")
            break

        current_nonzero = np.count_nonzero(adjusted_connectome) - np.count_nonzero(np.diag(adjusted_connectome))

        if cur_density < target_density:
            # Need to add connections
            connections_to_add = target_nonzero - current_nonzero
            if connections_to_add <= 0:
                break
            zero_indices = [(i, j) for i, j in off_diag_indices if adjusted_connectome[i, j] == 0]
            if not zero_indices:
                print("No available zero entries to add connections.")
                break
            num_to_add = min(connections_to_add, len(zero_indices))
            existing_weights = adjusted_connectome[adjusted_connectome != 0]
            if existing_weights.size == 0:
                raise ValueError("No existing nonzero weights to sample from.")

            selected_indices = np.random.choice(len(zero_indices), size=num_to_add, replace=False)
            for idx in tqdm(selected_indices, desc="Adding connections", leave=False):
                i, j = zero_indices[idx]
                new_weight = np.random.choice(existing_weights)
                adjusted_connectome[i, j] = new_weight
                adjusted_connectome[j, i] = new_weight  # Ensure symmetry
        elif cur_density > target_density:
            # Need to remove connections
            connections_to_remove = current_nonzero - target_nonzero
            if connections_to_remove <= 0:
                break
            nonzero_indices = [(i, j) for i, j in off_diag_indices if adjusted_connectome[i, j] != 0]
            if not nonzero_indices:
                print("No available nonzero entries to remove.")
                break
            num_to_remove = min(connections_to_remove, len(nonzero_indices))
            selected_indices = np.random.choice(len(nonzero_indices), size=num_to_remove, replace=False)
            for idx in tqdm(selected_indices, desc="Removing connections", leave=False):
                i, j = nonzero_indices[idx]
                adjusted_connectome[i, j] = 0
                adjusted_connectome[j, i] = 0  # Ensure symmetry
    else:
        print("Maximum iterations reached. Final density may not be exactly at target.")

    final_density = current_density(adjusted_connectome)
    print(f"Final density: {final_density:.4f}")
    return adjusted_connectome
def select_top_nodes(connectome, target_nodes=300, save_filename=None):
    """
    Selects the top 'target_nodes' from the connectome based on the overall
    absolute connectivity strength and returns the corresponding submatrix.

    Parameters:
        connectome (np.ndarray): Square connectivity matrix.
        target_nodes (int): The desired number of nodes to keep (rows/columns).
        save_filename (str): Optional filename to save the new connectome; if None, no file is saved.

    Returns:
        np.ndarray: The new, thresholded connectivity matrix of shape (target_nodes, target_nodes).
    """
    assert connectome.shape[0] == connectome.shape[1], "Connectome must be a square matrix."
    num_nodes = connectome.shape[0]
    if target_nodes > num_nodes:
        raise ValueError("target_nodes cannot be greater than the number of nodes in the original connectome.")

    # Compute the absolute connectivity strength for each node.
    # Ignore self-connections by setting the diagonal to zero.
    conn_copy = connectome.copy()
    np.fill_diagonal(conn_copy, 0)
    # Sum of absolute weights for each node
    node_strengths = np.sum(np.abs(conn_copy), axis=0)

    # Get indices for the top `target_nodes` nodes (largest overall strength)
    top_node_indices = np.argsort(node_strengths)[-target_nodes:]
    top_node_indices = np.sort(top_node_indices)  # sort indices to preserve ordering

    # Extract the submatrix corresponding to the selected nodes.
    new_connectome = connectome[np.ix_(top_node_indices, top_node_indices)]

    # Optionally save the new connectivity matrix.
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        np.save(save_filename, new_connectome)
        print(f"New top-{target_nodes} connectome saved to: {save_filename}")

    return new_connectome

# Load network, generate ablations
if THRESHOLD == 0 and TOP_300 == 0:
    original = np.load("../ablations/connectome_pearson.npy", allow_pickle=True)
    print("Connectome shape:", original.shape)

    # make thresholded and top_300 for first time...
    thresholded = threshold_connectome(original, percentile=95, binary=False)
    save_connectome_npy(thresholded, os.path.join("../ablations", "thresh_connectome.npy"))
    top300 = select_top_nodes(original, target_nodes=300,
                              save_filename=os.path.join("../ablations", "top300_connectome.npy"))

    if TESTING == 1:
        # Use a small subset for quick runtimes (testing)
        original = original[0:100, 0:100]
        print("Testing subset shape:", original.shape)

    rewired = rewire_connectome(original)
    sparse = adjust_connectome_density(original, target_density=0.2)
    save_connectome_npy(rewired, os.path.join("../ablations", "connectome_rewired.npy"))
    save_connectome_npy(sparse, os.path.join("../ablations", "connectome_sparse_ablated.npy"))

    if TESTING == 1:
        show_connectome(original, "Original Network (Testing Subset)", directed=True)
        show_connectome(rewired, "Rewired Network (Testing Subset)", directed=True)
        show_connectome(sparse, "Sparsity Adjusted Network (Testing Subset)", directed=True)
        show_connectome(thresholded, "Thresholded")
        show_connectome(top300, "Top 300")
    print("All connectomes have been saved successfully.")

elif THRESHOLD == 1 and (THRESHOLD & TOP_300 == 0):
    original = np.load("../ablations/thresh_connectome.npy", allow_pickle=True)
    print("Using thresholded connectome...")
    print("Connectome shape:", original.shape)

    if TESTING == 1:
        # Use a small subset for quick runtimes (testing)
        original = original[0:100, 0:100]
        print("Testing subset shape:", original.shape)

    rewired = rewire_connectome(original)
    sparse = adjust_connectome_density(original, target_density=0.2)

    if TESTING == 1:
        show_connectome(original, "THRESHOLD Original Network (Testing Subset)", directed=True)
        show_connectome(rewired, "THRESHOLD Rewired Network (Testing Subset)", directed=True)
        show_connectome(sparse, "THRESHOLD Sparsity Adjusted Network (Testing Subset)", directed=True)

    save_connectome_npy(rewired, os.path.join("../ablations", "thresh_connectome_rewired.npy"))
    save_connectome_npy(sparse, os.path.join("../ablations", "thresh_connectome_sparse_ablated.npy"))
    print("All connectomes have been saved successfully.")

elif TOP_300 == 1 and (THRESHOLD & TOP_300 == 0):
    original = np.load("../ablations/top300_connectome.npy", allow_pickle=True)
    print("Using top300 connectome...")
    print("Connectome shape:", original.shape)

    rewired = rewire_connectome(original)
    sparse = adjust_connectome_density(original, target_density=0.2)

    if TESTING == 1:
        show_connectome(original, "TOP300 Original Network (Testing Subset)", directed=True)
        show_connectome(rewired, "TOP300 Rewired Network (Testing Subset)", directed=True)
        show_connectome(sparse, "TOP300 Sparsity Adjusted Network (Testing Subset)", directed=True)

    save_connectome_npy(rewired, os.path.join("../ablations", "top300_connectome_rewired.npy"))
    save_connectome_npy(sparse, os.path.join("../ablations", "top300_connectome_sparse_ablated.npy"))
    print("All connectomes have been saved successfully.")

else:
    print("Nothing was done...")