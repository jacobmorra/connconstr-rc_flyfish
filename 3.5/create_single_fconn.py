import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

print(os.listdir())
# Parameters
file = "../data/20241119_elavl3rsChrm_H2bG6s_7dpf_OMR2Stim_fish10_OMR_stack-002"
num_planes = 1  # Planes 0 to 5

# Load neuron traces for all planes
arrays = []
neuron_counts = []
for plane_idx in range(num_planes):
    plane_path = os.path.join(file, f'plane_{plane_idx}', 'F.npy')
    if not os.path.exists(plane_path):
        raise FileNotFoundError(f"File not found: {plane_path}")
    array = np.load(plane_path)
    arrays.append(array)
    neuron_counts.append(array.shape[0])
    print(f"Plane {plane_idx} loaded with shape: {array.shape}")

# Concatenate all planes to form a full array (total_neurons, time_bins)
full_array = np.concatenate(arrays, axis=0)
print(f"Full array shape: {full_array.shape}")

# Standardize the responses (zero mean, unit variance)
scaler_standard = StandardScaler()
standardized_array = scaler_standard.fit_transform(full_array)
print("Standardization complete.")

# Normalize the responses to [0, 1]
scaler_minmax = MinMaxScaler()
normalized_array = scaler_minmax.fit_transform(standardized_array)
print("Normalization complete.")

print("Computing Pearson correlation matrix...")
correlation_matrix = np.corrcoef(normalized_array)
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# Handle any NaN values that might have arisen from constant signals
if np.isnan(correlation_matrix).any():
    print("Warning: NaN values detected in the correlation matrix. Replacing with 0.")
    correlation_matrix = np.nan_to_num(correlation_matrix)

print(os.listdir())
# Save the connectivity matrix
output_path = '../ablations/connectome_pearson.npy'
np.save(output_path, correlation_matrix)
print(f"Functional connectivity matrix saved to {output_path}.")

sns.heatmap(correlation_matrix)
plt.show()