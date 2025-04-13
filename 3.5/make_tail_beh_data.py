import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

TESTING = 1

# Path to the HDF5 file
data_loc = "../data/20241119_elavl3rsChrm_H2bG6s_7dpf_OMR2Stim_fish10_OMR_stack-002/plane_0"
taildata_loc = f"{data_loc}/tail_df.h5"

# Open the HDF5 file and extract data
with h5py.File(taildata_loc, "r") as h5file:
    # Extract data for each block
    block0_items = [item.decode('utf-8') for item in h5file["tail/block0_items"][:]]
    block0_values = h5file["tail/block0_values"][:]
    block1_items = [item.decode('utf-8') for item in h5file["tail/block1_items"][:]]
    block1_values = h5file["tail/block1_values"][:]
    block2_items = [item.decode('utf-8') for item in h5file["tail/block2_items"][:]]
    block2_values = h5file["tail/block2_values"][:]
    df_block0 = pd.DataFrame(block0_values, columns=block0_items)
    df_block1 = pd.DataFrame(block1_values, columns=block1_items)
    df_block2 = pd.DataFrame(block2_values, columns=block2_items)
    tail_df = pd.concat([df_block0, df_block1, df_block2], axis=1)

tail_df_useful = tail_df[["tail_sum","theta_00", "theta_01","theta_02", "theta_03", "theta_04", "theta_05", "t","frame"]]
if TESTING ==1:
    display(tail_df_useful)

tail_sum = tail_df['tail_sum'].values
print(tail_sum.shape)
time = tail_df['t'].values

# Create a time series array
time_series = np.array([time, tail_sum])
print(time_series[1])

# Transpose to make it (N, 2) for better readability
time_series = time_series.T

# Print the shape and a snippet of the array
print("Time series array shape:", time_series.shape)
print("Time series array (first 5 rows):")
print(time_series[:5])

if TESTING == 1:
    plt.plot(time, tail_sum)
    plt.xlabel("time (s)")
    plt.ylabel("tail sum")
    plt.show()

# Specify the filename for the .npy file
output_filename = f"{data_loc}/tail_sum_time_series.npy"

# Save the time series data as an .npy file
np.save(output_filename, time_series)

tail_data = np.load(output_filename)
print(tail_data.shape)