# create Figure 8... for Baboon5 vs Baboon4..
# recall Baboon5 is most MF, Baboon4 low MF.. why?

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.sparse import csr_matrix
from MAIN_22 import generate_M,generate_Win, generate_M_custom, Big_listen_stage,Big_train_stage,predict_stage,generate_NetOut
from MAIN_22 import Generate_predicitons,Generate_MF_predicitons
from Circle_error_tools import Error_analysis_of_Pred_Circle,test_Error_analysis_of_Pred_Circle, GetErrorBoth
from Circle_error_tools import check_errmaxminCA,check_errmaxminCB, fix_length_of_maxmins_with_nans

"""
------------------------------------------------------------------------------------
SETUP PARAMS...
"""
# Time constants for integration
dt = 0.01  # time step
Tlisten = 37.7  # Listening Time 6T
ListenEndTime = int(Tlisten / dt)  # Discretised Listen Time
Ttrain = 94.25 + Tlisten  # Training Time 15T
TrainEndTime = int(Ttrain / dt)  # Discretised Train Time
Tpredict = 94.25 + Ttrain  # Predicting Time 15T
PredictEndTime = int(Tpredict / dt)  # Discretised Predict Time
t_time = np.linspace(0.0, Tpredict, int(Tpredict / dt))  # (Total Time)
SysDim = 2  # Used in initialising Win and regression
# Res params
gama = 5  # damping coefficient
sigma = 0.2  # input strength
beta = 1e-2  # Regularization Parameter
alpha = 0.5  # Blending parameter
# Input Data Params
dd1 = 5
dd2 = -5
omega1 = 1
omega2 = -1
predtime = PredictEndTime - TrainEndTime
FP_err_lim = 1e-3
sample_start = predtime - 5000  # +10000
sample_end = predtime - 1000  # +10000
stepback = 20
FP_sample_start = predtime - 1000  # +10000
FP_sample_end = predtime  # +10000
iter_no = 1000
LC_err_tol = 0.01
rounding_no = 2
# Error criteria
LC_error_bound = 0.1
# KEY PARAMETERS ------------------------------------------------------
xcen = 0.0
Xcen1 = xcen
Xcen2 = -Xcen1
ycen = 0.0
# ---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def process_and_plot_connectome(connectome_path, win_path, rho=1.4, dt=0.01, time_range=None, bin_range=(-2, 2), num_bins=50):
    """
    Processes the connectome, trains the reservoir computer, and plots the activation differences over time.
    Additionally, saves the heatmap data into an array and optionally returns it.

    Parameters:
    - connectome_path: str
      Path to the connectome .npy file.
    - win_path: str
      Path to the input weight matrix .npy file.
    - rho: float, default=1.4
      Scaling factor for the spectral radius.
    - dt: float, default=0.01
      Time step duration in seconds.
    - time_range: tuple, optional
      Range of time steps to plot (start, end). If None, defaults to the entire duration.
    - bin_range: tuple, default=(-2, 2)
      The range of the bins for the histogram.
    - num_bins: int, default=50
      The number of bins to use in the histogram.

    Returns:
    - heatmap_data: np.ndarray
      The heatmap data array.
    - xpredict1_C1_MF, xpredict1_C2_MF: np.ndarray
      The predicted outputs for circles A and B.
    """

    # Load the connectome
    with open(connectome_path, 'rb') as f:
        M_conn = np.load(f)

    # Remaining params
    N_i = M_conn.shape[0]
    N = int(N_i)
    rho = np.round(rho, 4)
    M, Minit, largest_evalue = generate_M_custom(M_conn, rho)

    # Load or generate the Win matrix
    Win = csr_matrix(np.load(win_path, allow_pickle=True))

    # Train the reservoir
    Xpredict_1_MF, Xpredict_2_MF, Rpredictsq_1_MF, Rpredictsq_2_MF, xy_1, xy_2, r_1, r_2, Wout_alpha, NetOut_1_MF, NetOut_2_MF = Generate_MF_predicitons(
        rho, xcen, alpha, dt, t_time, ListenEndTime, TrainEndTime, PredictEndTime, M, Win, largest_evalue, N, dd1,
        omega1, dd2, omega2, gama, sigma, beta)

    # Extract reservoir activations and calculate differences
    rvals_extractCA = Rpredictsq_1_MF[0:300, :]  # Circle A rvals
    rvals_extractCB = Rpredictsq_2_MF[0:300, :]  # Circle B rvals
    rdiffs = rvals_extractCB - rvals_extractCA

    # Determine time range for plotting
    if time_range is None:
        time_range = (0, rdiffs.shape[1])
    rdiffs_to_plot = rdiffs[:, time_range[0]:time_range[1]]

    # Generate and save the heatmap data
    heatmap_data = generate_heatmap_data(rdiffs_to_plot, bin_range=bin_range, num_bins=num_bins)

    # Get predicted outputs for x_CA and x_CB
    xpredict1_C1_MF, xpredict2_C1_MF = NetOut_1_MF
    xpredict1_C2_MF, xpredict2_C2_MF = NetOut_2_MF

    # return array, xc1, xc2, yc1, yc2
    return heatmap_data, xpredict1_C1_MF, xpredict1_C2_MF, xpredict2_C1_MF, xpredict2_C2_MF

def generate_heatmap_data(rdiffs, bin_range=(-2, 2), num_bins=50):
    """
    Generates the heatmap data array from the reservoir activation differences.

    Parameters:
    - rdiffs: numpy array, shape (300, n_time_steps)
      The differences in reservoir activations.
    - bin_range: tuple, default=(-2, 2)
      The range of the bins for the histogram.
    - num_bins: int, default=50
      The number of bins to use in the histogram.

    Returns:
    - heatmap_data: np.ndarray
      The heatmap data array.
    """

    bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
    time_steps = rdiffs.shape[1]
    heatmap_data = np.zeros((num_bins, time_steps))
    for t in range(time_steps):
        rdiffs_at_step = rdiffs[:, t]
        hist, _ = np.histogram(rdiffs_at_step, bins=bins)
        heatmap_data[:, t] = hist

    return heatmap_data

def plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmap, dt=0.01, bin_range=(-2, 2)):
    """
    Plots the Baboon5, Baboon4, and difference heatmaps with enhanced mathematical notation.

    Parameters:
    - bab5_heatmap: np.ndarray
      The heatmap data for Baboon5.
    - bab4_heatmap: np.ndarray
      The heatmap data for Baboon4.
    - diff_heatmap: np.ndarray
      The difference heatmap data (|Baboon5 - Baboon4|).
    - dt: float, default=0.01
      Time step duration in seconds.
    - bin_range: tuple, default=(-2, 2)
      The range of the bins for the histogram.
    """

    time_steps = bab5_heatmap.shape[1]
    time_axis = np.arange(time_steps) * dt

    # Calculate the colorbar range across all heatmaps
    vmin = min(np.min(bab5_heatmap), np.min(bab4_heatmap))
    vmax = max(np.max(bab5_heatmap), np.max(bab4_heatmap))

    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

    # Enhanced LaTeX for rhats
    ylabel_text = r"$\hat{\mathbf{r}}_{2} - \hat{\mathbf{r}}_{1}$"

    # Baboon5 heatmap
    im1 = axs[0].imshow(bab5_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
                        cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[0].set_title('Baboon5')
    axs[0].set_ylabel(ylabel_text)
    fig.colorbar(im1, ax=axs[0], orientation='vertical')

    # Baboon4 heatmap
    im2 = axs[1].imshow(bab4_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
                        cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[1].set_title('Baboon4')
    axs[1].set_ylabel(ylabel_text)
    fig.colorbar(im2, ax=axs[1], orientation='vertical')

    # Difference heatmap
    im3 = axs[2].imshow(diff_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
                        cmap='coolwarm', origin='lower')
    axs[2].set_title('Baboon5 - Baboon4')
    axs[2].set_ylabel(ylabel_text)
    fig.colorbar(im3, ax=axs[2], orientation='vertical')

    plt.tight_layout()
    plt.show()



def plot_connectome(M_path, title):
    """
    Plots the connectome as a heatmap.

    Parameters:
    - M_conn: np.ndarray
      The connectivity matrix.
    - title: str
      The title of the plot.
    """
    with open(M_path, 'rb') as f:
        M_conn = np.load(f)
    plt.figure(figsize=(6, 6))
    sns.heatmap(M_conn, cmap='viridis', cbar=True, square=True)
    plt.title(title)
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.show()


def plot_predictions(x1C1, x2C1, x1C2, x2C2, xp1_C1, xp2_C1, xp1_C2, xp2_C2):
    """
    Plots the predictions of the RC versus the actual output.

    Parameters:
    - x1C1, x2C1: np.ndarray
      Predicted outputs for circle A.
    - x1C2, x2C2: np.ndarray
      Predicted outputs for circle B.
    - xp1_C1, xp2_C1: np.ndarray
      Actual outputs for circle A.
    - xp1_C2, xp2_C2: np.ndarray
      Actual outputs for circle B.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x1C1, x2C1, label='Circle 1 Predict')
    plt.plot(x1C1[-1], x2C1[-1], 'C0.')
    plt.plot(x1C2, x2C2, label='Circle 2 Predict')
    plt.plot(x1C2[-1], x2C2[-1], 'C1.')
    plt.plot(xp1_C1, xp2_C1, 'r--', label='Circle 1 Actual')
    plt.plot(xp1_C2, xp2_C2, 'g--', label='Circle 2 Actual')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predictions vs Actual Outputs')
    plt.show()

def plotpreds(x1C1,x2C1,x1C2,x2C2):
    plt.plot(x1C1,x2C1, label='Circle 1 Predict')
    plt.plot(x1C1[-1],x2C1[-1],'C0.')
    plt.plot(x1C2,x2C2, label = 'Circle 2 Predict')
    plt.plot(x1C2[-1],x2C2[-1],'C1.')
    plt.legend()
    print(x1C1[-1],x2C1[-1])
    print(x1C2[-1],x2C2[-1])
    plt.show()

# Generate heatmaps and predicted outputs for Baboon5 and Baboon4
bab5_heatmap, x_CA_bab5, x_CB_bab5, y_CA_bab5, y_CB_bab5 = process_and_plot_connectome('connectivity/mami/conn_150/Baboon5.npy', 'WinBaboon5.npy', time_range=(0, 4000))
bab4_heatmap, x_CA_bab4, x_CB_bab4, y_CA_bab4, y_CB_bab4 = process_and_plot_connectome('connectivity/mami/conn_150/Baboon4.npy', 'WinBaboon5.npy', time_range=(0, 4000))

# Calculate the difference heatmap
# diff_heatmaps = np.abs(bab5_heatmap - bab4_heatmap)
diff_heatmaps = bab4_heatmap - bab5_heatmap

# Plot all heatmaps along with x_CA - x_CB
# plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmaps, x_CA_bab5, x_CB_bab4)
# plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmaps, dt=0.01)


# plot_connectome('connectivity/mami/conn_150/Baboon5.npy', "Bab5")
# plot_connectome('connectivity/mami/conn_150/Baboon4.npy', 'Bab4')
#
# plotpreds(x_CA_bab5,y_CA_bab5,x_CB_bab5,y_CB_bab5)
# plotpreds(x_CA_bab4,y_CA_bab4,x_CB_bab4,y_CB_bab4)


def track_max_y_value(connectome_path, win_path, rho_start=1.4, rho_end=1.6, rho_step=0.01, dt=0.01, time_range=(0, 4000)):
    rho_values = np.arange(rho_start, rho_end + rho_step, rho_step)
    max_y_values = []

    for rho in tqdm(rho_values):
        # Process the connectome
        _, x_CA, _, y_CA, _ = process_and_plot_connectome(connectome_path, win_path, rho=rho, time_range=time_range)
        max_y = np.max(y_CA)
        max_y_values.append(max_y)

    return rho_values, max_y_values

def plot_max_y_vs_rho(rho_values, max_y_values, connectome_label):
    plt.figure(figsize=(10, 6))

    # Scatter plot for the given connectome
    plt.scatter(rho_values, max_y_values, label=connectome_label, marker='o')

    plt.xlabel('rho')
    plt.ylabel('Max y-value (x2)')
    plt.title(f'Max y-value vs rho ({connectome_label})')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_x_CA_vs_x_CB(x_CA, x_CB, dt=0.01):
    """
    Plots the difference between x_CA and x_CB over time.

    Parameters:
    - x_CA: np.ndarray
      The predicted output for circle A.
    - x_CB: np.ndarray
      The predicted output for circle B.
    - dt: float, default=0.01
      Time step duration in seconds.
    """
    time_steps = len(x_CA)
    time_axis = np.arange(time_steps) * dt

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, x_CA - x_CB, color='blue')
    plt.xlim(time_axis[0], time_axis[-1])
    plt.title('x_CA - x_CB vs Time')
    plt.xlabel("Time (seconds)")
    plt.ylabel("x_CA - x_CB")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot_x_CA_vs_x_CB(x_CA_bab5, x_CB_bab4, dt=0.01)


# # Example for Baboon5
# rho_values, max_y_values_bab5 = track_max_y_value(
#     'connectivity/mami/conn_150/Baboon5.npy', 'WinBaboon5.npy',
#     rho_start=1.4, rho_end=1.6, rho_step=0.01)
#
# # Plot the results for Baboon5
# plot_max_y_vs_rho(rho_values, max_y_values_bab5, connectome_label='Baboon5')
#
# # Example for Baboon4
# rho_values, max_y_values_bab4 = track_max_y_value(
#     'connectivity/mami/conn_150/Baboon4.npy', 'WinBaboon5.npy',
#     rho_start=1.4, rho_end=1.6, rho_step=0.01)
#
# # Plot the results for Baboon4
# plot_max_y_vs_rho(rho_values, max_y_values_bab4, connectome_label='Baboon4')



import matplotlib.pyplot as plt

# Set global font size
plt.rcParams.update({'font.size': 24})  # Adjust this value for a larger or smaller global font size

def plot_x_CA_vs_x_CB(x_CA, x_CB, dt=0.01):
    """
    Plots the difference between x_CA and x_CB over time with larger fonts and math notation.

    Parameters:
    - x_CA: np.ndarray
      The predicted output for circle A.
    - x_CB: np.ndarray
      The predicted output for circle B.
    - dt: float, default=0.01
      Time step duration in seconds.
    """
    time_steps = len(x_CA)
    time_axis = np.arange(time_steps) * dt

    plt.figure(figsize=(14, 7))  # Increase the figure size if needed
    plt.plot(time_axis, x_CA - x_CB, color='blue')
    plt.xlim(time_axis[0], time_axis[-1])
    # plt.title(r'$x_{\mathcal{C}_A} - x_{\mathcal{C}_B}$ vs $t$', fontsize=22)  # Math notation for title
    plt.xlabel(r"$t$", fontsize=22)  # Math notation for x-axis label
    plt.ylabel(r"$x_{\mathcal{C}_A} - x_{\mathcal{C}_B}$", fontsize=22)  # Math notation for y-axis label
    plt.xticks(fontsize=20)  # Increase x-ticks font size
    plt.yticks(fontsize=20)  # Increase y-ticks font size
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmap, dt=0.01, bin_range=(-2, 2)):
    """
    Plots the Baboon5, Baboon4, and difference heatmaps with larger fonts and math notation.

    Parameters:
    - bab5_heatmap: np.ndarray
      The heatmap data for Baboon5.
    - bab4_heatmap: np.ndarray
      The heatmap data for Baboon4.
    - diff_heatmap: np.ndarray
      The difference heatmap data (|Baboon5 - Baboon4|).
    - dt: float, default=0.01
      Time step duration in seconds.
    - bin_range: tuple, default=(-2, 2)
      The range of the bins for the histogram.
    """

    time_steps = bab5_heatmap.shape[1]
    time_axis = np.arange(time_steps) * dt

    # Calculate the colorbar range across all heatmaps
    vmin = min(np.min(bab5_heatmap), np.min(bab4_heatmap))
    vmax = max(np.max(bab5_heatmap), np.max(bab4_heatmap))

    fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

    # Enhanced LaTeX for rhats and math notation for labels
    ylabel_text = r"$\hat{\mathbf{r}}_{2} - \hat{\mathbf{r}}_{1}$"

    # Baboon5 heatmap
    im1 = axs[0].imshow(bab5_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
                        cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[0].set_title(r'Baboon4', fontsize=24)  # Math notation for title
    axs[0].set_ylabel(ylabel_text, fontsize=22)  # Increase y-axis label font size
    axs[0].tick_params(axis='y', labelsize=20)  # Increase y-ticks font size
    fig.colorbar(im1, ax=axs[0], orientation='vertical').ax.tick_params(labelsize=22)  # Increase colorbar ticks font size

    # Baboon4 heatmap
    im2 = axs[1].imshow(bab4_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
                        cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[1].set_title(r'Baboon5', fontsize=24)  # Math notation for title
    axs[1].set_ylabel(ylabel_text, fontsize=22)  # Increase y-axis label font size
    axs[1].tick_params(axis='y', labelsize=20)  # Increase y-ticks font size
    fig.colorbar(im2, ax=axs[1], orientation='vertical').ax.tick_params(labelsize=22)  # Increase colorbar ticks font size

    # Difference heatmap
    im3 = axs[2].imshow(diff_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
                        cmap='coolwarm', origin='lower')
    axs[2].set_title(r'Baboon5 - Baboon4', fontsize=24)  # Math notation for title
    axs[2].set_ylabel(ylabel_text, fontsize=22)  # Increase y-axis label font size
    axs[2].tick_params(axis='y', labelsize=20)  # Increase y-ticks font size
    axs[2].set_xlabel(r"$t$", fontsize=22)  # Math notation for x-axis label
    axs[2].tick_params(axis='x', labelsize=20)  # Increase x-ticks font size
    fig.colorbar(im3, ax=axs[2], orientation='vertical').ax.tick_params(labelsize=20)  # Increase colorbar ticks font size

    plt.tight_layout()
    plt.show()


plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmaps, dt=0.01)

x_CA_bab5 = x_CA_bab5[:4000]
x_CB_bab4 = x_CB_bab4[:4000]

print(x_CA_bab5.shape)
plot_x_CA_vs_x_CB(x_CA_bab5, x_CB_bab4, dt=0.01)


# uncomment in future after interview
# # create Figure 8... for Baboon5 vs Baboon4..
# # recall Baboon5 is most MF, Baboon4 low MF.. why?
#
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import pandas as pd
# from tqdm import tqdm
# from datetime import datetime
# from scipy.sparse import csr_matrix
# from MAIN_22 import generate_M,generate_Win, generate_M_custom, Big_listen_stage,Big_train_stage,predict_stage,generate_NetOut
# from MAIN_22 import Generate_predicitons,Generate_MF_predicitons
# from Circle_error_tools import Error_analysis_of_Pred_Circle,test_Error_analysis_of_Pred_Circle, GetErrorBoth
# from Circle_error_tools import check_errmaxminCA,check_errmaxminCB, fix_length_of_maxmins_with_nans
#
# """
# ------------------------------------------------------------------------------------
# SETUP PARAMS...
# """
# # Time constants for integration
# dt = 0.01  # time step
# Tlisten = 37.7  # Listening Time 6T
# ListenEndTime = int(Tlisten / dt)  # Discretised Listen Time
# Ttrain = 94.25 + Tlisten  # Training Time 15T
# TrainEndTime = int(Ttrain / dt)  # Discretised Train Time
# Tpredict = 94.25 + Ttrain  # Predicting Time 15T
# PredictEndTime = int(Tpredict / dt)  # Discretised Predict Time
# t_time = np.linspace(0.0, Tpredict, int(Tpredict / dt))  # (Total Time)
# SysDim = 2  # Used in initialising Win and regression
# # Res params
# gama = 5  # damping coefficient
# sigma = 0.2  # input strength
# beta = 1e-2  # Regularization Parameter
# alpha = 0.5  # Blending parameter
# # Input Data Params
# dd1 = 5
# dd2 = -5
# omega1 = 1
# omega2 = -1
# predtime = PredictEndTime - TrainEndTime
# FP_err_lim = 1e-3
# sample_start = predtime - 5000  # +10000
# sample_end = predtime - 1000  # +10000
# stepback = 20
# FP_sample_start = predtime - 1000  # +10000
# FP_sample_end = predtime  # +10000
# iter_no = 1000
# LC_err_tol = 0.01
# rounding_no = 2
# # Error criteria
# LC_error_bound = 0.1
# # KEY PARAMETERS ------------------------------------------------------
# xcen = 0.0
# Xcen1 = xcen
# Xcen2 = -Xcen1
# ycen = 0.0
# # ---------------------------------------------------------------------
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
#
# def process_and_plot_connectome(connectome_path, win_path, rho=1.4, dt=0.01, time_range=None, bin_range=(-2, 2), num_bins=50):
#     """
#     Processes the connectome, trains the reservoir computer, and plots the activation differences over time.
#     Additionally, saves the heatmap data into an array and optionally returns it.
#
#     Parameters:
#     - connectome_path: str
#       Path to the connectome .npy file.
#     - win_path: str
#       Path to the input weight matrix .npy file.
#     - rho: float, default=1.4
#       Scaling factor for the spectral radius.
#     - dt: float, default=0.01
#       Time step duration in seconds.
#     - time_range: tuple, optional
#       Range of time steps to plot (start, end). If None, defaults to the entire duration.
#     - bin_range: tuple, default=(-2, 2)
#       The range of the bins for the histogram.
#     - num_bins: int, default=50
#       The number of bins to use in the histogram.
#
#     Returns:
#     - heatmap_data: np.ndarray
#       The heatmap data array.
#     - xpredict1_C1_MF, xpredict1_C2_MF: np.ndarray
#       The predicted outputs for circles A and B.
#     """
#
#     # Load the connectome
#     with open(connectome_path, 'rb') as f:
#         M_conn = np.load(f)
#
#     # Remaining params
#     N_i = M_conn.shape[0]
#     N = int(N_i)
#     rho = np.round(rho, 4)
#     M, Minit, largest_evalue = generate_M_custom(M_conn, rho)
#
#     # Load or generate the Win matrix
#     Win = csr_matrix(np.load(win_path, allow_pickle=True))
#
#     # Train the reservoir
#     Xpredict_1_MF, Xpredict_2_MF, Rpredictsq_1_MF, Rpredictsq_2_MF, xy_1, xy_2, r_1, r_2, Wout_alpha, NetOut_1_MF, NetOut_2_MF = Generate_MF_predicitons(
#         rho, xcen, alpha, dt, t_time, ListenEndTime, TrainEndTime, PredictEndTime, M, Win, largest_evalue, N, dd1,
#         omega1, dd2, omega2, gama, sigma, beta)
#
#     # Extract reservoir activations and calculate differences
#     rvals_extractCA = Rpredictsq_1_MF[0:300, :]  # Circle A rvals
#     rvals_extractCB = Rpredictsq_2_MF[0:300, :]  # Circle B rvals
#     rdiffs = rvals_extractCB - rvals_extractCA
#
#     # Determine time range for plotting
#     if time_range is None:
#         time_range = (0, rdiffs.shape[1])
#     rdiffs_to_plot = rdiffs[:, time_range[0]:time_range[1]]
#
#     # Generate and save the heatmap data
#     heatmap_data = generate_heatmap_data(rdiffs_to_plot, bin_range=bin_range, num_bins=num_bins)
#
#     # Get predicted outputs for x_CA and x_CB
#     xpredict1_C1_MF, xpredict2_C1_MF = NetOut_1_MF
#     xpredict1_C2_MF, xpredict2_C2_MF = NetOut_2_MF
#
#     # return array, xc1, xc2, yc1, yc2
#     return heatmap_data, xpredict1_C1_MF, xpredict1_C2_MF, xpredict2_C1_MF, xpredict2_C2_MF
#
# def generate_heatmap_data(rdiffs, bin_range=(-2, 2), num_bins=50):
#     """
#     Generates the heatmap data array from the reservoir activation differences.
#
#     Parameters:
#     - rdiffs: numpy array, shape (300, n_time_steps)
#       The differences in reservoir activations.
#     - bin_range: tuple, default=(-2, 2)
#       The range of the bins for the histogram.
#     - num_bins: int, default=50
#       The number of bins to use in the histogram.
#
#     Returns:
#     - heatmap_data: np.ndarray
#       The heatmap data array.
#     """
#
#     bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
#     time_steps = rdiffs.shape[1]
#     heatmap_data = np.zeros((num_bins, time_steps))
#     for t in range(time_steps):
#         rdiffs_at_step = rdiffs[:, t]
#         hist, _ = np.histogram(rdiffs_at_step, bins=bins)
#         heatmap_data[:, t] = hist
#
#     return heatmap_data
#
# def plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmap, dt=0.01, bin_range=(-2, 2)):
#     """
#     Plots the Baboon5, Baboon4, and difference heatmaps with enhanced mathematical notation.
#
#     Parameters:
#     - bab5_heatmap: np.ndarray
#       The heatmap data for Baboon5.
#     - bab4_heatmap: np.ndarray
#       The heatmap data for Baboon4.
#     - diff_heatmap: np.ndarray
#       The difference heatmap data (|Baboon5 - Baboon4|).
#     - dt: float, default=0.01
#       Time step duration in seconds.
#     - bin_range: tuple, default=(-2, 2)
#       The range of the bins for the histogram.
#     """
#
#     time_steps = bab5_heatmap.shape[1]
#     time_axis = np.arange(time_steps) * dt
#
#     # Calculate the colorbar range across all heatmaps
#     vmin = min(np.min(bab5_heatmap), np.min(bab4_heatmap))
#     vmax = max(np.max(bab5_heatmap), np.max(bab4_heatmap))
#
#     fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
#
#     # Enhanced LaTeX for rhats
#     ylabel_text = r"$\hat{\mathbf{r}}_{2} - \hat{\mathbf{r}}_{1}$"
#
#     # Baboon5 heatmap
#     im1 = axs[0].imshow(bab5_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
#                         cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
#     axs[0].set_title('Baboon5')
#     axs[0].set_ylabel(ylabel_text)
#     fig.colorbar(im1, ax=axs[0], orientation='vertical')
#
#     # Baboon4 heatmap
#     im2 = axs[1].imshow(bab4_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
#                         cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
#     axs[1].set_title('Baboon4')
#     axs[1].set_ylabel(ylabel_text)
#     fig.colorbar(im2, ax=axs[1], orientation='vertical')
#
#     # Difference heatmap
#     im3 = axs[2].imshow(diff_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
#                         cmap='coolwarm', origin='lower')
#     axs[2].set_title('Baboon5 - Baboon4')
#     axs[2].set_ylabel(ylabel_text)
#     fig.colorbar(im3, ax=axs[2], orientation='vertical')
#
#     plt.tight_layout()
#     plt.show()
#
#
#
# def plot_connectome(M_path, title):
#     """
#     Plots the connectome as a heatmap.
#
#     Parameters:
#     - M_conn: np.ndarray
#       The connectivity matrix.
#     - title: str
#       The title of the plot.
#     """
#     with open(M_path, 'rb') as f:
#         M_conn = np.load(f)
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(M_conn, cmap='viridis', cbar=True, square=True)
#     plt.title(title)
#     plt.xlabel("Neuron Index")
#     plt.ylabel("Neuron Index")
#     plt.show()
#
#
# def plot_predictions(x1C1, x2C1, x1C2, x2C2, xp1_C1, xp2_C1, xp1_C2, xp2_C2):
#     """
#     Plots the predictions of the RC versus the actual output.
#
#     Parameters:
#     - x1C1, x2C1: np.ndarray
#       Predicted outputs for circle A.
#     - x1C2, x2C2: np.ndarray
#       Predicted outputs for circle B.
#     - xp1_C1, xp2_C1: np.ndarray
#       Actual outputs for circle A.
#     - xp1_C2, xp2_C2: np.ndarray
#       Actual outputs for circle B.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(x1C1, x2C1, label='Circle 1 Predict')
#     plt.plot(x1C1[-1], x2C1[-1], 'C0.')
#     plt.plot(x1C2, x2C2, label='Circle 2 Predict')
#     plt.plot(x1C2[-1], x2C2[-1], 'C1.')
#     plt.plot(xp1_C1, xp2_C1, 'r--', label='Circle 1 Actual')
#     plt.plot(xp1_C2, xp2_C2, 'g--', label='Circle 2 Actual')
#     plt.legend()
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Predictions vs Actual Outputs')
#     plt.show()
#
# def plotpreds(x1C1,x2C1,x1C2,x2C2):
#     plt.plot(x1C1,x2C1, label='Circle 1 Predict')
#     plt.plot(x1C1[-1],x2C1[-1],'C0.')
#     plt.plot(x1C2,x2C2, label = 'Circle 2 Predict')
#     plt.plot(x1C2[-1],x2C2[-1],'C1.')
#     plt.legend()
#     print(x1C1[-1],x2C1[-1])
#     print(x1C2[-1],x2C2[-1])
#     plt.show()
#
# # Generate heatmaps and predicted outputs for Baboon5 and Baboon4
# bab5_heatmap, x_CA_bab5, x_CB_bab5, y_CA_bab5, y_CB_bab5 = process_and_plot_connectome('connectivity/mami/conn_150/Baboon5.npy', 'WinBaboon5.npy', time_range=(0, 4000))
# bab4_heatmap, x_CA_bab4, x_CB_bab4, y_CA_bab4, y_CB_bab4 = process_and_plot_connectome('connectivity/mami/conn_150/Baboon4.npy', 'WinBaboon5.npy', time_range=(0, 4000))
#
# # Calculate the difference heatmap
# # diff_heatmaps = np.abs(bab5_heatmap - bab4_heatmap)
# diff_heatmaps = bab5_heatmap - bab4_heatmap
#
# # Plot all heatmaps along with x_CA - x_CB
# # plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmaps, x_CA_bab5, x_CB_bab4)
# # plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmaps, dt=0.01)
#
#
# # plot_connectome('connectivity/mami/conn_150/Baboon5.npy', "Bab5")
# # plot_connectome('connectivity/mami/conn_150/Baboon4.npy', 'Bab4')
# #
# # plotpreds(x_CA_bab5,y_CA_bab5,x_CB_bab5,y_CB_bab5)
# # plotpreds(x_CA_bab4,y_CA_bab4,x_CB_bab4,y_CB_bab4)
#
#
# def track_max_y_value(connectome_path, win_path, rho_start=1.4, rho_end=1.6, rho_step=0.01, dt=0.01, time_range=(0, 4000)):
#     rho_values = np.arange(rho_start, rho_end + rho_step, rho_step)
#     max_y_values = []
#
#     for rho in tqdm(rho_values):
#         # Process the connectome
#         _, x_CA, _, y_CA, _ = process_and_plot_connectome(connectome_path, win_path, rho=rho, time_range=time_range)
#         max_y = np.max(y_CA)
#         max_y_values.append(max_y)
#
#     return rho_values, max_y_values
#
# def plot_max_y_vs_rho(rho_values, max_y_values, connectome_label):
#     plt.figure(figsize=(10, 6))
#
#     # Scatter plot for the given connectome
#     plt.scatter(rho_values, max_y_values, label=connectome_label, marker='o')
#
#     plt.xlabel('rho')
#     plt.ylabel('Max y-value (x2)')
#     plt.title(f'Max y-value vs rho ({connectome_label})')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# def plot_x_CA_vs_x_CB(x_CA, x_CB, dt=0.01):
#     """
#     Plots the difference between x_CA and x_CB over time.
#
#     Parameters:
#     - x_CA: np.ndarray
#       The predicted output for circle A.
#     - x_CB: np.ndarray
#       The predicted output for circle B.
#     - dt: float, default=0.01
#       Time step duration in seconds.
#     """
#     time_steps = len(x_CA)
#     time_axis = np.arange(time_steps) * dt
#
#     plt.figure(figsize=(12, 5))
#     plt.plot(time_axis, x_CA - x_CB, color='blue')
#     plt.xlim(time_axis[0], time_axis[-1])
#     plt.title('x_CA - x_CB vs Time')
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("x_CA - x_CB")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# # plot_x_CA_vs_x_CB(x_CA_bab5, x_CB_bab4, dt=0.01)
#
#
# # # Example for Baboon5
# # rho_values, max_y_values_bab5 = track_max_y_value(
# #     'connectivity/mami/conn_150/Baboon5.npy', 'WinBaboon5.npy',
# #     rho_start=1.4, rho_end=1.6, rho_step=0.01)
# #
# # # Plot the results for Baboon5
# # plot_max_y_vs_rho(rho_values, max_y_values_bab5, connectome_label='Baboon5')
# #
# # # Example for Baboon4
# # rho_values, max_y_values_bab4 = track_max_y_value(
# #     'connectivity/mami/conn_150/Baboon4.npy', 'WinBaboon5.npy',
# #     rho_start=1.4, rho_end=1.6, rho_step=0.01)
# #
# # # Plot the results for Baboon4
# # plot_max_y_vs_rho(rho_values, max_y_values_bab4, connectome_label='Baboon4')
#
#
#
# import matplotlib.pyplot as plt
#
# # Set global font size
# plt.rcParams.update({'font.size': 24})  # Adjust this value for a larger or smaller global font size
#
# def plot_x_CA_vs_x_CB(x_CA, x_CB, dt=0.01):
#     """
#     Plots the difference between x_CA and x_CB over time with larger fonts and math notation.
#
#     Parameters:
#     - x_CA: np.ndarray
#       The predicted output for circle A.
#     - x_CB: np.ndarray
#       The predicted output for circle B.
#     - dt: float, default=0.01
#       Time step duration in seconds.
#     """
#     time_steps = len(x_CA)
#     time_axis = np.arange(time_steps) * dt
#
#     plt.figure(figsize=(14, 7))  # Increase the figure size if needed
#     plt.plot(time_axis, x_CA - x_CB, color='blue')
#     plt.xlim(time_axis[0], time_axis[-1])
#     # plt.title(r'$x_{\mathcal{C}_A} - x_{\mathcal{C}_B}$ vs $t$', fontsize=22)  # Math notation for title
#     plt.xlabel(r"$t$", fontsize=22)  # Math notation for x-axis label
#     plt.ylabel(r"$x_{\mathcal{C}_A} - x_{\mathcal{C}_B}$", fontsize=22)  # Math notation for y-axis label
#     plt.xticks(fontsize=20)  # Increase x-ticks font size
#     plt.yticks(fontsize=20)  # Increase y-ticks font size
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# def plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmap, dt=0.01, bin_range=(-2, 2)):
#     """
#     Plots the Baboon5, Baboon4, and difference heatmaps with larger fonts and math notation.
#
#     Parameters:
#     - bab5_heatmap: np.ndarray
#       The heatmap data for Baboon5.
#     - bab4_heatmap: np.ndarray
#       The heatmap data for Baboon4.
#     - diff_heatmap: np.ndarray
#       The difference heatmap data (|Baboon5 - Baboon4|).
#     - dt: float, default=0.01
#       Time step duration in seconds.
#     - bin_range: tuple, default=(-2, 2)
#       The range of the bins for the histogram.
#     """
#
#     time_steps = bab5_heatmap.shape[1]
#     time_axis = np.arange(time_steps) * dt
#
#     # Calculate the colorbar range across all heatmaps
#     vmin = min(np.min(bab5_heatmap), np.min(bab4_heatmap))
#     vmax = max(np.max(bab5_heatmap), np.max(bab4_heatmap))
#
#     fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
#
#     # Enhanced LaTeX for rhats and math notation for labels
#     ylabel_text = r"$\hat{\mathbf{r}}_{2} - \hat{\mathbf{r}}_{1}$"
#
#     # Baboon5 heatmap
#     im1 = axs[0].imshow(bab5_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
#                         cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
#     axs[0].set_title(r'Baboon5', fontsize=24)  # Math notation for title
#     axs[0].set_ylabel(ylabel_text, fontsize=22)  # Increase y-axis label font size
#     axs[0].tick_params(axis='y', labelsize=20)  # Increase y-ticks font size
#     fig.colorbar(im1, ax=axs[0], orientation='vertical').ax.tick_params(labelsize=22)  # Increase colorbar ticks font size
#
#     # Baboon4 heatmap
#     im2 = axs[1].imshow(bab4_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
#                         cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
#     axs[1].set_title(r'Baboon4', fontsize=24)  # Math notation for title
#     axs[1].set_ylabel(ylabel_text, fontsize=22)  # Increase y-axis label font size
#     axs[1].tick_params(axis='y', labelsize=20)  # Increase y-ticks font size
#     fig.colorbar(im2, ax=axs[1], orientation='vertical').ax.tick_params(labelsize=22)  # Increase colorbar ticks font size
#
#     # Difference heatmap
#     im3 = axs[2].imshow(diff_heatmap, aspect='auto', extent=[time_axis[0], time_axis[-1], bin_range[0], bin_range[1]],
#                         cmap='coolwarm', origin='lower')
#     axs[2].set_title(r'Baboon5 - Baboon4', fontsize=24)  # Math notation for title
#     axs[2].set_ylabel(ylabel_text, fontsize=22)  # Increase y-axis label font size
#     axs[2].tick_params(axis='y', labelsize=20)  # Increase y-ticks font size
#     axs[2].set_xlabel(r"$t$", fontsize=22)  # Math notation for x-axis label
#     axs[2].tick_params(axis='x', labelsize=20)  # Increase x-ticks font size
#     fig.colorbar(im3, ax=axs[2], orientation='vertical').ax.tick_params(labelsize=20)  # Increase colorbar ticks font size
#
#     plt.tight_layout()
#     plt.show()
#
#
# plot_heatmaps_with_difference(bab5_heatmap, bab4_heatmap, diff_heatmaps, dt=0.01)
#
# x_CA_bab5 = x_CA_bab5[:4000]
# x_CB_bab4 = x_CB_bab4[:4000]
#
# print(x_CA_bab5.shape)
# plot_x_CA_vs_x_CB(x_CA_bab5, x_CB_bab4, dt=0.01)