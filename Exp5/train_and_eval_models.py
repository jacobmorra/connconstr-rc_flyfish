import os
import numpy as np
import seaborn as sns
import glob
import json
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse, rsquare
from scipy.stats import t, mannwhitneyu

# parameters, set-up
TESTING = 1
num_trials = 5
data_size = 20000
is_preamble = 1
preamble = "thresh"

tr_1 = int(0.80 * data_size)
tr_2 = tr_1 + 1
val_1 = tr_2 + 1
val_2 = data_size
results = {}
all_predictions = {}
results_dir = "../results"

def compute_mean_and_ci(data, axis=None):
    """
    Compute the mean and 95% confidence interval along a specified axis.
    If axis=None, computes on the flattened data.

    Parameters:
        data (np.ndarray): Input data array.
        axis (int or None): Axis along which to compute statistics.

    Returns:
        tuple: (mean, ci), where ci is the 95% confidence interval.
    """
    mean = np.mean(data, axis=axis)
    sem = np.std(data, axis=axis) / np.sqrt(data.shape[axis] if axis is not None else data.size)
    ci = sem * t.ppf((1 + 0.95) / 2, data.shape[axis] - 1 if axis is not None else data.size - 1)
    return mean, ci
def save_results_to_file(results, filename):
    with open(filename, "w") as file:
        file.write("Connectome RMSE Results (Mean ± 95% CI)\n")
        file.write("=" * 50 + "\n")
        for name, metrics in results.items():
            file.write(
                f"{name}: RMSE = {metrics['mean_rmse']:.4f} ± {metrics['ci_rmse']:.4f}\n"
            )
    print(f"Results saved to {filename}")

# load networks
if is_preamble==0:
    original_fconn = np.load("../ablations/connectome_pearson.npy", allow_pickle=True)
    rewired_fconn = np.load("../ablations/connectome_rewired.npy", allow_pickle=True)
    sparsity_ablated_fconn = np.load("../ablations/connectome_sparse_ablated.npy", allow_pickle=True)
    thresholded_fconn = np.load("../ablations/thresh_connectome.npy", allow_pickle=True)

    # currently rewired and sparsity come from thresholded, not original... can change later?
    W_matrix1 = original_fconn
    W_matrix2 = rewired_fconn
    W_matrix3 = thresholded_fconn
    W_matrix4 = sparsity_ablated_fconn

    connectomes = {
        "original": W_matrix1,
        "rewired": W_matrix2,
        "thresholded": W_matrix3,
        "sparsity_ablated": W_matrix4,
    }

else:
    original_fconn = np.load(f"../ablations/{preamble}_connectome.npy", allow_pickle=True)
    rewired_fconn = np.load(f"../ablations/{preamble}_connectome_rewired.npy", allow_pickle=True)
    sparsity_ablated_fconn = np.load(f"../ablations/{preamble}_connectome_sparse_ablated.npy", allow_pickle=True)

    # currently rewired and sparsity come from thresholded, not original... can change later?
    W_matrix1 = original_fconn
    W_matrix2 = rewired_fconn
    W_matrix4 = sparsity_ablated_fconn

    connectomes = {
        preamble: W_matrix1,
        "rewired": W_matrix2,
        "sparsity_ablated": W_matrix4,
    }


# load data
output_filename = "../data/20241119_elavl3rsChrm_H2bG6s_7dpf_OMR2Stim_fish10_OMR_stack-002/plane_0/tail_sum_time_series.npy"
loaded_time_series = np.load(output_filename)[:, 1]  # tail sums
print("Time series shape:", loaded_time_series.shape)
if TESTING == 1:
    sns.lineplot(loaded_time_series)
    plt.show()
X = loaded_time_series.reshape(-1, 1)

# train, validate models
# We'll store the rmse_scores for each connectome for statistical testing.
rmse_data = {}

for name, W_matrix in connectomes.items():
    reservoir = Reservoir(W=W_matrix)
    readout = Ridge(output_dim=1, ridge=1e-5)
    esn = reservoir >> readout
    rmse_scores = []
    prediction_runs = []

    for i in range(num_trials):
        esn.fit(X[:tr_1], X[1:tr_2], warmup=100)
        predictions = esn.run(X[val_1:val_2])
        current_rmse = rmse(X[val_1:val_2], predictions)
        rmse_scores.append(current_rmse)
        prediction_runs.append(predictions.flatten())

    # Save the rmse scores for statistical testing
    rmse_data[name] = np.array(rmse_scores)

    mean_rmse, ci_rmse = compute_mean_and_ci(np.array(rmse_scores))
    results[name] = {"mean_rmse": mean_rmse, "ci_rmse": ci_rmse}

    # store predictions
    prediction_runs = np.array(prediction_runs)
    prediction_mean, prediction_ci = compute_mean_and_ci(prediction_runs, axis=0)
    all_predictions[name] = {
        "mean": prediction_mean,
        "ci": prediction_ci,
        "ground_truth": X[val_1:val_2].flatten(),
        "all_runs": prediction_runs  # save each run
    }

    np.save(os.path.join(results_dir, f"{name}_predictions.npy"), prediction_runs)
    np.save(os.path.join(results_dir, f"{name}_ground_truth.npy"), X[val_1:val_2].flatten())
    np.save(os.path.join(results_dir, f"{name}_rmse_scores.npy"), np.array(rmse_scores))

save_results_to_file(results, os.path.join(results_dir, "rmse_results.txt"))

mann_whitney_results = {}
names = list(rmse_data.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name1 = names[i]
        name2 = names[j]
        # Use two-sided test
        u_statistic, p_value = mannwhitneyu(rmse_data[name1], rmse_data[name2], alternative='two-sided')
        key = f"{name1} vs. {name2}"
        mann_whitney_results[key] = {"u_statistic": u_statistic, "p_value": p_value}
        print(f"Mann-Whitney U test ({name1} vs. {name2}): U = {u_statistic:.4f}, p = {p_value:.4f}")

# Save Mann–Whitney test results to JSON
with open(os.path.join(results_dir, "mann_whitney_results.json"), "w") as f:
    json.dump(mann_whitney_results, f, indent=4)

# JSON summary file
json_results = {}
for name, data in results.items():
    json_results[name] = {
        "mean_rmse": data["mean_rmse"],
        "ci_rmse": data["ci_rmse"],
        "prediction_mean": all_predictions[name]["mean"].tolist(),
        "prediction_ci": all_predictions[name]["ci"].tolist(),
        "ground_truth": all_predictions[name]["ground_truth"].tolist(),
        "rmse_scores": np.array(all_predictions[name]["all_runs"]).tolist(),
    }

json_results["mann_whitney"] = mann_whitney_results

with open(os.path.join(results_dir, "combined_results.json"), "w") as f:
    json.dump(json_results, f, indent=4)

# bar plot
if TESTING == 1:
    labels = list(results.keys())
    means = [results[name]["mean_rmse"] for name in labels]
    cis = [results[name]["ci_rmse"] for name in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=cis, capsize=5, color=['blue', 'orange', 'green', 'red'])
    plt.ylabel("Average RMSE")
    plt.title("Average RMSE with 95% CI for Each Connectome")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # predictions
    plt.figure(figsize=(12, 10))
    for i, (name, pred_data) in enumerate(all_predictions.items()):
        plt.subplot(2, 2, i + 1)
        ground_truth = pred_data["ground_truth"]
        mean_prediction = pred_data["mean"]
        ci_prediction = pred_data["ci"]

        # Plot ground truth
        plt.plot(ground_truth, label="Ground Truth", color="blue")

        # Plot mean prediction with 95% CI
        plt.plot(mean_prediction, label=f"Mean Prediction ({name})", color="orange")
        plt.fill_between(
            range(len(mean_prediction)),
            mean_prediction - ci_prediction,
            mean_prediction + ci_prediction,
            color="orange",
            alpha=0.3,
            label="95% CI",
        )

        plt.title(f"{name} - Predictions with 95% CI")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()
