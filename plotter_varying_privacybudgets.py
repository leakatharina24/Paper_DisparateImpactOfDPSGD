import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

epsilons = [0.5, 1, 5, 10]
path = "../Final_results"

dataset = "lsac"
method = "regular"
protected_group = 'sex'
epsilon = 5
num_samples = 128


results = pd.read_csv(f'{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', delimiter=',', header=0)
results = results.rename(columns={'equalized_odds_mean': 'equalized_odds_difference_mean', 'equalized_odds_std': 'equalized_odds_difference_std'})

pmetric = "accuracy"
fmetric = 'accuracy'

best_setting_valid = results[f"valid_{pmetric}"].argmax()
best_hps = results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer']].iloc[best_setting_valid]

method = "dpsgd"
best_dpsgd_performances = []
baseline_dpsgd_performances = []
best_dpsgd_fairness = []
baseline_dpsgd_fairness = []

best_dpsgd_performances_stds = []
baseline_dpsgd_performances_stds = []
best_dpsgd_fairness_stds = []
baseline_dpsgd_fairness_stds = []
for epsilon in epsilons:
    dp_results = pd.read_csv(f'{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', delimiter=',', header=0)
    if dataset == "celeba":
        dp_results = dp_results.iloc[:150]
    dp_results = dp_results.rename(columns={'equalized_odds_mean': 'equalized_odds_difference_mean', 'equalized_odds_std': 'equalized_odds_difference_std'})
    # dp_results = dp_results.drop(results[(dp_results.dropout != 0.) & (dp_results.num_blocks != 1)].index)

    best_setting_valid_dp = dp_results[f"valid_{pmetric}"].argmax()
    best_hps_dp = dp_results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer', 'max_grad_norm']].iloc[best_setting_valid_dp]
    
    best_dpsgd_performances.append(dp_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp])
    best_dpsgd_performances_stds.append(dp_results[f'{pmetric}_std'].iloc[best_setting_valid_dp])
    best_dpsgd_fairness.append(dp_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp])
    best_dpsgd_fairness_stds.append(dp_results[f'{fmetric}_difference_std'].iloc[best_setting_valid_dp])
    
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_results[hp] == best_hps[hp]
        overall_condition = overall_condition & new_condition

    dp_results_baseline = dp_results[overall_condition]
    
    i = dp_results_baseline[f'{pmetric}_mean'].argmax()
    baseline_dpsgd_performances.append(dp_results_baseline[f'{pmetric}_mean'].iloc[i])
    baseline_dpsgd_performances_stds.append(dp_results_baseline[f'{pmetric}_std'].iloc[i])
    baseline_dpsgd_fairness.append(dp_results_baseline[f'{fmetric}_difference_mean'].iloc[i])
    baseline_dpsgd_fairness_stds.append(dp_results_baseline[f'{fmetric}_difference_std'].iloc[i])
    
method = "dpsgd-global-adapt"
best_dpsgd_global_performances = []
baseline_dpsgd_global_performances = []
best_dpsgd_global_fairness = []
baseline_dpsgd_global_fairness = []

best_dpsgd_global_performances_stds = []
baseline_dpsgd_global_performances_stds = []
best_dpsgd_global_fairness_stds = []
baseline_dpsgd_global_fairness_stds = []
for epsilon in epsilons:
    dp_global_results = pd.read_csv(f'{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', delimiter=',', header=0)
    dp_global_results = dp_global_results.rename(columns={'equalized_odds_mean': 'equalized_odds_difference_mean', 'equalized_odds_std': 'equalized_odds_difference_std'})

    best_setting_valid_dp_global = dp_global_results[f"valid_{pmetric}"].argmax()
    best_hps_dp_global = dp_global_results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer', 'max_grad_norm']].iloc[best_setting_valid_dp_global]

    best_dpsgd_global_performances.append(dp_global_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp_global])
    best_dpsgd_global_performances_stds.append(dp_global_results[f'{pmetric}_std'].iloc[best_setting_valid_dp_global])
    best_dpsgd_global_fairness.append(dp_global_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp_global])
    best_dpsgd_global_fairness_stds.append(dp_global_results[f'{fmetric}_difference_std'].iloc[best_setting_valid_dp_global])

    overall_condition = pd.Series([True] * dp_global_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_global_results[hp] == best_hps[hp]
        overall_condition = overall_condition & new_condition

    dp_global_results_baseline = dp_global_results[overall_condition]
    
    i = dp_global_results_baseline[f'{pmetric}_mean'].argmax()
    baseline_dpsgd_global_performances.append(dp_global_results_baseline[f'{pmetric}_mean'].iloc[i])
    baseline_dpsgd_global_performances_stds.append(dp_global_results_baseline[f'{pmetric}_std'].iloc[i])
    baseline_dpsgd_global_fairness.append(dp_global_results_baseline[f'{fmetric}_difference_mean'].iloc[i])
    baseline_dpsgd_global_fairness_stds.append(dp_global_results_baseline[f'{fmetric}_difference_std'].iloc[i])

baseline_dpsgd_performances = np.array(baseline_dpsgd_performances)
baseline_dpsgd_performances_stds = np.array(baseline_dpsgd_performances_stds)
best_dpsgd_performances = np.array(best_dpsgd_performances)
best_dpsgd_performances_stds = np.array(best_dpsgd_performances_stds)
baseline_dpsgd_global_performances = np.array(baseline_dpsgd_global_performances)
baseline_dpsgd_global_performances_stds = np.array(baseline_dpsgd_global_performances_stds)
best_dpsgd_global_performances = np.array(best_dpsgd_global_performances)
best_dpsgd_global_performances_stds = np.array(best_dpsgd_global_performances_stds)

baseline_dpsgd_fairness = np.array(baseline_dpsgd_fairness)
baseline_dpsgd_fairness_stds = np.array(baseline_dpsgd_fairness_stds)
best_dpsgd_fairness = np.array(best_dpsgd_fairness)
best_dpsgd_fairness_stds = np.array(best_dpsgd_fairness_stds)
baseline_dpsgd_global_fairness = np.array(baseline_dpsgd_global_fairness)
baseline_dpsgd_global_fairness_stds = np.array(baseline_dpsgd_global_fairness_stds)
best_dpsgd_global_fairness = np.array(best_dpsgd_global_fairness)
best_dpsgd_global_fairness_stds = np.array(best_dpsgd_global_fairness_stds)

alpha = 0.2
non_dp_mean = np.array([results[f'{pmetric}_mean'].iloc[best_setting_valid]]*len(epsilons)) 
non_dp_std = np.array([results[f'{pmetric}_std'].iloc[best_setting_valid]]*len(epsilons))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 5))

axs[0].plot(epsilons, non_dp_mean)
axs[0].fill_between(epsilons, non_dp_mean - non_dp_std, non_dp_mean + non_dp_std, alpha = alpha)
axs[0].plot(epsilons, best_dpsgd_performances)
axs[0].fill_between(epsilons, best_dpsgd_performances - best_dpsgd_performances_stds, best_dpsgd_performances + best_dpsgd_performances_stds, alpha = alpha)

axs[0].plot(epsilons, baseline_dpsgd_performances)
axs[0].fill_between(epsilons, baseline_dpsgd_performances - baseline_dpsgd_performances_stds, baseline_dpsgd_performances + baseline_dpsgd_performances_stds, alpha = alpha)

axs[0].plot(epsilons, best_dpsgd_global_performances)
axs[0].fill_between(epsilons, best_dpsgd_global_performances - best_dpsgd_global_performances_stds, best_dpsgd_global_performances + best_dpsgd_global_performances_stds, alpha = alpha)

axs[0].plot(epsilons, baseline_dpsgd_global_performances)
axs[0].fill_between(epsilons, baseline_dpsgd_global_performances - baseline_dpsgd_global_performances_stds, baseline_dpsgd_global_performances + baseline_dpsgd_global_performances_stds, alpha = alpha)

fig.legend(['SGD', 'Tuned DPSGD', 'Untuned DPSGD', 'Tuned DPSGD-Global-Adapt', 'Untuned DPSGD-Global-Adapt'], loc='upper left', bbox_to_anchor=(0.535, 0.665))

axs[0].set_ylabel(f'{pmetric}')

non_dp_mean_fairness = np.array([results[f'{fmetric}_difference_mean'].iloc[best_setting_valid]]*len(epsilons))
non_dp_std_fairness = np.array([results[f'{fmetric}_difference_std'].iloc[best_setting_valid]]*len(epsilons))

axs[1].plot(epsilons, non_dp_mean_fairness)
axs[1].fill_between(epsilons, non_dp_mean_fairness - non_dp_std_fairness, non_dp_mean_fairness + non_dp_std_fairness, alpha=alpha)

axs[1].plot(epsilons, best_dpsgd_fairness)
axs[1].fill_between(epsilons, best_dpsgd_fairness - best_dpsgd_fairness_stds, best_dpsgd_fairness + best_dpsgd_fairness_stds, alpha = alpha)

axs[1].plot(epsilons, baseline_dpsgd_fairness)
axs[1].fill_between(epsilons, baseline_dpsgd_fairness - baseline_dpsgd_fairness_stds, baseline_dpsgd_fairness + baseline_dpsgd_fairness_stds, alpha = alpha)

axs[1].plot(epsilons, best_dpsgd_global_fairness)
axs[1].fill_between(epsilons, best_dpsgd_global_fairness - best_dpsgd_global_fairness_stds, best_dpsgd_global_fairness + best_dpsgd_global_fairness_stds, alpha = alpha)

axs[1].plot(epsilons, baseline_dpsgd_global_fairness)
axs[1].fill_between(epsilons, baseline_dpsgd_global_fairness - baseline_dpsgd_global_fairness_stds, baseline_dpsgd_global_fairness + baseline_dpsgd_global_fairness_stds, alpha = alpha)

plt.xlabel('Privacy budget $\epsilon$')
axs[1].set_ylabel(f'{fmetric} difference')
plt.tight_layout()