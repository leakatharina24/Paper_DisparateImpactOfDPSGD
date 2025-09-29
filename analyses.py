from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import warnings

# %% for performances

# null-hypothesis: mean1 >= mean2, alternative hypothesis: mean1 < mean2

def significance_test_larger(mean1, mean2, std1, std2):
    # mean1 = 0.8856
    # mean2 = 0.8491
    # std1 = 0.0013
    # std2 = 0.0005
    
    # # print(f'{mean1}+/-{std1} vs. {mean2}+/-{std2}')
    if np.isnan(mean1) or np.isnan(mean2):
        return np.nan

    try:    
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)        
            t_statistic = (mean1 - mean2) / np.sqrt((std1**2 / 5) + (std2**2 / 5))
        
            df = ((std1**2 / 5) + (std2**2 / 5))**2 / (((std1**2 / 5)**2 / (5 - 1)) + ((std2**2 / 5)**2 / (5 - 1)))
                 
            p_value = stats.t.sf(np.abs(t_statistic), df)  # one-tailed test (because null hypothesis is that mean1 > mean2)
            
            # print(f"T-statistic: {t_statistic:.4f}, P-value: {p_value:.10f}")
            if p_value < 0.05 and t_statistic < 0:
                return True
            else:
                return False
            
    except:
        if mean1 < mean2:
            return True
        else:
            return False

# t-statistics: if negative, mean1 < mean2
# if p_value < 0.05: reject null hypothesis

# %% for differences

# null-hypothesis: mean1 <= mean2, alternative hypothesis: mean1 > mean2

def significance_test_smaller(mean1, mean2, std1, std2):
    # mean1 = 0.1364
    # mean2 = 0.1656
    # std1 = 0.0114
    # std2 = 0.0154
    
    # print(f'{mean1}+/-{std1} vs. {mean2}+/-{std2}')
    
    if np.isnan(mean1) or np.isnan(mean2):
        return np.nan
    
    try:    
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            t_statistic = (mean1 - mean2) / np.sqrt((std1**2 / 5) + (std2**2 / 5))
            
            df = ((std1**2 / 5) + (std2**2 / 5))**2 / (((std1**2 / 5)**2 / (5 - 1)) + ((std2**2 / 5)**2 / (5 - 1)))
                 
            p_value = stats.t.sf(np.abs(t_statistic), df)  # one-tailed test (because null hypothesis is that mean1 > mean2)
            
            # print(f"T-statistic: {t_statistic:.4f}, P-value: {p_value:.10f}")
            if p_value < 0.05 and t_statistic > 0:
                return True
            else:
                return False
    except:
        if mean1 > mean2:
            return True
        else:
            return False 

# t-statistics: if positive, mean1 > mean2
# if p_value < 0.05: reject null hypothesis


# %% data loading

path = "../Final_results"
dataset = "folktable"
C_selection = 'worst' #'best' #

method = "regular"
epsilon = 5
if dataset == 'celeba':
    num_samples = 100
    protected_group = 'eyeglasses'
elif dataset == 'mnist':
    num_samples = 50
    protected_group = 'labels'
elif dataset == 'folktable':
    num_samples = 128
    protected_group = 'deye'
else:
    num_samples = 128
    protected_group = 'sex'

results = pd.read_csv(f'{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', delimiter=',', header=0) #1152
if dataset == 'celeba':
    results = results.iloc[:50]
if dataset != 'mnist':
    results = results.rename(columns={'equalized_odds_mean': 'equalized_odds_difference_mean', 'equalized_odds_std': 'equalized_odds_difference_std'})

method = "dpsgd"
#epsilon = 10
dp_results = pd.read_csv(f'{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', delimiter=',', header=0)
if dataset == "celeba":
    dp_results = dp_results.iloc[:150]
if dataset != 'mnist':
    dp_results = dp_results.rename(columns={'equalized_odds_mean': 'equalized_odds_difference_mean', 'equalized_odds_std': 'equalized_odds_difference_std'})

method = 'dpsgd-global-adapt'
if dataset == 'celeba':
    num_samples = 50
dp_global_results = pd.read_csv(f'{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', delimiter=',', header=0)
if dataset != 'mnist':
    dp_global_results = dp_global_results.rename(columns={'equalized_odds_mean': 'equalized_odds_difference_mean', 'equalized_odds_std': 'equalized_odds_difference_std'})

if dataset == 'folktable':
    results = results[:50]
    dp_results = dp_results[:150]
    dp_global_results = dp_global_results[:150]

# %% Results for Table 1: Disparate Impact of DPSGD

pmetrics = ['accuracy', 'roc_auc', 'pr_auc']
# C_selection = 'best' #'worst

print('Does DPSGD have significant negative impact?')
for pmetric in pmetrics:
    best_setting_valid = results[f"valid_{pmetric}"].argmax()
    best_hps = results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer']].iloc[best_setting_valid]
    
    mean1 = results[f'{pmetric}_mean'].iloc[best_setting_valid]
    std1 = results[f'{pmetric}_std'].iloc[best_setting_valid]
    
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_results[hp] == best_hps[hp]
        overall_condition = overall_condition & new_condition

    dp_results_baseline = dp_results[overall_condition]
    best_C_valid = dp_results_baseline[f"valid_{pmetric}"].argmax()
    worst_C_valid = dp_results_baseline[f'valid_{pmetric}'].argmin()
    
    if C_selection == 'best':
        C_valid = best_C_valid
    elif C_selection == 'worst':
        C_valid = worst_C_valid
    else:
        raise ValueError('C selection not valid.')
    
    mean2 = dp_results_baseline[f'{pmetric}_mean'].iloc[C_valid]
    std2 = dp_results_baseline[f'{pmetric}_std'].iloc[C_valid]
    
    print(significance_test_smaller(mean1, mean2, std1, std2))
    
    if dataset == "mnist":
        # fmetrics = pmetrics + ['PPV']
        fmetrics = [pmetric, 'PPV']
    else:
        fmetrics = [pmetric, 'acceptance_rate', 'equalized_odds', 'PPV']
        # fmetrics = pmetrics + ['acceptance_rate', 'equalized_odds', 'PPV']
    for fmetric in fmetrics:
        mean1 = results[f'{fmetric}_difference_mean'].iloc[best_setting_valid]
        std1 = results[f'{fmetric}_difference_std'].iloc[best_setting_valid]
        mean2 = dp_results_baseline[f'{fmetric}_difference_mean'].iloc[C_valid]
        std2 = dp_results_baseline[f'{fmetric}_difference_std'].iloc[C_valid]
        
        print(significance_test_larger(mean1, mean2, std1, std2))

    print('-------------------')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')    

# %% Results for Table 2: Hyperparameter tuning

pmetrics = ['accuracy', 'roc_auc', 'pr_auc']
# C_selection = 'best'

print('Does tuning significantly improve DPSGD?')
for pmetric in pmetrics:
    best_setting_valid = results[f"valid_{pmetric}"].argmax()
    best_hps = results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer']].iloc[best_setting_valid]
    
    # mean1 = results[f'{pmetric}_mean'].iloc[best_setting_valid]
    # std1 = results[f'{pmetric}_std'].iloc[best_setting_valid]
    
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_results[hp] == best_hps[hp]
        overall_condition = overall_condition & new_condition

    dp_results_baseline = dp_results[overall_condition]
    best_C_valid = dp_results_baseline[f"valid_{pmetric}"].argmax()
    worst_C_valid = dp_results_baseline[f'valid_{pmetric}'].argmin()
    
    if C_selection == 'best':
        C_valid = best_C_valid
    elif C_selection == 'worst':
        C_valid = worst_C_valid
    else:
        raise ValueError('C selection not valid.')
    
    mean1 = dp_results_baseline[f'{pmetric}_mean'].iloc[C_valid]
    std1 = dp_results_baseline[f'{pmetric}_std'].iloc[C_valid]
    
    best_setting_valid_dp = dp_results[f"valid_{pmetric}"].argmax()
    
    mean2 = dp_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp]
    std2 = dp_results[f'{pmetric}_std'].iloc[best_setting_valid_dp]
    
    print(significance_test_larger(mean1, mean2, std1, std2))
    
    if dataset == "mnist":
        fmetrics = [pmetric, 'PPV']
    else:
        fmetrics = [pmetric, 'acceptance_rate', 'equalized_odds', 'PPV']
    for fmetric in fmetrics:
        # mean1 = results[f'{fmetric}_difference_mean'].iloc[best_setting_valid]
        # std1 = results[f'{fmetric}_difference_std'].iloc[best_setting_valid]
        mean1 = dp_results_baseline[f'{fmetric}_difference_mean'].iloc[C_valid]
        std1 = dp_results_baseline[f'{fmetric}_difference_std'].iloc[C_valid]
        mean2 = dp_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp]
        std2 = dp_results[f'{fmetric}_difference_std'].iloc[best_setting_valid_dp]
        
        print(significance_test_smaller(mean1, mean2, std1, std2))
    print('-------------------')
    
print('---------------------------------------------------')
print('Is tuned SGD significantly better than tuned DPSGD?')
for pmetric in pmetrics:
    best_setting_valid = results[f"valid_{pmetric}"].argmax()
    best_hps = results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer']].iloc[best_setting_valid]
    
    mean1 = results[f'{pmetric}_mean'].iloc[best_setting_valid]
    std1 = results[f'{pmetric}_std'].iloc[best_setting_valid]
    
    best_setting_valid_dp = dp_results[f"valid_{pmetric}"].argmax()
    
    mean2 = dp_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp]
    std2 = dp_results[f'{pmetric}_std'].iloc[best_setting_valid_dp]
    
    print(significance_test_smaller(mean1, mean2, std1, std2))

    for fmetric in fmetrics:
        mean1 = results[f'{fmetric}_difference_mean'].iloc[best_setting_valid]
        std1 = results[f'{fmetric}_difference_std'].iloc[best_setting_valid]
        # mean1 = dp_results_baseline[f'{fmetric}_difference_mean'].iloc[C_valid]
        # std1 = dp_results_baseline[f'{fmetric}_difference_std'].iloc[C_valid]
        mean2 = dp_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp]
        std2 = dp_results[f'{fmetric}_difference_std'].iloc[best_setting_valid_dp]
        
        print(significance_test_larger(mean1, mean2, std1, std2))
    print('-------------------')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')  

# %% Fig. 1: scatterplot of 5% best hyperparameter settings

pmetric = 'accuracy'
fmetric = "accuracy"

percentage_best = 0.05
num_best = round(results.shape[0]*percentage_best)
num_best_dp = round(dp_results.shape[0]*percentage_best)

plt.figure(figsize=(5,3))
plt.scatter(results.sort_values(f'valid_{pmetric}')[f'{pmetric}_mean'].iloc[-num_best:], results.sort_values(f'valid_{pmetric}')[f'{fmetric}_difference_mean'].iloc[-num_best:], s=20, alpha=1)
plt.scatter(dp_results.sort_values(f'valid_{pmetric}')[f'{pmetric}_mean'].iloc[-num_best_dp:], dp_results.sort_values(f'valid_{pmetric}')[f'{fmetric}_difference_mean'].iloc[-num_best_dp:], s=20, alpha=1, c='red', marker='x')

for _, hps in results.sort_values(f'valid_{pmetric}').iloc[-num_best:].iterrows():
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_results[hp] == hps[hp]
        overall_condition = overall_condition & new_condition

    dp_results_baseline = dp_results[overall_condition]
    plt.scatter(dp_results_baseline[f'{pmetric}_mean'], dp_results_baseline[f'{fmetric}_difference_mean'], c='orange', marker='+')
    
# plt.scatter(results[f'{pmetric}_mean'].iloc[best_setting_valid], results[f'{fmetric}_difference_mean'].iloc[best_setting_valid], c='black', marker='x')
# 
# plt.scatter(dp_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp], dp_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp], c='red', marker='x')
plt.legend(['Tuned SGD', 'Tuned DPSGD', 'Untuned DPSGD'])
if fmetric == 'accuracy':
    plt.xlabel('Accuracy')
    plt.ylabel('Accuracy difference')
#plt.title(f'{dataset}')
#plt.savefig(f"C:/Users/ldemelius/OneDrive - know-center.at/Projects/DPFairness/TMLR_submission/svg_figures/fluctuations_example.svg") #, dpi=300


# %% Figs. 2-6: Lineplots and Heatmaps over all hyperparameter settings

pmetric = 'accuracy'
fmetric = 'accuracy'

with_std = True
alpha = 0.2

fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

# Create subplots using the GridSpec layout
ax1 = plt.subplot(gs[0, 0])  # Top-left subplot
ax2 = plt.subplot(gs[1, 0])  # Bottom-left subplot
ax3 = plt.subplot(gs[:, 1])  # Right-side subplot spanning both rows

sorted_indices = results[f'{pmetric}_mean'].argsort()

mean_sgd = results[f'{pmetric}_mean'][sorted_indices].to_numpy()
std_sgd = results[f'{pmetric}_std'][sorted_indices].to_numpy()
if with_std:
    ax1.fill_between(range(len(mean_sgd)), mean_sgd - std_sgd, mean_sgd + std_sgd, color='blue', alpha=alpha)
ax1.plot(mean_sgd)

dp_results['group'] = (dp_results.index // 3)

if C_selection == "best":
    dp_results_grouped = dp_results.loc[dp_results.groupby('group')[f"valid_{pmetric}"].idxmax()]
elif C_selection == "worst":
    dp_results_grouped = dp_results.loc[dp_results.groupby('group')[f"valid_{pmetric}"].idxmin()]
else:
    raise ValueError('C selection not valid.')
dp_results = dp_results.drop(columns=['group'])

mean_dp = dp_results_grouped.sort_values(f'{pmetric}_mean')[f'{pmetric}_mean'].to_numpy()
std_dp = dp_results_grouped.sort_values(f'{pmetric}_mean')[f'{pmetric}_std'].to_numpy()
if with_std:
    ax1.fill_between(range(len(mean_sgd)), mean_dp - std_dp, mean_dp + std_dp, color='orange', alpha=alpha)
ax1.plot(mean_dp, linestyle='--')

mean_dp2 = dp_results_grouped[f'{pmetric}_mean'].to_numpy()[sorted_indices]
std_dp2 = dp_results_grouped[f'{pmetric}_std'].to_numpy()[sorted_indices]
if with_std:
    ax1.fill_between(range(len(mean_sgd)), mean_dp2 - std_dp2, mean_dp2 + std_dp2, color='green', alpha=alpha)
ax1.plot(mean_dp2, linestyle='-.')
ax1.set_ylabel('Accuracy')
ax1.set_xticks([])
mean_sgd = results[f'{fmetric}_difference_mean'][sorted_indices].to_numpy()
std_sgd = results[f'{fmetric}_difference_std'][sorted_indices].to_numpy()
if with_std:
    ax2.fill_between(range(len(mean_sgd)), mean_sgd - std_sgd, mean_sgd + std_sgd, color='blue', alpha=alpha)
ax2.plot(mean_sgd)

mean_dp = dp_results_grouped.sort_values(f'{pmetric}_mean')[f'{fmetric}_difference_mean'].to_numpy()
std_dp = dp_results_grouped.sort_values(f'{pmetric}_mean')[f'{fmetric}_difference_std'].to_numpy()
if with_std:
    ax2.fill_between(range(len(mean_sgd)), mean_dp - std_dp, mean_dp + std_dp, color='orange', alpha=alpha)
ax2.plot(mean_dp, linestyle='--')

mean_dp2 = dp_results_grouped[f'{fmetric}_difference_mean'].to_numpy()[sorted_indices]
std_dp2 = dp_results_grouped[f'{fmetric}_difference_std'].to_numpy()[sorted_indices]
if with_std:
    ax2.fill_between(range(len(mean_sgd)), mean_dp2 - std_dp2, mean_dp2 + std_dp2, color='green', alpha=alpha)
ax2.plot(mean_dp2, linestyle='-.')
fig.legend(['SGD (Ord. by SGD acc.)', 'DPSGD (Ord. by DPSGD acc.)', 'DPSGD (Ord. by SGD acc.)'], loc='upper left', bbox_to_anchor=(0.375, 0.59))
ax2.set_ylabel('Accuracy difference')
ax2.set_xticks([])
#ax1.set_ylim([0.35, 0.95])
#ax2.set_ylim([0, 0.25])
ax2.set_xlabel('Hyperparameter settings')
ax1.set_title('A)')

percentage = 1. #0.05 #
num_settings = round(percentage*results.shape[0])

directions = np.array([]) 
# 0=significantly better and fairer
# 1=significantly better but significantly unfairer
# 2=significantly worse but significantly fairer
# 3=significantly worse and significanlty unfairer 
# 4=significantly better but ~fair
# 5=significantly worse but ~fair
# 6=~performance but significantly fairer
# 7=~performance but significantly unfairer
# 8=~performance and ~fair


distances = []
performances = []
dp_performances = []

for non_dp in results.iterrows():
    
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_results[hp] == non_dp[1][hp]
        overall_condition = overall_condition & new_condition
    
    if C_selection == 'best':
        dp_results_partner_idx = dp_results[f'valid_{pmetric}'][overall_condition].argmax()
    elif C_selection == 'worst':
        dp_results_partner_idx = dp_results[f'valid_{pmetric}'][overall_condition].argmin()
    dp_results_partner = dp_results[overall_condition].iloc[dp_results_partner_idx]
    regular_performance = non_dp[1][f'{pmetric}_mean']
    regular_unfairness = non_dp[1][f'{fmetric}_difference_mean']
    dpsgd_performance = dp_results_partner[f'{pmetric}_mean']
    dpsgd_unfairness = dp_results_partner[f'{fmetric}_difference_mean']
    performances.append(regular_performance)
    dp_performances.append(dpsgd_performance)
    point1 = [non_dp[1][f'{pmetric}_mean'], non_dp[1][f'{fmetric}_difference_mean']]
    point2 = [dp_results_partner[f'{pmetric}_mean'], dp_results_partner[f'{fmetric}_difference_mean']]
    distances.append(np.linalg.norm(np.array(point1) - np.array(point2)))
    
    if significance_test_larger(regular_performance, dpsgd_performance, non_dp[1][f'{pmetric}_std'], dp_results_partner[f'{pmetric}_std']):
        if significance_test_smaller(regular_unfairness, dpsgd_unfairness, non_dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 0)
        elif significance_test_larger(regular_unfairness, dpsgd_unfairness, non_dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 1)
        else:
            directions = np.append(directions, 4)
    elif significance_test_smaller(regular_performance, dpsgd_performance, non_dp[1][f'{pmetric}_std'], dp_results_partner[f'{pmetric}_std']):
        if significance_test_smaller(regular_unfairness, dpsgd_unfairness, non_dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 2)
        elif significance_test_larger(regular_unfairness, dpsgd_unfairness, non_dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 3)
        else:
            directions = np.append(directions, 5)
    else:
        if significance_test_smaller(regular_unfairness, dpsgd_unfairness, non_dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 6)
        elif significance_test_larger(regular_unfairness, dpsgd_unfairness, non_dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 7)
        else:
            directions = np.append(directions, 8)

performances = np.array(performances)
dp_performances = np.array(dp_performances)
print(f'Results show the best {percentage*100}% of HP settings')

better_and_fairer = (directions==0) & (performances>np.sort(performances)[-num_settings]) 
better_but_unfairer = (directions==1) & (performances>=np.sort(performances)[-num_settings])
worse_but_fairer = (directions==2) & (performances>=np.sort(performances)[-num_settings])
worse_and_unfairer = (directions==3) & (performances>=np.sort(performances)[-num_settings])
better_similarly_fair = (directions == 4) & (performances>np.sort(performances)[-num_settings])
worse_similarly_fair = (directions == 5) & (performances>np.sort(performances)[-num_settings])
similar_but_fairer = (directions == 6) & (performances>np.sort(performances)[-num_settings])
similar_but_unfairer = (directions == 7) & (performances>np.sort(performances)[-num_settings])
similar = (directions == 8) & (performances>np.sort(performances)[-num_settings])

heatmap = ax3.imshow([[worse_and_unfairer.sum(), similar_but_unfairer.sum(), better_but_unfairer.sum()], 
            [worse_similarly_fair.sum(), similar.sum(), better_similarly_fair.sum()],
            [worse_but_fairer.sum(), similar_but_fairer.sum(), better_and_fairer.sum()]], cmap='Greys', vmin=0, vmax=num_settings) #'Greys', 'cividis'
cbar = fig.colorbar(heatmap, ax=ax3)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / num_settings *100}%'))
x_labels = ['worse', 'similar accuracy', 'better']
y_labels = ['unfairer', 'similarly fair', 'fairer']
ax3.set_xticks(ticks=np.arange(3))
ax3.set_xticklabels(labels=x_labels, rotation = 10)
ax3.set_yticks(ticks=np.arange(3))
ax3.set_yticklabels(labels=y_labels)
ax3.set_title('B)')
fig.tight_layout()
#fig.savefig(f"C:/Users/ldemelius/OneDrive - know-center.at/Projects/DPFairness/TMLR_submission/svg_figures/{dataset}_{protected_group}_lineplot_heatmap_{C_selection}.svg") #, dpi=300

# %% Results for Table 3: DPSGD-Global-Adapt

pmetrics = ['accuracy', 'roc_auc', 'pr_auc']
# C_selection = 'best'

print('Is DPSGD-Global-Adapt signficantly better than standard DPSGD? (untuned)')
for pmetric in pmetrics:
    best_setting_valid = results[f"valid_{pmetric}"].argmax()
    best_hps = results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer']].iloc[best_setting_valid]
    
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_results[hp] == best_hps[hp]
        overall_condition = overall_condition & new_condition

    dp_results_baseline = dp_results[overall_condition]
    best_C_valid = dp_results_baseline[f"valid_{pmetric}"].argmax()
    worst_C_valid = dp_results_baseline[f'valid_{pmetric}'].argmin()
    
    if C_selection == 'best':
        C_valid = best_C_valid
    elif C_selection == 'worst':
        C_valid = worst_C_valid
    else:
        raise ValueError('C selection not valid.')
    
    dp_global_results_baseline = dp_global_results[overall_condition]
    best_C_valid_global = dp_global_results_baseline[f"valid_{pmetric}"].argmax()
    worst_C_valid_global = dp_global_results_baseline[f'valid_{pmetric}'].argmin()
    
    if C_selection == 'best':
        C_valid_global = best_C_valid_global
    elif C_selection == 'worst':
        C_valid_global = worst_C_valid_global
    else:
        raise ValueError('C selection not valid.')
    
    mean1 = dp_results_baseline[f'{pmetric}_mean'].iloc[C_valid]
    std1 = dp_results_baseline[f'{pmetric}_std'].iloc[C_valid]
    
    best_setting_valid_dp = dp_results[f"valid_{pmetric}"].argmax()
    
    mean2 = dp_global_results_baseline[f'{pmetric}_mean'].iloc[C_valid_global]
    std2 = dp_global_results_baseline[f'{pmetric}_std'].iloc[C_valid_global]
    
    print(significance_test_larger(mean1, mean2, std1, std2))
    
    if dataset == "mnist":
        fmetrics = [pmetric, 'PPV']
    else:
        fmetrics = [pmetric, 'acceptance_rate', 'equalized_odds', 'PPV']
    for fmetric in fmetrics:
        mean1 = dp_results_baseline[f'{fmetric}_difference_mean'].iloc[C_valid]
        std1 = dp_results_baseline[f'{fmetric}_difference_std'].iloc[C_valid]
        mean2 = dp_global_results_baseline[f'{fmetric}_difference_mean'].iloc[C_valid_global]
        std2 = dp_global_results_baseline[f'{fmetric}_difference_std'].iloc[C_valid_global]

        
        print(significance_test_smaller(mean1, mean2, std1, std2))
    print('-------------------')
print('-------------------------------------------------')    

print('Is DPSGD-Global-Adapt signficantly better than standard DPSGD? (tuned)')
for pmetric in pmetrics:  
    best_setting_valid_dp = dp_results[f"valid_{pmetric}"].argmax()
    
    mean1 = dp_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp]
    std1 = dp_results[f'{pmetric}_std'].iloc[best_setting_valid_dp]
    
    best_setting_valid_dp_global = dp_global_results[f"valid_{pmetric}"].argmax()
    
    mean2 = dp_global_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp_global]
    std2 = dp_global_results[f'{pmetric}_std'].iloc[best_setting_valid_dp_global]
    
    print(significance_test_larger(mean1, mean2, std1, std2))
    
    for fmetric in fmetrics:
        mean1 = dp_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp]
        std1 = dp_results[f'{fmetric}_difference_std'].iloc[best_setting_valid_dp]
        
        mean2 = dp_global_results[f'{fmetric}_difference_mean'].iloc[best_setting_valid_dp_global]
        std2 = dp_global_results[f'{fmetric}_difference_std'].iloc[best_setting_valid_dp_global]
        
        print(significance_test_smaller(mean1, mean2, std1, std2))
    print('-------------------')
    
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

# %% Fig. 7: DPSGD-Global-Adapt heatmaps 
### for all-in-one-figure version see "dpsgd-global-adapt_heatmaps.py"

pmetric = 'accuracy'
fmetric = 'accuracy'
#C_selection = 'best'

percentage = 1. #0.05 #
num_settings = round(percentage*results.shape[0])

directions = np.array([]) 
# 0=significantly better and fairer
# 1=significantly better but significantly unfairer
# 2=significantly worse but significantly fairer
# 3=significantly worse and significanlty unfairer 
# 4=significantly better but ~fair
# 5=significantly worse but ~fair
# 6=~performance but significantly fairer
# 7=~performance but significantly unfairer
# 8=~performance and ~fair


distances = []
dp_global_performances = []
dp_performances = []

for dp in dp_results.iterrows():
    
    overall_condition = pd.Series([True] * dp_results.shape[0])
    for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
        new_condition = dp_global_results[hp] == dp[1][hp]
        overall_condition = overall_condition & new_condition
    
    if C_selection == 'best':
        dp_results_partner_idx = dp_global_results[f'valid_{pmetric}'][overall_condition].argmax()
    elif C_selection == 'worst':
        dp_results_partner_idx = dp_global_results[f'valid_{pmetric}'][overall_condition].argmin() #max() 
    dp_results_partner = dp_global_results[overall_condition].iloc[dp_results_partner_idx]
    dpsgd_performance = dp[1][f'{pmetric}_mean']
    dpsgd_unfairness = dp[1][f'{fmetric}_difference_mean']
    dp_global_performance = dp_results_partner[f'{pmetric}_mean']
    dp_global_unfairness = dp_results_partner[f'{fmetric}_difference_mean']
    dp_performances.append(dpsgd_performance)
    dp_global_performances.append(dp_global_performance)
    if significance_test_larger(dpsgd_performance, dp_global_performance, dp[1][f'{pmetric}_std'], dp_results_partner[f'{pmetric}_std']):
        if significance_test_smaller(dpsgd_unfairness, dp_global_unfairness, dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 0)
        elif significance_test_larger(dpsgd_unfairness, dp_global_unfairness, dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 1)
        else:
            directions = np.append(directions, 4)
    elif significance_test_smaller(dpsgd_performance, dp_global_performance, dp[1][f'{pmetric}_std'], dp_results_partner[f'{pmetric}_std']):
        if significance_test_smaller(dpsgd_unfairness, dp_global_unfairness, dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 2)
        elif significance_test_larger(dpsgd_unfairness, dp_global_unfairness, dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 3)
        else:
            directions = np.append(directions, 5)
    else:
        if significance_test_smaller(dpsgd_unfairness, dp_global_unfairness, dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 6)
        elif significance_test_larger(dpsgd_unfairness, dp_global_unfairness, dp[1][f'{fmetric}_difference_std'], dp_results_partner[f'{fmetric}_difference_std']):
            directions = np.append(directions, 7)
        else:
            directions = np.append(directions, 8)

dp_global_performances = np.array(dp_global_performances)
dp_performances = np.array(dp_performances)
print(f'Results show the best {percentage*100}% of HP settings')

better_and_fairer = (directions==0)
better_but_unfairer = (directions==1) 
worse_but_fairer = (directions==2) 
worse_and_unfairer = (directions==3) 
better_similarly_fair = (directions == 4) 
worse_similarly_fair = (directions == 5) 
similar_but_fairer = (directions == 6) 
similar_but_unfairer = (directions == 7)
similar = (directions == 8) 

plt.figure()
heatmap = plt.imshow([[worse_and_unfairer.sum(), similar_but_unfairer.sum(), better_but_unfairer.sum()], 
            [worse_similarly_fair.sum(), similar.sum(), better_similarly_fair.sum()],
            [worse_but_fairer.sum(), similar_but_fairer.sum(), better_and_fairer.sum()]], cmap='Greys', vmin=0, vmax=num_settings*3) #'Greys', 'cividis'

x_labels = ['worse', 'similar accuracy', 'better']
y_labels = ['unfairer', 'similarly fair', 'fairer']
plt.xticks(ticks=np.arange(3), labels=x_labels)
plt.yticks(ticks=np.arange(3), labels=y_labels) #, rotation=15)

cbar = fig.colorbar(heatmap, ax=plt.gca())
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{round(x / (num_settings*3) *100)}%'))

# %% print detailed results

pmetric = "roc_auc" #"accuracy" #"pr_auc" #

best_setting_valid = results[f"valid_{pmetric}"].argmax()
best_hps = results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer']].iloc[best_setting_valid]

best_setting_valid_dp = dp_results[f"valid_{pmetric}"].argmax()
best_hps_dp = dp_results[["epochs", 'lr', 'batch_size', 'activation', 'optimizer', 'max_grad_norm']].iloc[best_setting_valid_dp]

print('-----------------------')
print(str(round(results[f'{pmetric}_mean'].iloc[best_setting_valid], 4))+' ± '+str(round(results[f'{pmetric}_std'].iloc[best_setting_valid], 4)))
if dataset == 'mnist':
    fmetrics = [pmetric, 'PPV']
else:
    fmetrics = [pmetric, "acceptance_rate", 'equalized_odds', 'PPV']
for fmetric in fmetrics:
    print(str(round(results[f"{fmetric}_difference_mean"].iloc[best_setting_valid], 4))+' ± '+str(round(results[f"{fmetric}_difference_std"].iloc[best_setting_valid], 4)))
print('-----------------------')
overall_condition = pd.Series([True] * dp_results.shape[0])

for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
    new_condition = dp_results[hp] == best_hps[hp]
    overall_condition = overall_condition & new_condition

dp_results_baseline = dp_results[overall_condition]
best_C_valid = dp_results_baseline[f"valid_{pmetric}"].argmax()
worst_C_valid = dp_results_baseline[f'valid_{pmetric}'].argmin()

print(str(round(dp_results_baseline[f'{pmetric}_mean'].iloc[best_C_valid], 4))+' ± '+str(round(dp_results_baseline[f'{pmetric}_std'].iloc[best_C_valid], 4)))
for fmetric in fmetrics:
    print(str(round(dp_results_baseline[f"{fmetric}_difference_mean"].iloc[best_C_valid], 4)) +' ± '+str(round(dp_results_baseline[f"{fmetric}_difference_std"].iloc[best_C_valid], 4)))

print('-----------------------')

print(str(round(dp_results_baseline[f'{pmetric}_mean'].iloc[worst_C_valid], 4))+' ± '+str(round(dp_results_baseline[f'{pmetric}_std'].iloc[worst_C_valid], 4)))
for fmetric in fmetrics:
    print(str(round(dp_results_baseline[f"{fmetric}_difference_mean"].iloc[worst_C_valid], 4)) +' ± '+str(round(dp_results_baseline[f"{fmetric}_difference_std"].iloc[worst_C_valid], 4)))

print('-----------------------')

print(str(round(dp_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp], 4))+' ± '+str(round(dp_results[f'{pmetric}_std'].iloc[best_setting_valid_dp], 4)))
for fmetric in fmetrics:
    print(str(round(dp_results[f"{fmetric}_difference_mean"].iloc[best_setting_valid_dp], 4)) +' ± '+str(round(dp_results[f"{fmetric}_difference_std"].iloc[best_setting_valid_dp], 4)))
print('-----------------------')

best_setting_valid_dp_global = dp_global_results[f"valid_{pmetric}"].argmax()

overall_condition = pd.Series([True] * dp_global_results.shape[0])
if dataset == 'mnist':
    fmetrics = [pmetric, 'PPV']
else:
    fmetrics = [pmetric, 'acceptance_rate', 'equalized_odds', 'PPV']

for hp in ["epochs", 'lr', 'batch_size', 'activation', 'optimizer']:
    new_condition = dp_global_results[hp] == best_hps[hp]
    overall_condition = overall_condition & new_condition

dp_global_results_baseline = dp_global_results[overall_condition]
best_C_valid = dp_global_results_baseline[f"valid_{pmetric}"].argmax()
worst_C_valid = dp_global_results_baseline[f'valid_{pmetric}'].argmin()

print(str(round(dp_global_results_baseline[f'{pmetric}_mean'].iloc[best_C_valid], 4))+' ± '+str(round(dp_global_results_baseline[f'{pmetric}_std'].iloc[best_C_valid], 4)))
for fmetric in fmetrics:
    print(str(round(dp_global_results_baseline[f"{fmetric}_difference_mean"].iloc[best_C_valid], 4)) +' ± '+str(round(dp_global_results_baseline[f"{fmetric}_difference_std"].iloc[best_C_valid], 4)))

print('-----------------------')

print(str(round(dp_global_results_baseline[f'{pmetric}_mean'].iloc[worst_C_valid], 4))+' ± '+str(round(dp_global_results_baseline[f'{pmetric}_std'].iloc[worst_C_valid], 4)))
for fmetric in fmetrics:
    print(str(round(dp_global_results_baseline[f"{fmetric}_difference_mean"].iloc[worst_C_valid], 4)) +' ± '+str(round(dp_global_results_baseline[f"{fmetric}_difference_std"].iloc[worst_C_valid], 4)))

print('-----------------------')

print(str(round(dp_global_results[f'{pmetric}_mean'].iloc[best_setting_valid_dp_global], 4))+' ± '+str(round(dp_global_results[f'{pmetric}_std'].iloc[best_setting_valid_dp_global], 4)))
for fmetric in fmetrics:
    print(str(round(dp_global_results[f"{fmetric}_difference_mean"].iloc[best_setting_valid_dp_global], 4)) +' ± '+str(round(dp_global_results[f"{fmetric}_difference_std"].iloc[best_setting_valid_dp_global], 4)))
print('-----------------------')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')  