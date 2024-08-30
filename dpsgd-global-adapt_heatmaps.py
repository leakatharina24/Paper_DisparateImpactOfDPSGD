from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# %% for performances

# null-hypothesis: mean1 >= mean2, alternative hypothesis: mean1 < mean2

def significance_test_larger(mean1, mean2, std1, std2):
    # mean1 = 0.8856
    # mean2 = 0.8491
    # std1 = 0.0013
    # std2 = 0.0005
    
    # print(f'{mean1}+/-{std1} vs. {mean2}+/-{std2}')
    
    t_statistic = (mean1 - mean2) / np.sqrt((std1**2 / 5) + (std2**2 / 5))
    
    df = ((std1**2 / 5) + (std2**2 / 5))**2 / (((std1**2 / 5)**2 / (5 - 1)) + ((std2**2 / 5)**2 / (5 - 1)))
         
    p_value = stats.t.sf(np.abs(t_statistic), df)  # one-tailed test (because null hypothesis is that mean1 > mean2)
    
    # print(f"T-statistic: {t_statistic:.4f}, P-value: {p_value:.10f}")
    if p_value < 0.05 and t_statistic < 0:
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
    
    t_statistic = (mean1 - mean2) / np.sqrt((std1**2 / 5) + (std2**2 / 5))
    
    df = ((std1**2 / 5) + (std2**2 / 5))**2 / (((std1**2 / 5)**2 / (5 - 1)) + ((std2**2 / 5)**2 / (5 - 1)))
         
    p_value = stats.t.sf(np.abs(t_statistic), df)  # one-tailed test (because null hypothesis is that mean1 > mean2)
    
    # print(f"T-statistic: {t_statistic:.4f}, P-value: {p_value:.10f}")
    if p_value < 0.05 and t_statistic > 0:
        return True
    else:
        return False

# t-statistics: if positive, mean1 > mean2
# if p_value < 0.05: reject null hypothesis


# %% data loading

path = "../Final_results"
dataset = "compas"
C_selection = 'best' #'worst

pmetric = 'accuracy'
fmetric = 'accuracy'

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])
# First row (3 subplots)
ax1 = fig.add_subplot(gs[0, 0:2])  # Row 0, Column 0
ax2 = fig.add_subplot(gs[0, 2:4])  # Row 0, Column 1
ax3 = fig.add_subplot(gs[0, 4:6])  # Row 0, Column 2

# Second row (2 subplots, taking up two columns each)
ax4 = fig.add_subplot(gs[1, 1:3]) # Row 1, Columns 0 and 1
ax5 = fig.add_subplot(gs[1, 3:5])  # Row 1, Column 2

axs = [ax1, ax2, ax3, ax4, ax5]

titles = ['A) Adult', 'B) LSAC', 'C) Compas', 'D) CelebA', 'E) MNIST']

for i, dataset in enumerate(["adult", "lsac", "compas", "celeba", "mnist"]):
    method = "regular"
    epsilon = 5
    if dataset == 'celeba':
        num_samples = 100
        protected_group = 'eyeglasses'
    elif dataset == 'mnist':
        num_samples = 50
        protected_group = 'labels'
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
    
    heatmap = axs[i].imshow([[worse_and_unfairer.sum(), similar_but_unfairer.sum(), better_but_unfairer.sum()], 
                [worse_similarly_fair.sum(), similar.sum(), better_similarly_fair.sum()],
                [worse_but_fairer.sum(), similar_but_fairer.sum(), better_and_fairer.sum()]], cmap='Greys', vmin=0, vmax=num_settings*3) #'Greys', 'cividis'

    x_labels = ['worse', 'similar accuracy', 'better']
    y_labels = ['unfairer', 'similarly fair', 'fairer']
    axs[i].set_xticks(ticks=np.arange(3))
    axs[i].set_xticklabels(labels=x_labels, rotation = 15)
    axs[i].set_yticks(ticks=np.arange(3))
    axs[i].set_yticklabels(labels=y_labels) #, rotation=15)
    axs[i].set_title(f'{titles[i]}')

fig.tight_layout(w_pad=2.5, h_pad=-3)
cbar = fig.colorbar(heatmap, ax=[ax1, ax2, ax3, ax4, ax5], orientation='vertical', fraction=0.03, pad=0.04)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{round(x / (num_settings*3) *100)}%'))



