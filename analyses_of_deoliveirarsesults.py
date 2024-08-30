###2-sample one-sided t-test

from scipy import stats
import numpy as np
import pandas as pd

# %% for performances

# null-hypothesis: mean1 <= mean2, alternative hypothesis: mean1 > mean2 (model 2 is significantly worse than model 1)

def significance_test_performances(mean1, mean2, std1, std2):
    # mean1 = 0.8856
    # mean2 = 0.8491
    # std1 = 0.0013
    # std2 = 0.0005
    
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

# %% for differences

# null-hypothesis: mean1 >= mean2, alternative hypothesis: mean1 < mean2 (model 2 is significantly worse than model 1)

def significance_test_differences(mean1, mean2, std1, std2):
    # mean1 = 0.1364
    # mean2 = 0.1656
    # std1 = 0.0114
    # std2 = 0.0154
    
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

# %% ACS Emp.

mean1 = 0.8837
std1 = 0.001
mean2 = 0.8110
std2 = 0.0062
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.3401
std1 = 0.0875
mean2 = 0.3134
std2 = 0.0653
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.4383
std1 = 0.0805
mean2 = 0.5317
std2 = 0.1561
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5884
std1 = 0.136
mean2 = 0.6623
std2 = 0.1701
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5534
std1 = 0.0998
mean2 = 0.4833
std2 = 0.1296
print(significance_test_differences(mean1, mean2, std1, std2))

# %% ACS Inc.

mean1 = 0.8878
std1 = 0.0011
mean2 = 0.8155
std2 = 0.0045
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2546
std1 = 0.0569
mean2 = 0.3498
std2 = 0.0867
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.4223
std1 = 0.0613
mean2 = 0.6105
std2 = 0.1234
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.436
std1 = 0.078
mean2 = 0.8499
std2 = 0.1656
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.355
std1 = 0.0518
mean2 = 0.7882
std2 = 0.2009
print(significance_test_differences(mean1, mean2, std1, std2))

# %% LSAC

mean1 = 0.8343
std1 = 0.0029
mean2 = 0.7755
std2 = 0.0125
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.0435
std1 = 0.0056
mean2 = 0.0422
std2 = 0.0142
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3064
std1 = 0.0653
mean2 = 0.2007
std2 = 0.0437
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.2548
std1 = 0.0862
mean2 = 0.2853
std2 = 0.0467
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.1688
std1 = 0.0485
mean2 = 0.2202
std2 = 0.0061
print(significance_test_differences(mean1, mean2, std1, std2))

# %% Adult

mean1 = 0.9056
std1 = 0.0011
mean2 = 0.8476
std2 = 0.0073
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.1264
std1 = 0.0249
mean2 = 0.2226
std2 = 0.0556
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.275
std1 = 0.0155
mean2 = 0.4683
std2 = 0.0612
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.7845
std1 = 0.0492
mean2 = 0.8005
std2 = 0.0472
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.94
std1 = 0.0966
mean2 = 0.7567
std2 = 0.1933
print(significance_test_differences(mean1, mean2, std1, std2))

# %% Compas

mean1 = 0.6895
std1 = 0.0041
mean2 = 0.5349
std2 = 0.0359
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.1162
std1 = 0.0273
mean2 = 0.1295
std2 = 0.029
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5101
std1 = 0.0209
mean2 = 0.3322
std2 = 0.1331
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5592
std1 = 0.0476
mean2 = 0.3905
std2 = 0.1356
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3347
std1 = 0.0749
mean2 = 0.2009
std2 = 0.1052
print(significance_test_differences(mean1, mean2, std1, std2))

# %% HP tuning ACS Emp.

mean1 = 0.8110
std1 = 0.0062
mean2 = 0.8702
std2 = 0.0013
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3134
std1 = 0.0653
mean2 = 0.2073
std2 = 0.048
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.5317
std1 = 0.1561
mean2 = 0.3154
std2 = 0.0359
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.6623
std1 = 0.1701
mean2 = 0.2874
std2 = 0.0534
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.4833
std1 = 0.1296
mean2 = 0.2968
std2 = 0.0674
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning ACS Inc.

mean1 = 0.8155
std1 = 0.0045
mean2 = 0.882
std2 = 0.0008
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3498
std1 = 0.0867
mean2 = 0.2225
std2 = 0.0219
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.6105
std1 = 0.1234
mean2 = 0.2556
std2 = 0.049
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.8499
std1 = 0.1656
mean2 = 0.3756
std2 = 0.0019
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.7882
std1 = 0.2009
mean2 = 0.4032
std2 = 0.0271
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning LSAC

mean1 = 0.7755
std1 = 0.0125
mean2 = 0.7962
std2 = 0.0077
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.0422
std1 = 0.0142
mean2 = 0.0575
std2 = 0.0128
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2007
std1 = 0.0437
mean2 = 0.1687
std2 = 0.0151
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2853
std1 = 0.0467
mean2 = 0.1975
std2 = 0.0722
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2202
std1 = 0.0061
mean2 = 0.2197
std2 = 0.0082
print(significance_test_performances(mean1, mean2, std1, std2))
# %% HP tuning  Adult

mean1 = 0.8476
std1 = 0.0073
mean2 = 0.9005
std2 = 0.0009
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.2226
std1 = 0.0556
mean2 = 0.1841
std2= 0.0128
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.4683
std1 = 0.0612
mean2 = 0.2375
std2 = 0.0207
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.8005
std1 = 0.0472
mean2 = 0.8
std2 = 0
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.7567
std1 = 0.1933
mean2 = 0.8
std2 = 0
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning Compas

mean1 = 0.5349
std1 = 0.0359
mean2 = 0.6963
std2 = 0.003
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.1295
std1 = 0.029
mean2 = 0.0824
std2 = 0.0264
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.3322
std1 = 0.1331
mean2 = 0.3694
std2 = 0.023
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.3905
std1 = 0.1356
mean2 = 0.3726
std2 = 0.0375
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2009
std1 = 0.1052
mean2 = 0.3168
std2 = 0.0467
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning v2 ACS Emp.

mean1 = 0.8837
std1 = 0.001
mean2 = 0.8702
std2 = 0.0013
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.3401
std1 = 0.0875
mean2 = 0.2073
std2 = 0.048
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.4383
std1 = 0.0805
mean2 = 0.3154
std2 = 0.0359
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5884
std1 = 0.136
mean2 = 0.2874
std2 = 0.0534
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5534
std1 = 0.0998
mean2 = 0.2968
std2 = 0.0674
print(significance_test_differences(mean1, mean2, std1, std2))

# %% HP tuning v2 ACS Inc.

mean1 = 0.8878
std1 = 0.0011
mean2 = 0.882
std2 = 0.0008
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2546
std1 = 0.0569
mean2 = 0.2225
std2 = 0.0219
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.4223
std1 = 0.0613
mean2 = 0.2556
std2 = 0.049
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.436
std1 = 0.078
mean2 = 0.3756
std2 = 0.0019
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.355
std1 = 0.0518
mean2 = 0.4032
std2 = 0.0271
print(significance_test_differences(mean1, mean2, std1, std2))

# %% HP tuning v2 LSAC

mean1 = 0.8343
std1 = 0.0029
mean2 = 0.7962
std2 = 0.0077
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.0435
std1 = 0.0056
mean2 = 0.7962
std2 = 0.0077
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3064
std1 = 0.0653
mean2 = 0.1687
std2 = 0.0151
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.2548
std1 = 0.0862
mean2 = 0.1975
std2 = 0.0722
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.1688
std1 = 0.0485
mean2 = 0.2197
std2 = 0.0082
print(significance_test_differences(mean1, mean2, std1, std2))

# %% HP tuning v2 Adult

mean1 = 0.9056
std1 = 0.0011
mean2 = 0.9005
std2 = 0.0009
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.1264
std1 = 0.0249
mean2 = 0.1841
std2= 0.0128
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.275
std1 = 0.0155
mean2 = 0.2375
std2 = 0.0207
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.7845
std1 = 0.0492
mean2 = 0.8
std2 = 0
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.94
std1 = 0.0966
mean2 = 0.8
std2 = 0
print(significance_test_differences(mean1, mean2, std1, std2))

# %% HP tuning v2 Compas

mean1 = 0.6895
std1 = 0.0041
mean2 = 0.6963
std2 = 0.003
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.1162
std1 = 0.0273
mean2 = 0.0824
std2 = 0.0264
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5101
std1 = 0.0209
mean2 = 0.3694
std2 = 0.023
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.5592
std1 = 0.0476
mean2 = 0.3726
std2 = 0.0375
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3347
std1 = 0.0749
mean2 = 0.3168
std2 = 0.0467
print(significance_test_differences(mean1, mean2, std1, std2))

# %% HP tuning better than SGD: ACS Emp.

mean1 = 0.8837
std1 = 0.001
mean2 = 0.8702
std2 = 0.0013
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.3401
std1 = 0.0875
mean2 = 0.2073
std2 = 0.048
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.4383
std1 = 0.0805
mean2 = 0.3154
std2 = 0.0359
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.5884
std1 = 0.136
mean2 = 0.2874
std2 = 0.0534
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.5534
std1 = 0.0998
mean2 = 0.2968
std2 = 0.0674
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning better than SGD ACS Inc.

mean1 = 0.8878
std1 = 0.0011
mean2 = 0.882
std2 = 0.0008
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.2546
std1 = 0.0569
mean2 = 0.2225
std2 = 0.0219
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.4223
std1 = 0.0613
mean2 = 0.2556
std2 = 0.049
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.436
std1 = 0.078
mean2 = 0.3756
std2 = 0.0019
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.355
std1 = 0.0518
mean2 = 0.4032
std2 = 0.0271
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning better than SGD LSAC

mean1 = 0.8343
std1 = 0.0029
mean2 = 0.7962
std2 = 0.0077
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.0435
std1 = 0.0056
mean2 = 0.7962
std2 = 0.0077
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.3064
std1 = 0.0653
mean2 = 0.1687
std2 = 0.0151
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.2548
std1 = 0.0862
mean2 = 0.1975
std2 = 0.0722
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.1688
std1 = 0.0485
mean2 = 0.2197
std2 = 0.0082
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning better than SGD Adult

mean1 = 0.9056
std1 = 0.0011
mean2 = 0.9005
std2 = 0.0009
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.1264
std1 = 0.0249
mean2 = 0.1841
std2= 0.0128
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.275
std1 = 0.0155
mean2 = 0.2375
std2 = 0.0207
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.7845
std1 = 0.0492
mean2 = 0.8
std2 = 0
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.94
std1 = 0.0966
mean2 = 0.8
std2 = 0
print(significance_test_performances(mean1, mean2, std1, std2))

# %% HP tuning better than SGD Compas

mean1 = 0.6895
std1 = 0.0041
mean2 = 0.6963
std2 = 0.003
print(significance_test_differences(mean1, mean2, std1, std2))

mean1 = 0.1162
std1 = 0.0273
mean2 = 0.0824
std2 = 0.0264
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.5101
std1 = 0.0209
mean2 = 0.3694
std2 = 0.023
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.5592
std1 = 0.0476
mean2 = 0.3726
std2 = 0.0375
print(significance_test_performances(mean1, mean2, std1, std2))

mean1 = 0.3347
std1 = 0.0749
mean2 = 0.3168
std2 = 0.0467
print(significance_test_performances(mean1, mean2, std1, std2))
