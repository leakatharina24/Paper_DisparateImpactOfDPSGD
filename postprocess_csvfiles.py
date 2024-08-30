import pandas as pd

### combine csv files
path = "scatterplots_gridsearch_esipova_final" #"scatterplots_gridsearch_deoliveira"

dataset = "adult"
method = "dpsgd"
protected_group = 'sex' #'sex' #'eyeglasses' #
epsilon = 10
num_samples = 128 #100  #

df1 = pd.read_csv(f'results/{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}_1bis83.csv', delimiter=',', header=0)
df2 = pd.read_csv(f'results/{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}_83bis128.csv', delimiter=',', header=0)

df1 = df1.iloc[:-1]

# Concatenate DataFrames
df_combined = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
df_combined.to_csv(f'results/{path}/{dataset}_{method}_{protected_group}_{num_samples}_epsilon{epsilon}.csv', index=False)