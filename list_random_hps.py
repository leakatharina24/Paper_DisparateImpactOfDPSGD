import random
import csv

# %% grid search

with open('hp_list_gridsearch_esipova.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['epochs','lr','batch_size','activation', 'optimizer']) 
    
    for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
        for batch_size in [256, 512]:
            for epochs in [5, 10, 20, 40]:
                for activation in ['tanh', 'relu']:
                    for optimizer in ['sgd', 'adam']:
                        new_setting = [epochs, lr, batch_size, activation, optimizer]  
                        csvwriter.writerow(new_setting)

# %% CelebA
num_experiments = 100

counter = 0
prev_settings = []
with open(f'hp_list_{num_experiments}_celeba.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['epochs','lr','batch_size','activation', 'optimizer']) 
    
    while counter<num_experiments:
        epochs = random.choice([5, 10, 20, 40])
        lr = random.choice([1e-4, 1e-3, 1e-2, 1e-1])
        batch_size = random.choice([256, 512])
        activation = random.choice(['tanh', 'relu'])
        optimizer = random.choice(['sgd', 'adam'])
        new_setting = [epochs, lr, batch_size, activation, optimizer]  
        if new_setting not in prev_settings:
            csvwriter.writerow(new_setting)
            prev_settings.append(new_setting)
            counter += 1

# %% MNIST
num_experiments = 50

counter = 0
prev_settings = []
with open(f'hp_list_{num_experiments}_mnist.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['epochs','lr','batch_size','activation', 'optimizer']) 
    
    while counter<num_experiments:
        lr = random.choice([1e-4, 1e-3, 1e-2, 1e-1])
        batch_size = random.choice([256, 512])
        epochs = random.choice([5, 10, 20, 40])
        activation = random.choice(['tanh', 'relu'])
        optimizer = random.choice(['sgd', 'adam'])
        new_setting = [epochs, lr, batch_size, activation, optimizer]  
        if new_setting not in prev_settings:
            csvwriter.writerow(new_setting)
            prev_settings.append(new_setting)
            counter += 1