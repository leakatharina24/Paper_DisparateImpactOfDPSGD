import argparse
import pprint
import random
import sys

import numpy as np
import torch
from opacus import PrivacyEngine

from config import get_config, parse_config_arg
from evaluators import create_evaluator
from models import create_model
from privacy_engines.dpsgd_f_engine import DPSGDF_PrivacyEngine
from privacy_engines.dpsgd_global_adaptive_engine import DPSGDGlobalAdaptivePrivacyEngine
from privacy_engines.dpsgd_global_engine import DPSGDGlobalPrivacyEngine
from trainers import create_trainer

import pandas as pd
import os
from datasets.tabular import preprocess_adult
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
from typing import Any, Tuple
from datasets.image import get_raw_image_tensors

import warnings
import sklearn
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import csv
import PIL
import torchvision.transforms as transforms

class GroupLabelDataset(torch.utils.data.Dataset):
    ''' 
    Implementation of torch Dataset that returns features 'x', classification labels 'y', and protected group labels 'z'
    '''

    def __init__(self, x, y=None, z=None):
        if y is None:
            y = torch.zeros(x.shape[0]).long()

        if z is None:
            z = torch.zeros(x.shape[0]).long()

        assert x.shape[0] == y.shape[0] and x.shape[0] == z.shape[0]

        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self.x[index], self.y[index], self.z[index]

    def to(self, device):
        return GroupLabelDataset(
            self.x.to(device),
            self.y.to(device),
            self.z.to(device),
        )
    
def scale_vars(df, mapper):
    warnings.filterwarnings('ignore',
              category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if
                  is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def train_cats(df, cat_vars, cont_vars):
    # numercalize/categoricalize
    for name, col in df.items(): 
        if name in cat_vars:
            df[name] = col.cat.codes + 1
    df = pd.get_dummies(df, columns= cat_vars, dummy_na=True)
    return df

def encode_dataframe(df, cat_vars, cont_vars):
    for v in cat_vars: df[v] = df[v].astype('category').cat.as_ordered()
    for v in cont_vars: df[v] = df[v].astype('float32')
    df = train_cats(df, cat_vars, cont_vars)
    return df

def get_adult(cfg):
    data_root = cfg["data_root"]
    protected_group = cfg["protected_group"]
    group_ratios = cfg["group_ratios"]
    seed = cfg["seed"]
    target = "income"
    
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
               "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
               "hours_per_week", "native_country", "income"]

    df_1 = pd.read_csv(os.path.join(data_root, "adult", "adult.data"), sep=", ", engine='python', header=None)
    df_2 = pd.read_csv(os.path.join(data_root, "adult", "adult.test"), sep=", ", engine='python', header=None,
                       skiprows=1)
    df_1.columns = columns
    df_2.columns = columns
    df = pd.concat((df_1, df_2), ignore_index=True)

    df = df.drop("fnlwgt", axis=1)
    for column in df.columns:
        df = df[df[column] != "?"]
    df.to_csv(os.path.join(data_root, "adult", "adult_data_formatted.csv"), index=False)

    df = pd.read_csv(os.path.join(data_root, "adult", "adult_data_formatted.csv"))

    df_preprocessed = preprocess_adult(df, protected_group, target, group_ratios, seed)
    
    feature_columns = df_preprocessed.columns.to_list()
    feature_columns.remove(target)
    # feature_columns.remove("protected_group")
    
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(df_preprocessed[feature_columns].values,df_preprocessed[target].values,
                                                                         df_preprocessed[protected_group].values,
                                                                         stratify=df_preprocessed[target],test_size=0.3,random_state=seed)
    return x_train, x_test, y_train, y_test, z_train, z_test

def load_mnist(cfg):
    images, labels = get_raw_image_tensors('mnist', train=True, data_root=cfg["data_root"], group_ratios=cfg["group_ratios"],
                                           seed=cfg["seed"])
    test_images, test_labels = get_raw_image_tensors('mnist', train=False, data_root=cfg["data_root"])
    
    return images, test_images, labels, test_labels, labels, test_labels
    
def find_restricted(group_ratios, num_samples, sample_idx):
    """
    group_ratios: -1, -1, 0.09, -1, ...
    num_samples: a list of sample counts that falls into each group
    sample_idx: a list of index that is not -1 in group_ratios
    """
    candidates = []
    for i in sample_idx:
        if all(group_ratios[j] * num_samples[i] <= num_samples[j] for j in sample_idx):
            candidates.append(i)
    restricted_index = np.argmax([group_ratios[i] * num_samples[i] for i in candidates])
    restricted = candidates[restricted_index]
    return restricted

def find_sample_weights(group_ratios, num_samples):
    to_sample_idx = [i for i, item in enumerate(group_ratios) if item != -1]
    if to_sample_idx == []:
        return {j: 1 for j in range(len(group_ratios))}
    restricted = find_restricted(group_ratios, num_samples, to_sample_idx)
    sample_weights = {j: group_ratios[j] * num_samples[restricted] / num_samples[j] if j in to_sample_idx else 1 for j
                      in range(len(group_ratios))}
    return sample_weights
    
def load_celeba(cfg):
    root = os.path.join(cfg["data_root"],'celeba')
    group_ratios = cfg["group_ratios"]
    seed = cfg["seed"]

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    role_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
        "all": None,
    }
    
    splits_df = pd.read_csv(os.path.join(root,"list_eval_partition.txt"), sep='\s+', header=None)
    splits_df.rename(columns={0:'image_id', 1:'partition'}, inplace=True)

    fields = ['202599', 'Male', 'Eyeglasses']
    attrs_df = pd.read_csv(os.path.join(root,"list_attr_celeba.txt"), sep='\s+', usecols=fields)
    attrs_df.rename(columns={'202599':'image_id'}, inplace=True)
    
    df = pd.merge(splits_df, attrs_df, on='image_id')

    df_train = df[df['partition'] == role_map["train"]].drop(labels='partition', axis=1)
    df_val = df[df['partition'] == role_map["valid"]].drop(labels='partition', axis=1)
    df_trainval = pd.concat([df_train, df_val])
    
    df_test = df[df['partition'] == role_map["test"]].drop(labels='partition', axis=1)
    
    df_trainval = df_trainval.replace(to_replace=-1, value=0)
    df_test = df_test.replace(to_replace=-1, value=0)
    
    labels_trainval = df_trainval["Male"]
    if group_ratios:
        # don't alter the test set, refer to sample_weights.py
        label_counts = labels_trainval.value_counts(dropna=False).tolist()
        sample_weights = find_sample_weights(group_ratios, label_counts)
        # print(f"Number of samples by label (before sampling) in {role}:")
        print(f"Female: {label_counts[0]}, Male: {label_counts[1]}")

        random.seed(seed)
        idx = [random.random() <= sample_weights[label] for label in labels_trainval]
        labels = labels_trainval[idx]
        label_counts_after = labels.value_counts(dropna=False).tolist()

        print("Number of samples by label (after sampling):")
        print(f"Female: {label_counts_after[0]}, Male: {label_counts_after[1]}")
        df_trainval = df_trainval[idx]
    
    filenames_trainval = df_trainval["image_id"].tolist()
    # Male is 1, Female is 0
    y_trainval = torch.Tensor(df_trainval["Male"].values).long()
    # Wearing glasses is 1, otherwise zero
    z_trainval = torch.Tensor(df_trainval["Eyeglasses"].values).long()
    
    shape = (len(filenames_trainval), 3, 64, 64)
    x_trainval = np.zeros(shape)
    for i, file in enumerate(filenames_trainval):
        img_path = os.path.join(root, "img_align_celeba", file)
        img = PIL.Image.open(img_path)
        img = transform(img)
        x_trainval[i,:] = img
    
    filenames_test = df_test["image_id"].tolist()
    # Male is 1, Female is 0
    y_test = torch.Tensor(df_test["Male"].values).long()
    # Wearing glasses is 1, otherwise zero
    z_test = torch.Tensor(df_test["Eyeglasses"].values).long()
    
    shape = (len(filenames_test), 3, 64, 64)
    x_test = np.zeros(shape)
    for i, file in enumerate(filenames_test):
        img_path = os.path.join(root, "img_align_celeba", file)
        img = PIL.Image.open(img_path)
        img = transform(img)
        x_test[i,:] = img

    return x_trainval, x_test, y_trainval, y_test, z_trainval, z_test

def prep_compas(cfg):
    root = os.path.join(cfg["data_root"],'compas')
    seed = cfg["seed"]
    names = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                'age', 
                'c_charge_degree', 
                'c_charge_desc',
                'age_cat',
                'sex', 'race',  'is_recid']
    cat_features = [
        'c_charge_degree', 'c_charge_desc', 'age_cat' ] 
    cont_features = ['age',  'priors_count',  'juv_misd_count', 'juv_fel_count',  'juv_other_count' ]
    
    target = 'is_recid'

    df_train = pd.read_csv(os.path.join(root, 'train.csv'), names=names)
    df_test = pd.read_csv(os.path.join(root, 'test.csv'), names=names, skiprows=1)
    
    df_train[target] = df_train[target].str.replace('No', '0')
    df_train[target] = df_train[target].str.replace('Yes', '1')
    df_train[target] = df_train[target].astype(np.int64)

    df_test[target] = df_test[target].str.replace('No', '0')
    df_test[target] = df_test[target].str.replace('Yes', '1')
    df_test[target] = df_test[target].astype(np.int64)
    train_X = df_train.drop(columns= [target])
    Y = df_train[target]
    test_X = df_test.drop(columns= [target])
    Y_test =  df_test[target]
    
    stacked_df = train_X.append(test_X)
    stacked_Y = Y.append(Y_test)
    
    if cfg["protected_group"] == 'sex':
        z = stacked_df["sex"]
        z = z.map({'Male':0, 'Female':1})
    elif cfg["protected_group"] == 'race':
        z = stacked_df['race']
        z = z.map({"Black": 1, "Other": 1, "White": 0})
    # A = stacked_df[["race", "sex"]]
    #A_str = A.map({ 0:"black", 1:"other", 2:"white",})
    
    cat_features.append('race')
    cat_features.append('sex')

    stacked_df = encode_dataframe(stacked_df, cat_features, cont_features)
    mapper = scale_vars(stacked_df, None)
   
    X_train, x_test, Y_train, y_test,  z_train, z_test   = train_test_split(stacked_df, stacked_Y,  
    z, test_size=0.3, random_state=seed)
                                             
    return X_train.to_numpy(dtype='float'), x_test.to_numpy(dtype='float'), Y_train.to_numpy(dtype='int'), y_test.to_numpy(dtype='int'), z_train.to_numpy(dtype='int'), z_test.to_numpy(dtype='int')

def prep_lawschool(cfg):
    root = os.path.join(cfg["data_root"],'law_school')
    seed = cfg["seed"]

    target = 'pass_bar'

    df_train = pd.read_csv(os.path.join(root, 'lsac_new.csv'))
    
    df_train[target] = df_train[target].str.replace('Failed_or_not_attempted', '0')
    df_train[target] = df_train[target].str.replace('Passed', '1')
    df_train[target] = df_train[target].astype(np.int64)

    train_X = df_train.drop(columns= [target])
    Y = df_train[target]

    stacked_df = train_X
    stacked_Y = Y

    cat_features = ['isPartTime'] 
    cont_features = ['zfygpa', 'zgpa', 'DOB_yr',  'cluster_tier','family_income', 'lsat', 'ugpa', 'weighted_lsat_ugpa' ]
    z = stacked_df["sex"]
    z = z.map({'Male':0, 'Female':1})
    
    cat_features.append('race')
    cat_features.append('sex')

    stacked_df = encode_dataframe(stacked_df, cat_features, cont_features)
    mapper = scale_vars(stacked_df, None)

    X_train, x_test, Y_train, y_test, z_train, z_test   = train_test_split(stacked_df, stacked_Y, 
        z, test_size=0.3, random_state=seed)

    return X_train.to_numpy(dtype='float'), x_test.to_numpy(dtype='float'), Y_train.to_numpy(dtype='int'), y_test.to_numpy(dtype='int'), z_train.to_numpy(dtype='int'), z_test.to_numpy(dtype='int')

    
def to_loader(x, y, z, batch_size):
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)
    tensor_dataset = GroupLabelDataset(x.float(), y.long(), z.long())
    
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def training_loop(cfg):
    
    if cfg["dataset"] == "adult":
            x_train_val, x_test, y_train_val, y_test, z_train_val, z_test = get_adult(cfg)     
    elif cfg["dataset"] == "mnist":
        x_train_val, x_test, y_train_val, y_test, z_train_val, z_test = load_mnist(cfg)
    elif cfg["dataset"] == "celeba":
        x_train_val, x_test, y_train_val, y_test, z_train_val, z_test = load_celeba(cfg)
    elif cfg["dataset"] == "compas":
        x_train_val, x_test, y_train_val, y_test, z_train_val, z_test = prep_compas(cfg)
    elif cfg["dataset"] == "lsac":
        x_train_val, x_test, y_train_val, y_test, z_train_val, z_test = prep_lawschool(cfg)
    else:
        raise ValueError("Crossvalidation not implemented yet for the dataset {cfg['dataset']")
    
    cfg["data_shape"] = x_train_val.shape[1:]
    cfg["data_dim"] = int(np.prod(cfg["data_shape"]))
        
    test_loader = to_loader(x_test, y_test, z_test, cfg["test_batch_size"])
    
    
    kf=KFold(n_splits=5,shuffle=True,random_state=cfg["seed"])
    
    fold_acc = []
    fold_roc_auc = []
    fold_pr_auc = []
    test_metrics_full = []
    for test_metric in cfg["test_metrics"]:
        if test_metric.endswith("per_group"):
            test_metrics_full.extend([f"{test_metric[:-10]}_{k}" for k in range(cfg["num_groups"])])
            test_metrics_full.append(f"{test_metric[:-10]}_difference")
        else:
            test_metrics_full.append(test_metric)
    results_dict = {k: [] for k in test_metrics_full}    
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_val)):
        print(f"Fold {fold + 1}")
        
        x_train = x_train_val[train_idx]
        y_train = y_train_val[train_idx]
        z_train= z_train_val[train_idx]
        x_val = x_train_val[val_idx]
        y_val = y_train_val[val_idx]
        z_val= z_train_val[val_idx]
    
        cfg["train_dataset_size"] = x_train.shape[0]
        
        train_loader = to_loader(x_train, y_train, z_train, cfg["train_batch_size"])
        valid_loader = to_loader(x_val, y_val, z_val, cfg["valid_batch_size"])

        model, optimizer = create_model(cfg, cfg["device"])
    
        if cfg["method"] in ["dpsgd", "fairness-lens"]:
            privacy_engine = PrivacyEngine(accountant=cfg["accountant"])
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=cfg["epsilon"],
                target_delta=cfg["delta"],
                epochs=cfg["max_epochs"],
                max_grad_norm=cfg["l2_norm_clip"]
            )
        elif cfg["method"] == "dpsgd-global":
            privacy_engine = DPSGDGlobalPrivacyEngine(accountant=cfg["accountant"])
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=cfg["epsilon"],
                target_delta=cfg["delta"],
                epochs=cfg["max_epochs"],
                max_grad_norm=cfg["l2_norm_clip"]
            )
        elif cfg["method"] == "dpsgd-f":
            privacy_engine = DPSGDF_PrivacyEngine(accountant=cfg["accountant"])
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=cfg["epsilon"],
                target_delta=cfg["delta"],
                epochs=cfg["max_epochs"],
                max_grad_norm=0
            )
        elif cfg["method"] == "dpsgd-global-adapt":
            privacy_engine = DPSGDGlobalAdaptivePrivacyEngine(accountant=cfg["accountant"])
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=cfg["epsilon"],
                target_delta=cfg["delta"],
                epochs=cfg["max_epochs"],
                max_grad_norm=cfg["l2_norm_clip"]
            )
        else:
            # doing regular training
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=0,
                max_grad_norm=sys.float_info.max,
                poisson_sampling=False
            )
    
        evaluator = create_evaluator(
            model,
            valid_loader=valid_loader, test_loader=test_loader,
            valid_metrics=cfg["valid_metrics"],
            test_metrics=cfg["test_metrics"],
            num_classes=cfg["output_dim"],
            num_groups=cfg["num_groups"],
        )
        
        trainer = create_trainer(
            train_loader,
            valid_loader,
            test_loader,
            model,
            optimizer,
            privacy_engine,
            evaluator,
            None,
            cfg["device"],
            cfg
        )
    
        valid_results, test_results = trainer.train(write_checkpoint=False)
        fold_acc.append(valid_results['accuracy'])
        fold_roc_auc.append(valid_results["roc_auc"])
        fold_pr_auc.append(valid_results["pr_auc"])

        for test_metric in cfg["test_metrics"]:
            if test_metric.endswith("per_group"):
                for k in range(cfg["num_groups"]):
                    results_dict[f"{test_metric[:-10]}_{k}"].append(test_results[test_metric][k])
                if cfg["num_groups"] == 2:
                    results_dict[f"{test_metric[:-10]}_difference"].append(abs(test_results[test_metric][0]-test_results[test_metric][1]))
                else:
                    results_dict[f"{test_metric[:-10]}_difference"].append(abs(test_results[test_metric][2]-test_results[test_metric][8]))
            else:
                results_dict[test_metric].append(test_results[test_metric])
    
    print({"accuracy": np.mean(fold_acc), "roc_auc": np.mean(fold_roc_auc), "pr_auc": np.mean(fold_pr_auc)})
    
    results_dict_stats = {}
    for test_metric in test_metrics_full:
        results_dict_stats[f"{test_metric}_mean"] = np.mean(results_dict[test_metric])
        results_dict_stats[f"{test_metric}_std"] = np.std(results_dict[test_metric])
    if cfg["dataset"] != "mnist":
        equalized_odds = []
        for a in zip(results_dict['FP_errorrate_difference'], results_dict['FN_errorrate_difference']):
            equalized_odds.append(np.max(a))
        results_dict_stats["equalized_odds_mean"] = np.mean(equalized_odds)
        results_dict_stats["equalized_odds_std"] = np.std(equalized_odds)
    with open(f'scatterplots_results/{cfg["dataset"]}_{cfg["method"]}_{cfg["protected_group"]}_{cfg["num_samples"]}_epsilon{cfg["epsilon"]}.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        # csvwriter.writerow(results_dict_stats.keys())
        if cfg["method"] == 'regular':
            hps = [cfg["max_epochs"], cfg["lr"], cfg["train_batch_size"], cfg["activation"], cfg["optimizer"]]
        else:
            hps = [cfg["max_epochs"], cfg["lr"], cfg["train_batch_size"], cfg["activation"], cfg["optimizer"], cfg["l2_norm_clip"]]
        valid_results = [np.mean(fold_acc), np.mean(fold_roc_auc), np.mean(fold_pr_auc)]
        results = list(results_dict_stats.values())
        csvwriter.writerow(hps + valid_results + results)

def main():
    parser = argparse.ArgumentParser(description="Fairness for DP-SGD")

    parser.add_argument("--dataset", type=str, default="adult",
                        help="Dataset to train on.")
    parser.add_argument("--method", type=str, default="dpsgd",
                        choices=["regular", "dpsgd", "dpsgd-f", "fairness-lens", "dpsgd-global", "dpsgd-global-adapt"],
                        help="Method for training and clipping.")

    parser.add_argument("--config", default=[], action="append",
                        help="Override config entries. Specify as `key=value`.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_config(
        dataset=args.dataset,
        method=args.method,
    )
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}
    
    
        # ---------------------------------- CHANGED GONFIG: General ----------------------
    cfg["device"] = device
    cfg['evaluate_angles'] = False
    cfg['evaluate_hessian'] = False
    cfg["epsilon"] = 5
    cfg["delta"] = 1e-5
    del cfg["noise_multiplier"]
    cfg["data_root"] = "/data/"
    cfg["make_valid_loader"] = True
    cfg["early_stopping"] = False
    
    cfg["num_samples"] = 50 #128
    
    cfg["valid_metrics"] = ["accuracy", "accuracy_per_group", "roc_auc", "roc_auc_per_group", "pr_auc", "pr_auc_per_group"]
    cfg["test_metrics"] = ["accuracy", "accuracy_per_group", "macro_accuracy",
                           "roc_auc", "roc_auc_per_group", "pr_auc", "pr_auc_per_group", 
                            "recall", "recall_per_group", "precision", "PPV_per_group",
                            "acceptance_rate_per_group", "treatment_per_group", 
                             "FP_errorrate_per_group", "FN_errorrate_per_group",
                            ]
    
    # ---------------------------------- CHANGED GONFIG: for Adult ----------------------
    #cfg["protected_group"] = 'race'
    #cfg['group_ratios'] = [0.5, 0.5]
    
    # ---------------------------------- CHANGED GONFIG: for Compas & LSAC ----------------------
    #cfg["protected_group"] = 'race'
    #cfg["net"] = 'deoliveira'
    
    # ---------------------------------- CHANGED GONFIG: for MNIST ----------------------
    # cfg['group_ratios'] = [-1, -1, -1, -1, -1, -1, -1, -1, 0.09, -1]
    # cfg["delta"] = 1e-6
    # cfg["valid_metrics"] = ["accuracy", "accuracy_per_group", "roc_auc", "roc_auc_per_group", "pr_auc", "pr_auc_per_group"]
    # cfg["test_metrics"] = ["accuracy", "accuracy_per_group", "macro_accuracy",
                           # "roc_auc", "roc_auc_per_group", "pr_auc", "pr_auc_per_group", 
                           #  "recall", "recall_per_group", "precision", "PPV_per_group"
                           #  ]
    
    # ---------------------------------- CHANGED GONFIG: for CelebA ----------------------
    #cfg["delta"] = 1e-6
    
    # ------------------------------------------------------------------------


    # Checks group_ratios is specified correctly
    if len(cfg["group_ratios"]) != cfg["num_groups"]:
        raise ValueError(
            "Number of group ratios, {}, not equal to number of groups of {}, {}"
                .format(len(cfg["group_ratios"]), cfg["protected_group"], cfg["num_groups"])
        )

    if any(x > 1 or (x < 0 and x != -1) for x in cfg["group_ratios"]):
        raise ValueError("All elements of group_ratios must be in [0,1]. Indicate no sampling with -1.")

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10 * "-" + "-cfg--" + 10 * "-")
    pp.pprint(cfg)

    # Set random seeds based on config
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    test_metrics_full = []
    test_metrics_final = []
    for test_metric in cfg["test_metrics"]:
        if test_metric.endswith("per_group"):
            test_metrics_full.extend([f"{test_metric[:-10]}_{k}" for k in range(cfg["num_groups"])])
            test_metrics_full.append(f"{test_metric[:-10]}_difference")
        else:
            test_metrics_full.append(f"{test_metric}")
    for test_metric in test_metrics_full:
        test_metrics_final.extend([f"{test_metric}_mean", f"{test_metric}_std"])
    if cfg["dataset"] != "mnist":
        test_metrics_final.extend(["equalized_odds_mean", "equalized_odds_std"])
    with open(f'scatterplots_results/{cfg["dataset"]}_{cfg["method"]}_{cfg["protected_group"]}_{cfg["num_samples"]}_epsilon{cfg["epsilon"]}.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        if cfg["method"] == 'regular':
            csvwriter.writerow(["epochs", "lr", "batch_size", "activation", "optimizer"] + ["valid_accuracy", "valid_roc_auc", "valid_pr_auc"] + test_metrics_final)
        else:
            csvwriter.writerow(["epochs", "lr", "batch_size", "activation", "optimizer", "max_grad_norm"] + ["valid_accuracy", "valid_roc_auc", "valid_pr_auc"] + test_metrics_final)
    
    if cfg["dataset"] == 'adult':
        hyperparams = pd.read_csv('hp_list_gridsearch_esipova.csv')
    elif cfg["dataset"] == 'lsac':
        hyperparams = pd.read_csv('hp_list_gridsearch_esipova.csv')
    elif cfg["dataset"] == 'compas':
        hyperparams = pd.read_csv('hp_list_gridsearch_esipova.csv')
    elif cfg['dataset'] == 'celeba':
        hyperparams = pd.read_csv('hp_list_100_celeba.csv')
    elif cfg["dataset"] == 'mnist':
        hyperparams = pd.read_csv('hp_list_100_mnist.csv')
    
    if cfg["method"] == "regular":
        for i in range(cfg["num_samples"]):
            cfg["lr"] = float(hyperparams['lr'][i])
            cfg["optimizer"] = hyperparams['optimizer'][i]
            cfg["train_batch_size"] = int(hyperparams['batch_size'][i])
            cfg["activation"] = hyperparams['activation'][i]
            cfg['max_epochs'] = int(hyperparams['epochs'][i])
            
            training_loop(cfg)
    
    else:
        for i in range(cfg["num_samples"]):
            cfg["lr"] = float(hyperparams['lr'][i])
            cfg["optimizer"] = hyperparams['optimizer'][i]
            cfg["train_batch_size"] = int(hyperparams['batch_size'][i])
            cfg["activation"] = hyperparams['activation'][i]
            cfg['max_epochs'] = int(hyperparams['epochs'][i])
            for max_grad_norm in [0.01, 0.1, 1.]:
                cfg["l2_norm_clip"] = max_grad_norm
        
                training_loop(cfg)
    
    
if __name__ == "__main__":
    main()
