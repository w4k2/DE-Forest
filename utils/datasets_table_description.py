import os
import numpy as np
import warnings
from pathlib import Path
from .load_datasets import load_dataset


warnings.filterwarnings("ignore")


def calc_imbalance_ratio(X, y):
    unique, counts = np.unique(y, return_counts=True)

    if len(counts) == 1:
        raise ValueError("Only one class in procesed data.")
    elif counts[0] > counts[1]:
        majority_name = unique[0]
        minority_name = unique[1]
    else:
        majority_name = unique[1]
        minority_name = unique[0]

    minority_ma = np.ma.masked_where(y == minority_name, y)
    minority = X[minority_ma.mask]

    majority_ma = np.ma.masked_where(y == majority_name, y)
    majority = X[majority_ma.mask]

    imbalance_ratio = majority.shape[0]/minority.shape[0]

    return imbalance_ratio


def make_description_table(DATASETS_DIR="./datasets"):
    print(DATASETS_DIR)
    dataset_names = []
    X_all = []
    y_all = []
    imbalance_ratios = []
    for root, _, files in os.walk(DATASETS_DIR):
        for filename in filter(lambda _: _.endswith('.dat'), files):
            dataset_path = os.path.join(root, filename)
            dataset_name = Path(dataset_path).stem
            dataset_names.append(dataset_name)

            X, y = load_dataset(dataset_path)
            IR = calc_imbalance_ratio(X, y)
            imbalance_ratios.append(IR)
            X_all.append(X)
            y_all.append(y)

    IR_argsorted = np.argsort(imbalance_ratios)
    if not os.path.exists("../results/tables/"):
        os.makedirs("../results/tables/")
    with open("../results/tables/datasets.tex", "w+") as file:
        for id, arg in enumerate(IR_argsorted):
            id += 1
            number_of_features = X_all[arg].shape[1]
            number_of_objects = len(y_all[arg])
            ds_name = dataset_names[arg].replace("_", "\\_")
            print("%d & \\emph{%s} & %0.2f & %d & %d \\\\" % (id, ds_name, imbalance_ratios[arg], number_of_objects, number_of_features), file=file)

# make_description_table(DATASETS_DIR="./datasets")