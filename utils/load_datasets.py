import io
import re
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def parse_keel_dat(dat_file):
    with open(dat_file, "r") as fp:
        data = fp.read()
        header, payload = data.split("@data\n")

    attributes = re.findall(
        r"@[Aa]ttribute (.*?)[ {](integer|real|.*)", header)
    output = re.findall(r"@[Oo]utput[s]? (.*)", header)

    dtype_map = {"integer": np.int, "real": np.float}

    columns, types = zip(*attributes)
    types = [*map(lambda _: dtype_map.get(_, np.object), types)]
    dtype = dict(zip(columns, types))

    # Replace missing values with NaN in datasets
    data = pd.read_csv(io.StringIO(payload), names=columns, dtype=dtype, na_values=[" <null>"])

    # Replace NaN values with most frequent values
    if data.isnull().values.any():
        imputer = SimpleImputer(strategy='most_frequent')
        imputer = imputer.fit(data)
        data = imputer.transform(data)
        data = pd.DataFrame(data, columns=columns)

    if not output:  # if it was not found
        output = columns[-1]
    target = data[output]
    data.drop(labels=output, axis=1, inplace=True)

    return data, target


def prepare_X_y(data, target):
    class_encoder = LabelEncoder()
    target = class_encoder.fit_transform(target.values.ravel())
    return data.values, target


def load_dataset(dataset_path, return_X_y=True):
    data, target = parse_keel_dat(dataset_path)
    if return_X_y:
        return prepare_X_y(data, target)
    return Bunch(data=data, target=target, filename=dataset_path)
