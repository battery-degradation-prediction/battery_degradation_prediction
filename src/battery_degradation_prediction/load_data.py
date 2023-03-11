"""load data module"""
from typing import Tuple
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.utils import shuffle
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.window import windowing, windowing_numpy


def dev_test_split_supervised(df_discharge: pd.DataFrame, test_size: float, window_size: int=5):
    """TODO"""
    dev_x_data = []
    test_x_data = []
    dev_y_data = []
    test_y_data = []
    num_train = int(len(df_discharge.groupby("cycle")) * (1-test_size))
    random_list = shuffle(range(len(df_discharge.groupby("cycle"))))
    #random_list = range(len(df_discharge.groupby("cycle")))
    dev_index, test_index = random_list[:num_train], random_list[num_train:]
    
    for idx, group in enumerate(df_discharge.groupby("cycle")):
        cycle_data = group[1].iloc[:, 1:]
        if idx in dev_index:
            cycle_data_windows, capacities = windowing(cycle_data, window_size, 1)
            for (dev_x, dev_y) in zip(cycle_data_windows, capacities):
                dev_x_data.append(dev_x[:,:-1])
                dev_y_data.append(dev_y)
        else:
            cycle_data_windows, capacities = windowing(cycle_data, window_size, 1)
            for (test_x, test_y) in zip(cycle_data_windows, capacities):
                test_x_data.append(test_x[:,:-1])
                test_y_data.append(test_y)
        
    dev_x_data, dev_y_data = (np.asarray(dev_x_data), np.asarray(dev_y_data))
    test_x_data, test_y_data= (np.asarray(test_x_data), np.asarray(test_y_data))
    return (
        dev_x_data,
        dev_y_data[..., np.newaxis],
        test_x_data,
        test_y_data[..., np.newaxis],
    )

def dev_test_split_unsupervised(df_discharge: pd.DataFrame, test_size: float, window_size: int=5, randomize=True):
    """TODO"""
    num_train = int(len(df_discharge.groupby("cycle")) * (1-test_size))
    num_data_per_group = df_discharge.groupby("cycle").count().min().iloc[0]
    cycle_data = np.zeros((len(df_discharge.groupby("cycle")), num_data_per_group, len(df_discharge.iloc[0]))) # [# of cycles, # of data per cycle, # of features]
    for idx, group in enumerate(df_discharge.groupby("cycle")):
        cycle_data[idx] = group[1].iloc[:num_data_per_group]
    windows = windowing_numpy(cycle_data, window_size, 1)

    if randomize: 
        random_list = shuffle(range(len(windows)))
    else: 
        random_list = range(len(windows))
    dev_index, test_index = random_list[:num_train], random_list[num_train:]
    
    dev_x_data = windows[dev_index]
    test_x_data = windows[test_index]
    assert len(dev_x_data) + len(test_x_data) == len(windows), "# of dev + # of test != # of windows"
    return dev_x_data, test_x_data


def standard_transform_y(dev_y, test_y):
    """TODO"""
    standard_scaler_y = sklearn.preprocessing.StandardScaler()
    dev_y = standard_scaler_y.fit_transform(dev_y)
    test_y = standard_scaler_y.transform(test_y)
    return dev_y, test_y, standard_scaler_y


def standard_transform_x(dev_x, test_x):
    """TODO"""
    standard_scaler_X = sklearn.preprocessing.StandardScaler()
    init_dev_shape = dev_x.shape
    init_test_shape = test_x.shape

    fit_data = np.concatenate((dev_x[0], dev_x[1:, -1]))
    standard_scaler_X.fit(np.reshape(fit_data, (-1, init_dev_shape[-1])))
    dev_x = standard_scaler_X.transform(np.reshape(dev_x, (-1, init_dev_shape[-1])))
    dev_x = np.reshape(dev_x, (init_dev_shape[0], init_dev_shape[1], -1))
    test_x = standard_scaler_X.transform(np.reshape(test_x, (-1, init_dev_shape[-1])))
    test_x = np.reshape(test_x, (init_test_shape[0], init_test_shape[1], -1))
    return dev_x, test_x, standard_scaler_X

def load_supervised_data(
    df_discharge: pd.DataFrame, test_size: float, feature_names: list[str], window_size:int=5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""
    df_feature = df_discharge[feature_names]
    dev_x, dev_y, test_x, test_y = dev_test_split_supervised(df_feature, test_size, window_size)
    dev_x, test_x, X_scaler = standard_transform_x(dev_x, test_x)
    dev_y, test_y, y_scaler = standard_transform_y(dev_y, test_y)
    return (dev_x, dev_y), (test_x, test_y), X_scaler, y_scaler

def load_unsupervised_data(
    df_discharge: pd.DataFrame, test_size: float, feature_names: list[str], window_size:int=5, randomize=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""
    df_feature = df_discharge[feature_names]
    dev_x_data, test_x_data = dev_test_split_unsupervised(df_feature, test_size, window_size, randomize)
    dev_x_data, test_x_data, X_scaler = standard_transform_x(dev_x_data, test_x_data)
    dev_y = dev_x_data[:, 1:]
    dev_x = dev_x_data[:, :-1]
    test_y = test_x_data[:, 1:]
    test_x = test_x_data[:, :-1]
    return (dev_x, dev_y), (test_x, test_y), X_scaler


def main():
    """TODO"""
    path = "../../data/B0005.csv"
    df_discharge = get_clean_data(path, int(5e6))
    feature_names = [
        "cycle",
        "voltage_measured",
        "current_measured",
        "temperatrue_measured",
        "capcity_during_discharge",
        "capacity"
    ]
    test_size = 0.3
    #(dev_x, dev_y), (test_x, test_y), X_scaler = load_unsupervised_data(df_discharge, test_size, feature_names)
    (dev_x, dev_y), (test_x, test_y), X_scaler, y_scaler = load_supervised_data(df_discharge, test_size, feature_names)
    print("===== dev_x, dev_y =====")
    print(dev_x.shape, dev_y.shape)
    print("===== test_x, test_y =====")
    print(test_x.shape, test_y.shape)

if __name__ == "__main__":
    main()
