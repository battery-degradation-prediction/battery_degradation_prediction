"""load data module"""
from typing import Tuple
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from preprocessing import get_clean_data
from window import windowing, windowing_numpy


def dev_test_split(df_discharge: pd.DataFrame, test_size: float):
    """TODO"""
    dev_x_data = []
    test_x_data = []
    dev_y_data = []
    test_y_data = []
    for group in df_discharge.groupby("cycle"):
        cycle_data = group[1].iloc[:, 1:]
        # print(cycle_data)
        cycle_data_windows, capacities = windowing(cycle_data, 5, 1)
        (
            dev_windows,
            test_windows,
            dev_labels,
            test_labels,
        ) = sklearn.model_selection.train_test_split(
            cycle_data_windows, capacities, test_size=test_size
        )
        # print(dev_labels.shape)
        for (dev_x, dev_y) in zip(dev_windows, dev_labels):
            dev_x_data.append(dev_x)
            dev_y_data.append(dev_y)
        for (test_x, test_y) in zip(test_windows, test_labels):
            test_x_data.append(test_x)
            test_y_data.append(test_y)
    dev_x_data, dev_y_data, test_x_data, test_y_data = (
        np.asarray(dev_x_data),
        np.asarray(dev_y_data),
        np.asarray(test_x_data),
        np.asarray(test_y_data),
    )
    return (
        dev_x_data,
        dev_y_data[..., np.newaxis],
        test_x_data,
        test_y_data[..., np.newaxis],
    )

def dev_test_split_2(df_discharge: pd.DataFrame, test_size: float):
    """TODO"""
    dev_x_data = []
    test_x_data = []
    dev_y_data = []
    test_y_data = []
    num_train = int(len(df_discharge.groupby("cycle")) * (1-test_size))
    current_num_train = 0
    for idx, group in enumerate(df_discharge.groupby("cycle")):
        cycle_data = group[1].iloc[:, 1:]
        if idx <= num_train:
            cycle_data_windows, capacities = windowing(cycle_data, 5, 1)
            for (dev_x, dev_y) in zip(cycle_data_windows, capacities):
                dev_x_data.append(dev_x[:,:-1])
                dev_y_data.append(dev_y)
            current_num_train += len(cycle_data)
        else:
            cycle_data = group[1].iloc[:, 1:]
            cycle_data_windows, capacities = windowing(cycle_data, 5, 1)
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

def dev_test_split_3(df_discharge: pd.DataFrame, test_size: float):
    """TODO"""
    num_train = int(len(df_discharge.groupby("cycle")) * (1-test_size))
    num_data_per_group = df_discharge.groupby("cycle").count().min().iloc[0]
    cycle_data = np.zeros((len(df_discharge.groupby("cycle")), num_data_per_group, len(df_discharge.iloc[0]))) # [# of cycles, # of data per cycle, # of features]
    for idx, group in enumerate(df_discharge.groupby("cycle")):
        cycle_data[idx] = group[1].iloc[:num_data_per_group]
    windows, x_labels, y_labels = windowing_numpy(cycle_data, 5, 1)
    #windows = np.reshape(windows, (windows.shape[0], windows.shape[1], -1))
    dev_x_data, dev_x_labels, dev_y_labels = (windows[:num_train], x_labels[:num_train], y_labels[:num_train, np.newaxis])
    test_x_data, test_x_labels, test_y_labels = (windows[num_train:], x_labels[num_train:], y_labels[num_train:, np.newaxis])
    assert len(dev_x_data) + len(test_x_data) == len(windows), "# of dev + # of test != # of windows"
    return (dev_x_data, dev_x_labels, dev_y_labels), (test_x_data, test_x_labels, test_y_labels)


def standard_transform_y(dev_y, test_y):
    """TODO"""
    standard_scaler_y = sklearn.preprocessing.StandardScaler()
    dev_y = standard_scaler_y.fit_transform(dev_y)
    test_y = standard_scaler_y.transform(test_y)
    return dev_y, test_y, standard_scaler_y


def standard_transform_x(dev_x, dev_x_labels,
                         test_x, test_x_labels):
    """TODO"""
    standard_scaler_X = sklearn.preprocessing.StandardScaler()
    standard_scaler_x_label = sklearn.preprocessing.StandardScaler()
    init_dev_shape = dev_x.shape
    
    init_test_shape = test_x.shape
    num_dev_data = init_dev_shape[0]
    num_test_data = init_test_shape[0]
    print("dev_init = ", init_dev_shape)
    #dev_x = standard_scaler_X.fit(np.reshape(dev_x, (-1, init_dev_shape[-1])))
    dev_x = standard_scaler_X.fit_transform(np.reshape(dev_x, (num_dev_data, -1)))
    print("dev = ", np.reshape(dev_x, (num_dev_data, -1)).shape)
    dev_x = np.reshape(dev_x, init_dev_shape)
    dev_x_labels = standard_scaler_x_label.fit_transform(np.reshape(dev_x_labels, (num_dev_data, -1)))
    print("dev_label = ", np.reshape(dev_x_labels, (num_dev_data, -1)).shape)
    dev_x_labels = np.reshape(dev_x_labels, (num_dev_data, init_dev_shape[2]))

    test_x = standard_scaler_X.transform(np.reshape(test_x, (num_test_data, -1)))
    test_x = np.reshape(test_x, init_test_shape)
    test_x_labels = standard_scaler_x_label.transform(np.reshape(test_x_labels, (num_test_data, -1)))
    test_x_labels = np.reshape(test_x_labels, (num_test_data, init_test_shape[2]))

    return (dev_x, dev_x_labels), (test_x, test_x_labels), standard_scaler_X

def standard_transform_x_label(dev_x_labels,
                               test_x_labels):
    """TODO"""
    standard_scaler_x_label = sklearn.preprocessing.StandardScaler()
    init_dev_shape = dev_x_labels.shape
    init_test_shape = test_x_labels.shape
    num_dev_data = init_dev_shape[0]
    num_test_data = init_test_shape[0]

    dev_x_labels = standard_scaler_x_label.fit_transform(np.reshape(dev_x_labels, (num_dev_data, -1)))
    dev_x_labels = np.reshape(dev_x_labels, init_dev_shape)

    test_x_labels = standard_scaler_x_label.transform(np.reshape(test_x_labels, (num_test_data, -1)))
    test_x_labels = np.reshape(test_x_labels, init_test_shape)

    return dev_x_labels, test_x_labels, standard_scaler_x_label


def load_data(
    df_discharge: pd.DataFrame, test_size: float, feature_names: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""
    df_feature = df_discharge[feature_names]
    (dev_x_data, dev_x_labels, dev_y_labels), (test_x_data, test_x_labels, test_y_labels) = dev_test_split_3(df_feature, test_size)
    (dev_x_data, dev_x_labels), (test_x_data, test_x_labels), X_scaler = standard_transform_x(dev_x_data, dev_x_labels, test_x_data, test_x_labels)
    dev_y_labels, test_y_labels, y_scaler = standard_transform_y(dev_y_labels, test_y_labels)
    return (dev_x_data, dev_x_labels, dev_y_labels), (test_x_data, test_x_labels, test_y_labels), X_scaler, y_scaler

def load_data_reduction(
    df_discharge: pd.DataFrame, test_size: float, feature_names: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""
    df_feature = df_discharge[feature_names]
    (dev_x_data, dev_x_labels, dev_y_labels), (test_x_data, test_x_labels, test_y_labels) = dev_test_split_3(df_feature, test_size)
    reduction_dev_x_labels, reduction_test_x_labels, X_scaler = standard_transform_x_label(dev_x_labels, test_x_labels)
    return reduction_dev_x_labels, reduction_test_x_labels, X_scaler

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
    (dev_x, dev_x_labels, dev_y), (test_x, test_x_labels, test_y), X_scaler, y_scaler = load_data(df_discharge, test_size, feature_names)
    reduction_dev_x_labels, reduction_test_x_labels, _ = load_data_reduction(df_discharge, test_size, feature_names)
    print("===== reduction_dev, reduction_test =====")
    print(reduction_dev_x_labels.shape, reduction_test_x_labels.shape)    
    print("===== dev_x, dev_x_labels, dev_y =====")
    print(dev_x.shape, dev_x_labels.shape, dev_y.shape)
    print("===== test_x, test_x_labels, test_y =====")
    print(test_x.shape, test_x_labels.shape, test_y.shape)

if __name__ == "__main__":
    main()
