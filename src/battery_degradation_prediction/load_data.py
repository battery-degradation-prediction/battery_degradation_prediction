"""load data module"""
from typing import Tuple
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.window import windowing


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


def min_max_transform(dev_x, dev_y, test_x, test_y):
    """TODO"""
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    init_dev_shape = dev_x.shape
    init_test_shape = test_x.shape
    num_dev_data = init_dev_shape[0]
    num_test_data = init_test_shape[0]
    dev_x = min_max_scaler.fit_transform(np.reshape(dev_x, (num_dev_data, -1)))
    dev_x = np.reshape(dev_x, init_dev_shape)
    test_x = min_max_scaler.transform(np.reshape(test_x, (num_test_data, -1)))
    test_x = np.reshape(test_x, init_test_shape)

    dev_y = min_max_scaler.fit_transform(dev_y)
    test_y = min_max_scaler.transform(test_y)
    return dev_x, dev_y, test_x, test_y, min_max_scaler


def load_data(
    df_discharge: pd.DataFrame, test_size: float, feature_names: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""
    df_feature = df_discharge[feature_names]
    dev_x, dev_y, test_x, test_y = dev_test_split(df_feature, test_size)
    dev_x, dev_y, test_x, test_y, y_scaler = min_max_transform(
        dev_x, dev_y, test_x, test_y
    )
    return dev_x, dev_y, test_x, test_y, y_scaler


def main():
    """TODO"""
    path = "../../data/B0005.csv"
    df_discharge = get_clean_data(path)
    feature_names = [
        "cycle",
        "voltage_measured",
        "current_measured",
        "temperatrue_measured",
        "capcity_during_discharge",
    ]
    test_size = 0.1
    dev_x, dev_y, test_x, test_y, _ = load_data(df_discharge, test_size, feature_names)


if __name__ == "__main__":
    main()
