import numpy as np
import pandas as pd


def windowing(data: pd.DataFrame, window_size: int, stride: int):
    """
    Generate windows and labels from a time series dataset.

    Parameters:
        data (DataFrame): The time series data.
        window_size (int): The size of the sliding window.
        stride (int): The stride of the sliding window.

    Returns:
        pd.DataFrame
    """
    if len(data) < window_size + 1:
        print(
            f"Error: Data array length ({len(data)}) is not long enough to generate windows of size {window_size}."
        )
        return (None, None)

    num_windows = int((len(data) - window_size) / stride)
    windows = []
    labels = []
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        window = data.iloc[start_idx:end_idx]
        label = data["capcity_during_discharge"].iloc[end_idx]
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)


if __name__ == "__main__":
    # Generate some sample time series data
    data = np.arange(100)
    print(data)
    # Define the window size and stride
    window_size = 50
    stride = 10

    # Generate the windows and labels
    window_data = windowing(data, window_size, stride)
    if window_data is not None:
        windows, labels = window_data
        # Print the first 5 windows and labels
        print("Windows:\n", windows[:])
        print("Labels:\n", labels[:])
