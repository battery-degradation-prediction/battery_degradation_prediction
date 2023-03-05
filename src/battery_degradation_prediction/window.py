import numpy as np


def windowing(data, window_size, stride):
    """
    Generate windows and labels from a time series dataset.

    Parameters:
        data (ndarray): The time series data.
        window_size (int): The size of the sliding window.
        stride (int): The stride of the sliding window.

    Returns:
        Tuple of ndarrays: A tuple containing the windows and labels arrays, or None if the data array is too short.
    """
    if len(data) < window_size + 1:
        print(f"Error: Data array length ({len(data)}) is not long enough to generate windows of size {window_size}.")
        return (None, None)

    num_windows = int((len(data) - window_size) / stride)
    windows = np.zeros((num_windows, window_size))
    labels = np.zeros((num_windows,))

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
        labels[i] = data[end_idx]
    return windows, labels


if __name__ == '__main__':
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
