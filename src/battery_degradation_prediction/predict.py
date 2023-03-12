"""predict the first lifetime of the battery given the first cycle"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from sklearn.metrics import mean_squared_error
from battery_degradation_prediction.model import Transformer, TransformerReduction
from battery_degradation_prediction.preprocessing import get_clean_data, get_cycle_data
from battery_degradation_prediction.load_data import (
    load_unsupervised_data,
    load_supervised_data,
)
from battery_degradation_prediction.train_reduction import inference


def get_window(data, window_size: int, stride: int):
    """TODO"""
    if len(data) < window_size + 1:
        print(
            f"Error: Data array length ({len(data)}) \
            is not long enough to generate windows of size {window_size}."
        )
        return (None, None)

    num_windows = int((len(data) - window_size) / stride)
    windows = []
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows.append(data[start_idx:end_idx, :])
    return np.array(windows)


def predict(supervised_model, unsupervised_model, cycle_window, future_cycle=10):
    """TODO"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised_model = supervised_model.to(device)
    unsupervised_model = unsupervised_model.to(device)
    x_pred = inference(unsupervised_model, future_cycle, cycle_window)
    num_rows = 100
    num_features = 4
    window_size = 5
    x_pred = x_pred.cpu()[0, 0].view(num_rows, num_features).detach().numpy()
    x_windows = get_window(x_pred, window_size, 1)  # [# of windows, window_size, num_features]
    x_windows = torch.from_numpy(x_windows).type(torch.float32).to(device)

    pred = supervised_model(x_windows)
    pred = pred.cpu().detach().numpy()
    return x_windows.cpu().detach().numpy(), pred


def load_model(MODEL, model_path: str, model_hyper: Tuple):
    """TODO"""
    model = MODEL(*model_hyper)
    model.load_state_dict(torch.load(model_path))
    return model


def main():
    """TODO"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised_model_path = "./supervised_model"
    unsupervised_model_path = "./unsupervised_model"
    window_size = 5
    num_features = 4
    input_shape = (window_size, num_features)
    d_model = 8
    nhead = 2
    num_layers = 2
    output_size = 1
    dropout = 0.2
    model_hyper = (input_shape, d_model, nhead, num_layers, output_size, dropout)
    supervised_model = load_model(Transformer, supervised_model_path, model_hyper)

    num_rows = 100
    input_shape = (window_size - 1, num_features * num_rows)
    d_model = 8
    nhead = 2
    num_layers = 2
    dropout = 0.2
    latent_size = 10
    model_hyper = (input_shape, d_model, nhead, num_layers, latent_size, dropout)
    unsupervised_model = load_model(TransformerReduction, unsupervised_model_path, model_hyper)

    data_path = "../../data/B0005.csv"
    df_discharge = get_clean_data(data_path, int(1e7))

    feature_names = [
        "cycle",
        "voltage_measured",
        "current_measured",
        "temperatrue_measured",
        "capcity_during_discharge",
        "capacity",
    ]
    test_size = 0.2
    (_, _), (_, _), X_scaler, y_scaler = load_supervised_data(
        df_discharge, test_size, feature_names, window_size
    )
    (dev_x, dev_y), (test_x, test_y), _ = load_unsupervised_data(
        df_discharge, test_size, feature_names, randomize=False
    )
    print(f"dev_x = {dev_x.shape}, dev_y = {dev_y.shape}")
    print(f"test_x = {test_x.shape}, test_y = {test_y.shape}")
    dev_x = torch.from_numpy(dev_x).type(torch.float32).to(device)
    future_cycles = range(10, 161, 10)
    gif_image = []
    for future_cycle in future_cycles:
        cycle_data = get_cycle_data(df_discharge, future_cycle)
        true = cycle_data[["voltage_measured", "capcity_during_discharge"]]
        x_windows, pred = predict(supervised_model, unsupervised_model, dev_x[:1], future_cycle)
        x_windows_shape = x_windows.shape
        pred_inv = y_scaler.inverse_transform(pred)
        x_windows_inv = X_scaler.inverse_transform(x_windows.reshape(-1, num_features)).reshape(
            x_windows_shape
        )
        capacity_pred = np.concatenate((x_windows_inv[0, :, -1], pred_inv[:-1, 0]))
        voltage_pred = np.concatenate((x_windows_inv[:-1, 0, 0], x_windows_inv[-1, :, 0]))
        capacity_true = true["capcity_during_discharge"]
        voltage_true = true["voltage_measured"]
        fig, ax = plt.subplots(figsize=(5, 5))
        mse = mean_squared_error(capacity_true[: len(capacity_pred)], capacity_pred)
        ax.scatter(
            capacity_pred,
            voltage_pred,
            label="Pred",
            s=6,
            facecolors="none",
            edgecolors="k",
        )
        ax.plot(capacity_true, voltage_true, "--k", label="GT")
        ax.annotate(f"MSE = {mse:1.5f}", xy=(0.6, 0.05), xycoords="axes fraction")
        plt.title(f"Predict cycle {future_cycle}")
        plt.xlabel("capacity")
        plt.ylabel("voltage")
        plt.legend()
        plt.xlim([-0.1, 2.0])
        plt.ylim([2.5, 4.3])

        gif_image.append(get_gif(fig))
    filename = "future_cycle"
    imageio.mimsave(f"{filename}.gif", gif_image, duration=0.75)
    return gif_image


def get_gif(fig):
    """TODO"""
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    height = fig.canvas.get_width_height()[-1]
    width = fig.canvas.get_width_height()[0]
    image = image.reshape((height, width) + (3,))
    return image


if __name__ == "__main__":
    main()
