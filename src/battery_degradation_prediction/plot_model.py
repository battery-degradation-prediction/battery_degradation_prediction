"""Plot module"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
from battery_degradation_prediction.window import windowing


def plot_train_val_loss(histories):
    """TODO"""
    epochs = len(histories[0]["train_loss"])
    mean_train_loss = np.mean([history["train_loss"] for history in histories], axis=0)
    mean_val_loss = np.mean([history["val_loss"] for history in histories], axis=0)
    # std_train_loss = np.std([history["train_loss"] for history in histories], axis=0)
    # std_val_loss = np.std([history["val_loss"] for history in histories], axis=0)
    _, ax = plt.subplots(figsize=(5, 5))
    """
    ax.errorbar(range(epochs), mean_train_loss,
                yerr=std_train_loss, label='train loss', markersize=6, ls='-')
    ax.errorbar(range(epochs), mean_val_loss,
                yerr=std_val_loss, label='val loss', markersize=6, ls='--')
    """
    ax.plot(range(epochs), mean_train_loss, "-o", label="train loss", markersize=4)
    ax.plot(range(epochs), mean_val_loss, "-o", label="val loss", markersize=4)
    plt.title("Loss versus epochs")
    plt.xlabel("Epochs [-]")
    plt.ylabel("Loss [-]")
    plt.legend()
    plt.show()
    # print(f"mean_train_loss = {mean_train_loss[0]:2.5f} ± {std_train_loss[0]:2.3f}")
    # print(f"mean_val_loss = {mean_val_loss[0]:2.5f} ± {std_val_loss[0]:2.3f}")


def parity_plot(test_y, predictions):
    """TODO"""
    _, ax = plt.subplots(figsize=(5, 5))
    r2 = r2_score(test_y, predictions)

    ax.scatter(test_y, predictions, s=6, facecolors="none", edgecolors="k", label="Transformer")
    ax.plot([-100, 100], [-100, 100], "--k", label="Perfect predictions")
    ax.set_title("Predicted capacity")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_xlim(
        [
            np.min(np.concatenate((test_y, predictions))) - 0.1,
            np.max(np.concatenate((test_y, predictions))) + 0.1,
        ]
    )
    ax.set_ylim(
        [
            np.min(np.concatenate((test_y, predictions))) - 0.1,
            np.max(np.concatenate((test_y, predictions))) + 0.1,
        ]
    )
    ax.annotate(f"R² = {r2:.2f}", xy=(0.80, 0.05), xycoords="axes fraction")
    plt.legend()
    plt.show()


def plot_future_capacity(
    df_feature, input_shape, model, X_scaler, y_scaler, future_cycle, window_size
):
    """TODO"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for idx, group in enumerate(df_feature.groupby("cycle")):
        data = []
        data_y = []
        if idx == future_cycle:
            cycle_data = group[1].iloc[:, 1:]
            cycle_data_windows, capacities = windowing(cycle_data, window_size, 1)
            for (test_x, test_y) in zip(cycle_data_windows, capacities):
                data.append(test_x[:, :-1])
                data_y.append(test_y)
            data = np.asarray(data)
            data = X_scaler.transform(np.reshape(data, (-1, input_shape[-1])))
            data = np.reshape(data, (-1, input_shape[0], input_shape[-1]))
            pred = model(torch.from_numpy(data).type(torch.float32).to(device))
            data = X_scaler.inverse_transform(np.reshape(data, (-1, input_shape[-1])))
            data = np.reshape(data, (-1, input_shape[0], input_shape[-1]))
            pred = pred.cpu().detach().numpy()
            pred = y_scaler.inverse_transform(pred).reshape(
                -1,
            )

            plt.figure(figsize=(6, 6))
            plt.plot(data_y, data[:, 0, 0], "--k", label="True")
            plt.scatter(
                pred,
                data[:, 0, 0],
                s=6,
                facecolors="none",
                edgecolors="k",
                label="Prediction",
            )
            plt.xlabel("cummulative capacity")
            plt.ylabel("voltage measure")
            plt.title(f"Test cycle {idx}")
            plt.legend()
            plt.show()
            break


def plot_future_capacities(
    df_feature, input_shape, models, X_scaler, y_scaler, future_cycle, window_size
):
    """TODO"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for idx, group in enumerate(df_feature.groupby("cycle")):
        data = []
        data_y = []
        if idx == future_cycle:
            cycle_data = group[1].iloc[:, 1:]
            cycle_data_windows, capacities = windowing(cycle_data, window_size, 1)
            for (test_x, test_y) in zip(cycle_data_windows, capacities):
                data.append(test_x[:, :-1])
                data_y.append(test_y)
            data = np.asarray(data)
            data = X_scaler.transform(np.reshape(data, (-1, input_shape[-1])))
            data = np.reshape(data, (-1, input_shape[0], input_shape[-1]))
            break
    cycles = [10, 50, 100, 159]
    edgecolors = ["k", "r", "b", "g"]
    markers = ["<", "^", "s", "o"]
    plt.figure(figsize=(6, 6))
    for cycle_index, model in enumerate(models):
        model.eval()
        pred = model(torch.from_numpy(data).type(torch.float32).to(device))
        data_inv = X_scaler.inverse_transform(np.reshape(data, (-1, input_shape[-1])))
        data_inv = np.reshape(data_inv, (-1, input_shape[0], input_shape[-1]))
        pred = pred.cpu().detach().numpy()
        pred = y_scaler.inverse_transform(pred).reshape(
            -1,
        )
        plt.scatter(
            pred,
            data_inv[:, 0, 0],
            s=12,
            marker=markers[cycle_index],
            facecolors="none",
            edgecolors=edgecolors[cycle_index],
            label=f"Prediction {cycles[cycle_index]}",
        )

    plt.plot(data_y, data_inv[:, 0, 0], "--k", label="True")
    plt.xlabel("cummulative capacity")
    plt.ylabel("voltage measure")
    plt.title(f"Test cycle {future_cycle}")
    plt.legend()
    plt.show()
