"""train module"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.load_data import load_data
from battery_degradation_prediction.model import Net, Transformer
from battery_degradation_prediction.evaluate import evaluate


def train(X_train, y_train, model, epochs, optimizer, criterion):
    """TODO"""
    for epoch in range(epochs):
        optimizer.zero_grad()  # zero the gradient buffers
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()  # Does the update
        if epoch % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            print(f"Epoch = {epoch}, loss = {loss:2.5f}")
    return model


def parity_plot(test_y, predictions):
    """TODO"""
    _, ax = plt.subplots(figsize=(5, 5))
    r2 = r2_score(test_y, predictions)

    ax.scatter(
        test_y, predictions, s=6, facecolors="none", edgecolors="k", label="Transformer"
    )
    ax.plot([-5, 5], [-5, 5], "--k", label="Perfect predictions")
    ax.set_title("Predicted capacity")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_xlim([np.min(test_y) - 0.1, np.max(test_y) + 0.1])
    ax.set_ylim([np.min(test_y) - 0.1, np.max(test_y) + 0.1])
    ax.annotate(f"R² = {r2:.2f}", xy=(0.80, 0.05), xycoords="axes fraction")
    plt.legend()
    plt.show()


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
    dev_x, dev_y, test_x, test_y, y_scaler = load_data(
        df_discharge, test_size, feature_names
    )
    device = torch.device("cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_x = torch.from_numpy(dev_x).type(torch.float32).to(device)
    dev_y = torch.from_numpy(dev_y).type(torch.float32).to(device)
    test_x = torch.from_numpy(test_x).type(torch.float32).to(device)
    test_y = torch.from_numpy(test_y).type(torch.float32).to(device)
    # Set hyperparameters
    epochs = 100
    input_shape = dev_x.shape[-1]
    d_model = 16
    nhead = 4
    num_layers = 2
    output_size = 1

    # Define model
    model = Transformer(input_shape, d_model, nhead, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train
    model = train(dev_x, dev_y, model, epochs, optimizer, criterion)

    # Evaluate
    test_loss = evaluate(model, test_x, test_y, criterion)
    pred = model(test_x).detach().numpy()[:, 0]
    test_y = test_y.detach().numpy()[:, 0]
    pred_inv = y_scaler.inverse_transform(pred.reshape(1, -1)).reshape(
        -1,
    )
    test_inv = y_scaler.inverse_transform(test_y.reshape(1, -1)).reshape(
        -1,
    )
    parity_plot(test_inv, pred_inv)

    print(f"test loss = {test_loss}")


if __name__ == "__main__":
    main()
