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
from battery_degradation_prediction.window import windowing
#from preprocessing import get_clean_data
#from load_data import load_data
#from model import Net, Transformer
#from evaluate import evaluate
#from window import windowing


def train(dev_x, dev_x_labels, dev_y, model, epochs, optimizer, criterion):
    """TODO"""
    for epoch in range(epochs):
        optimizer.zero_grad()  # zero the gradient buffers
        outputs, x_outputs = model(dev_x)
        loss_y = criterion(outputs, dev_y)
        loss_x = criterion(x_outputs, dev_x_labels)
        loss = loss_y + loss_x
        loss.backward()
        optimizer.step()  # Does the update
        if epoch % 10 == 0:
            print(f"Epoch = {epoch}, loss = {loss_y:2.5f}")
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

    #device = torch.device("cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    dev_x = torch.from_numpy(dev_x).type(torch.float32).to(device)
    dev_x_labels = torch.from_numpy(dev_x_labels).type(torch.float32).to(device)
    dev_y = torch.from_numpy(dev_y).type(torch.float32).to(device)
    test_x = torch.from_numpy(test_x).type(torch.float32).to(device)
    test_y = torch.from_numpy(test_y).type(torch.float32).to(device)
    test_x_labels = torch.from_numpy(test_x_labels).type(torch.float32).to(device)

    # Set hyperparameters
    epochs = 201
    input_shape = dev_x.shape[1:]
    d_model = 8
    nhead = 2
    num_layers = 2
    output_size = 1
    dropout = 0.2

    # Define model
    model = Transformer(input_shape, d_model, nhead, num_layers, output_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train
    model = train(dev_x, dev_x_labels, dev_y, model, epochs, optimizer, criterion)
    # Evaluate
    print('evaluate')
    
    test_loss = evaluate(model, test_x, test_y, criterion)
    pred, x_outputs = model(test_x)
    #pred, x_outputs = model(dev_x)
    pred = pred.cpu().detach().numpy()[:, 0]
    test_y = test_y.cpu().detach().numpy()[:, 0]
    #test_y = dev_y.cpu().detach().numpy()[:, 0]
    #x_output_inv = X_scaler.inverse_transform(x_outputs.reshape(1, -1)).reshape(-1,)
    pred_inv = y_scaler.inverse_transform(pred.reshape(1, -1)).reshape(
        -1,
    )
    test_inv = y_scaler.inverse_transform(test_y.reshape(1, -1)).reshape(
        -1,
    )
    
    parity_plot(test_inv, pred_inv)
    print(f"test loss = {test_loss}")
    
    """
    df_discharge = get_clean_data(path, 1000000)
    df_feature = df_discharge[feature_names]
    #dev_x, dev_y, test_x, test_y, y_scaler = load_data(
    #    df_discharge, test_size, feature_names
    #)
    for idx, group in enumerate(df_feature.groupby("cycle")):
        data = []
        data_y = []
        cycle_data = group[1].iloc[:, 1:]
        if idx == 40:
            cycle_data_windows, capacities = windowing(cycle_data, 5, 1)
            for (test_x, test_y) in zip(cycle_data_windows, capacities):
                data.append(test_x[:,:-1])
                data_y.append(test_y)
            data = np.asarray(data)
            data = X_scaler.transform(np.reshape(data, (len(data), -1)))
            data = np.reshape(data, (-1, 5, 3))
            pred = model(torch.from_numpy(data).type(torch.float32).to(device))
            data = X_scaler.inverse_transform(np.reshape(data, (len(data), -1)))
            data = np.reshape(data, (-1, 5, 3))
            pred = pred.detach().numpy()[:, 0]
            pred = y_scaler.inverse_transform(pred.reshape(1, -1)).reshape(-1,)
            plt.figure(figsize=(6,6))
            plt.plot(data_y, data[:, 0, 0], '--k', label='True')
            plt.scatter(pred, data[:, 0, 0], s=6, facecolors="none", edgecolors="k", label="Prediction")
            plt.xlabel("cummulative capacity")
            plt.ylabel("voltage measure")
            plt.title(f"Test cycle {idx}")
            plt.legend()
            plt.show()
    """
if __name__ == "__main__":
    main()
