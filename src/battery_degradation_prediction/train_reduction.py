"""train module"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
#from battery_degradation_prediction.preprocessing import get_clean_data
#from battery_degradation_prediction.load_data import load_data, load_data_reduction
#from battery_degradation_prediction.model import Net, Transformer, TransformerReduction
#from battery_degradation_prediction.evaluate import evaluate
#from battery_degradation_prediction.window import windowing
#from battery_degradation_prediction.lstm_vae import LSTMVAE
from preprocessing import get_clean_data
from load_data import load_data, load_data_reduction
from model import Net, Transformer
from evaluate import evaluate
from window import windowing
from lstm_vae import LSTMVAE


def train(dev_x_labels, model, epochs, optimizer, criterion):
    """TODO"""
    for epoch in range(epochs):
        #for dev_x_label in dev_x_labels:
        optimizer.zero_grad()  # zero the gradient buffers
        #x_outputs, _ = model(dev_x_labels)
        #loss_x = criterion(x_outputs, dev_x_labels)
        #loss = loss_x
        loss, _, (_, _) = model(dev_x_labels)
        loss.backward()
        optimizer.step()  # Does the update
        if epoch % 10 == 0:
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


def evaluate(
    model: nn.Module, eval_data: torch.Tensor, targets: torch.Tensor, criterion
) -> float:
    model.eval()  # turn on evaluation mode
    with torch.no_grad():
        output, _, (_, _) = model(eval_data)
        #total_loss = criterion(output, targets).item()
    return output



def main():
    """TODO"""
    path = "~/B0005.csv"
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
    dev_x_labels, test_x_labels, X_scaler = load_data_reduction(df_discharge, test_size, feature_names)

    #device = torch.device("cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    dev_x_labels = torch.from_numpy(dev_x_labels).type(torch.float32).to(device)
    test_x_labels = torch.from_numpy(test_x_labels).type(torch.float32).to(device)

    # Set hyperparameters
    epochs = 251
    input_shape = dev_x_labels.shape[1:]
    print("input shape = ", input_shape)
    d_model = 8
    nhead = 2
    num_layers = 2
    output_size = 1
    dropout = 0.2
    latent_size = 10

    input_size = dev_x_labels.shape[2]
    hidden_size = 64
    num_lstm_layer = 1

    # Define model
    #model = TransformerReduction(input_shape, d_model, nhead, num_layers, output_size, latent_size, dropout).to(device)
    model = LSTMVAE(input_size, hidden_size, latent_size, device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # Train
    model = train(dev_x_labels, model, epochs, optimizer, criterion)
    # Evaluate    
    test_loss = evaluate(model, test_x_labels, test_x_labels, criterion)
    print(f"test loss = {test_loss}")


if __name__ == "__main__":
    main()
