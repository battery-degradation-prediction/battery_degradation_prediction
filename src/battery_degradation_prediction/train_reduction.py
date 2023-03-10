"""train module"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.load_data import load_data, load_data_reduction
from battery_degradation_prediction.model import Net, Transformer, TransformerReduction
#from battery_degradation_prediction.evaluate import evaluate
from battery_degradation_prediction.window import windowing
from battery_degradation_prediction.lstm_vae import LSTMVAE

from sklearn.utils import shuffle
from sklearn.model_selection import KFold

#from preprocessing import get_clean_data
#from load_data import load_data, load_data_reduction
#from model import Net, Transformer
#from evaluate import evaluate
#from window import windowing
#from lstm_vae import LSTMVAE


def get_batch(X, y, batch_size, batch_num):
    """TODO"""
    X, y = shuffle(X, y)
    features = X[batch_num*batch_size:(batch_num+1)*batch_size]
    targets = y[batch_num*batch_size:(batch_num+1)*batch_size]
    return features, targets

def train(train_x, train_y, val_x, val_y, model, batch_size, epochs, optimizer, criterion):
    """TODO"""
    num_batches = len(train_x) // batch_size
    history = {}
    model.train()
    for epoch in range(1, epochs+1):
        print(f" ===== Epoch: {epoch}/{epochs} =====")
        loss_sum = 0
        for batch in range(num_batches):
            data, targets = get_batch(train_x, train_y, batch_size, batch)
            optimizer.zero_grad()  # zero the gradient buffers
            x_outputs, _ = model(data)
            loss = criterion(x_outputs, targets)
            loss_sum += loss
            loss.backward()
            optimizer.step()  # Does the update
            if batch % 20 == 0:
                print(f"Batch {batch+1}/{num_batches} | loss = {loss:2.5f}")
        history.setdefault('train_loss', []).append((loss_sum/num_batches).cpu().detach().numpy())
        val_loss = val_eval(val_x, val_y, model, criterion)
        print(f'Epoch = {epoch} | val_loss = {val_loss:2.5f}')
        history.setdefault('val_loss', []).append(val_loss.cpu().detach().numpy())
    return model, history

def val_eval(val_x, val_y, model, criterion):
    model.eval()
    x_outputs, _ = model(val_x)
    val_loss = criterion(x_outputs, val_y)
    return val_loss


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
        output, _ = model(eval_data)
        loss = criterion(output, targets).item()
    return loss



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
    test_size = 0.2
    num_features = 4
    #dev_x_labels, test_x_labels, X_scaler = load_data_reduction(df_discharge, test_size, feature_names)
    (dev_x, dev_x_labels, _), (test_x, _, _), X_scaler, x_label_scaler, y_scaler = load_data(df_discharge, test_size, feature_names)
    print(dev_x.shape, test_x.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    dev_x = torch.from_numpy(dev_x).type(torch.float32).to(device)
    test_x = torch.from_numpy(test_x).type(torch.float32).to(device)

    dev_y = dev_x[:, 1:]
    dev_x = dev_x[:, :-1]
    assert not (torch.isnan(dev_x).any() or torch.isnan(dev_y).any()), "Input X contains nan"
    test_y = test_x[:, 1:]
    test_x = test_x[:, :-1]

    # Set hyperparameters
    epochs = 200
    input_shape = dev_x.shape[1:]
    d_model = 8
    nhead = 2
    num_layers = 2
    output_size = 1
    dropout = 0.2
    latent_size = 10
    batch_size = 4

    # Define model
    model = TransformerReduction(input_shape, d_model, nhead, num_layers, output_size, latent_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train
    histories = []
    models = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(dev_x)

    for fold, (train_index, val_index) in enumerate(kf.split(dev_x)):
        train_x, val_x = dev_x[train_index], dev_x[val_index]
        train_y, val_y = dev_y[train_index], dev_y[val_index]
        print(f"ʕ •ᴥ•ʔ -- Fold {fold} -- ʕ •ᴥ•ʔ")
        model, history = train(train_x, train_y, 
                               val_x, val_y,
                               model, batch_size, epochs, optimizer, criterion)
        models.append(model)
        histories.append(history)
    train_val_loss = 1
    if train_val_loss:
        mean_train_loss = np.mean([history['train_loss'] for history in histories], axis=0)
        mean_val_loss = np.mean([history['val_loss'] for history in histories], axis=0)
        std_train_loss = np.std([history['train_loss'] for history in histories], axis=0)
        std_val_loss = np.std([history['val_loss'] for history in histories], axis=0)
        _, ax = plt.subplots(figsize=(5,5))
        #ax.errorbar(range(epochs), mean_train_loss, yerr=std_train_loss, label='train loss', markersize=6, ls='-')
        #ax.errorbar(range(epochs), mean_val_loss, yerr=std_val_loss, label='val loss', markersize=6, ls='--')
        ax.plot(range(epochs), mean_train_loss, '-o', label='train loss', markersize=4)
        ax.plot(range(epochs), mean_val_loss, '-o', label='val loss', markersize=4)
        plt.title("Loss versus epochs")
        plt.xlabel('Epochs [-]')
        plt.ylabel('Loss [-]')
        plt.legend()
        plt.show()
        #print(f"mean_train_loss = {mean_train_loss[0]:2.5f} ± {std_train_loss[0]:2.3f}")
        #print(f"mean_val_loss = {mean_val_loss[0]:2.5f} ± {std_val_loss[0]:2.3f}")
        
    model.eval()
    draw_voltage_capacity(model, dev_x, X_scaler, dev_y, num_features)
    # Evaluate
    # TODO: remember to train the model with the entire dataset
    test_loss = evaluate(model, test_x, test_y, criterion)
    print(f"test loss = {test_loss:2.5f}")
    draw_voltage_capacity(model, test_x, X_scaler, test_y, num_features)

    # Inference
    future_cycle = 10
    x_pred = inference(model, future_cycle, dev_x[:1])
    draw_voltage_capacity(model, x_pred, X_scaler, dev_y[future_cycle+1:future_cycle+2], num_features)

    _, ax = plt.subplots(figsize=(8,8))
    first_window = dev_x[:1]
    for future_cycle in range(10, 151, 20):
        x_pred = inference(model, future_cycle, first_window)
        if len(dev_y[future_cycle+1:future_cycle+2]):
            (capacity_overtime_pred, 
            voltage_overtime_pred, 
            capacity_overtime, 
            voltage_overtime) = get_voltage_capacity(model, x_pred, X_scaler, dev_y[future_cycle+1:future_cycle+2], num_features)
            
            ax.scatter(capacity_overtime_pred[0],
                        voltage_overtime_pred[0],
                        label=f'Pred (cycle={future_cycle})', s=6)#, facecolors="none", edgecolors="k")
            ax.plot(capacity_overtime[0],
                    voltage_overtime[0],
                    '--k')
                    #label='GT')
        else:
            (capacity_overtime_pred, 
            voltage_overtime_pred, 
            _, 
            _) = get_voltage_capacity(model, x_pred, X_scaler, dev_y[:1], num_features)
            ax.scatter(capacity_overtime_pred[0],
                        voltage_overtime_pred[0],
                        label=f'Pred (cycle={future_cycle})', s=6)#, facecolors="none", edgecolors="k")
        first_window = x_pred
    plt.xlabel("capacity")
    plt.ylabel("voltage")
    plt.legend()
    plt.show()
        
def inference(model, future_cycle, first_window):
    """TODO"""
    x_out = first_window
    for _ in range(future_cycle):
        x_out, _ = model(x_out)
    return x_out

def get_voltage_capacity(model, X, X_scaler, y, num_features):
    """TODO"""
    init_shape = y.shape
    x_outputs, _ = model(X)
    x_outputs = x_outputs.cpu().detach().numpy()
    x_outputs = X_scaler.inverse_transform(x_outputs.reshape(-1, num_features))
    x_outputs = np.reshape(x_outputs, (init_shape[0], init_shape[1], -1, num_features))
    capacity_overtime_pred = x_outputs[:, -1, :, -1]
    voltage_overtime_pred = x_outputs[:, -1, :, 0]

    dev_inv = y.cpu().detach().numpy()
    dev_inv = X_scaler.inverse_transform(dev_inv.reshape(-1, num_features))
    dev_y_features = np.reshape(dev_inv, (init_shape[0], init_shape[1], -1, num_features))
    capacity_overtime = dev_y_features[:, -1, :, -1]
    voltage_overtime = dev_y_features[:, -1, :, 0]
    return capacity_overtime_pred, voltage_overtime_pred, capacity_overtime, voltage_overtime

def draw_voltage_capacity(model, X, X_scaler, y, num_features):
    """TODO"""
    init_shape = y.shape
    x_outputs, _ = model(X)
    x_outputs = x_outputs.cpu().detach().numpy()
    x_outputs = X_scaler.inverse_transform(x_outputs.reshape(-1, num_features))
    x_outputs = np.reshape(x_outputs, (init_shape[0], init_shape[1], -1, num_features))
    capacity_overtime_pred = x_outputs[:, -1, :, -1]
    voltage_overtime_pred = x_outputs[:, -1, :, 0]

    dev_inv = y.cpu().detach().numpy()
    dev_inv = X_scaler.inverse_transform(dev_inv.reshape(-1, num_features))
    dev_y_features = np.reshape(dev_inv, (init_shape[0], init_shape[1], -1, num_features))
    capacity_overtime = dev_y_features[:, -1, :, -1]
    voltage_overtime = dev_y_features[:, -1, :, 0]

    _, ax = plt.subplots(figsize=(5,5))
    for cycle_num in range(1):
        ax.scatter(capacity_overtime_pred[cycle_num],
                   voltage_overtime_pred[cycle_num],
                   label='Pred', s=6, facecolors="none", edgecolors="k")
        ax.plot(capacity_overtime[cycle_num],
                voltage_overtime[cycle_num],
                '--k',
                label='GT')
    plt.xlabel("capacity")
    plt.ylabel("voltage")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
