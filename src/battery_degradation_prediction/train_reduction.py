"""train module"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.load_data import load_unsupervised_data
from battery_degradation_prediction.model import TransformerReduction
from battery_degradation_prediction.plot_model import plot_train_val_loss



def get_batch(X, y, batch_size, batch_num):
    """TODO"""
    X, y = shuffle(X, y)
    features = X[batch_num*batch_size:(batch_num+1)*batch_size]
    targets = y[batch_num*batch_size:(batch_num+1)*batch_size]
    return features, targets

def train(train_x, train_y, val_x, val_y, model, epochs, batch_size, optimizer, criterion):
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
    """TODO"""
    model.eval()
    x_outputs, _ = model(val_x)
    val_loss = criterion(x_outputs, val_y)
    return val_loss


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
    df_discharge = get_clean_data(path, int(1e7))
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
    (dev_x, dev_y), (test_x, test_y), X_scaler = load_unsupervised_data(df_discharge, test_size, feature_names, randomize=False)
    print(f'dev_x = {dev_x.shape}, dev_y = {dev_y.shape}\ntest_x = {test_x.shape}, test_y = {test_y.shape}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    dev_x = torch.from_numpy(dev_x).type(torch.float32).to(device)
    dev_y = torch.from_numpy(dev_y).type(torch.float32).to(device)
    test_x = torch.from_numpy(test_x).type(torch.float32).to(device)
    test_y = torch.from_numpy(test_y).type(torch.float32).to(device)
    assert not (torch.isnan(dev_x).any() or torch.isnan(dev_y).any()), "Input X contains nan"

    # Set hyperparameters
    epochs = 0
    input_shape = dev_x.shape[1:]
    d_model = 8
    nhead = 2
    num_layers = 2
    dropout = 0.2
    latent_size = 10
    batch_size = 4
    num_folds = 5
    
    # Define model
    model = TransformerReduction(input_shape, d_model, nhead, num_layers, latent_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train
    histories = []
    models = []
    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(dev_x)
    cross_val = 0
    if cross_val:
        for fold, (train_index, val_index) in enumerate(kf.split(dev_x)):
            train_x, val_x = dev_x[train_index], dev_x[val_index]
            train_y, val_y = dev_y[train_index], dev_y[val_index]
            print(f"ʕ •ᴥ•ʔ -- Fold {fold} -- ʕ •ᴥ•ʔ")
            model, history = train(train_x, train_y, 
                                val_x, val_y,
                                model, epochs, batch_size, optimizer, criterion)
            models.append(model)
            histories.append(history)
        plot_train_val_loss(histories)

    model_path = "./unsupervised_model"
    model = TransformerReduction(input_shape, d_model, nhead, num_layers, latent_size, dropout).to(device)
    optimizer = optim.Adam(model.parameters())
    model, history = train(dev_x, dev_y, 
                            dev_x, dev_y,
                            model, epochs, batch_size, optimizer, criterion)
    #torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    #draw_voltage_capacity(model, dev_x, X_scaler, dev_y, num_features)
    # Evaluate

    test_loss = evaluate(model, test_x, test_y, criterion)
    print(f"test loss = {test_loss:2.5f}")
    #draw_voltage_capacity(model, test_x, X_scaler, test_y, num_features)

    
    # Inference
    future_cycle = 10
    x_pred = inference(model, future_cycle, dev_x[:1])
    #draw_voltage_capacity(model, x_pred, X_scaler, dev_y[future_cycle+1:future_cycle+2], num_features)
    
    _, ax = plt.subplots(figsize=(8,8))
    first_window = dev_x[:1]
    markers = ["^", "X", "<", ">", "o", '*', "v"]
    for idx, future_cycle in enumerate(range(5, 131, 20)):
        first_window = dev_x[future_cycle-5:future_cycle-4]
        future = 5
        x_pred = inference(model, future, first_window)
        if len(dev_y[future_cycle+1:future_cycle+2]):
            (capacity_overtime_pred, 
            voltage_overtime_pred, 
            capacity_overtime, 
            voltage_overtime) = get_voltage_capacity(model, x_pred, X_scaler, dev_y[future_cycle-future:future_cycle-future+1], num_features)
            
            ax.scatter(capacity_overtime_pred[0],
                        voltage_overtime_pred[0],
                        marker=markers[idx],
                        label=f'Pred (cycle={future_cycle})', s=25)#, facecolors="none", edgecolors="k")
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
                        label=f'Pred (cycle={future_cycle})', s=12)#, facecolors="none", edgecolors="k")
        #first_window = x_pred

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
