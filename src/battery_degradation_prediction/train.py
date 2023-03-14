"""train module"""
import torch
from torch import nn
from torch import optim
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.load_data import load_supervised_data
from battery_degradation_prediction.model import Transformer
from battery_degradation_prediction.plot_model import (
    plot_train_val_loss,
    parity_plot,
)  # , plot_future_capacity, plot_future_capacities

# from preprocessing import get_clean_data
# from load_data import load_data
# from model import Net, Transformer
# from evaluate import evaluate
# from window import windowing


def get_batch(X, y, batch_size, batch_num):
    """TODO"""
    X, y = shuffle(X, y)
    features = X[batch_num * batch_size : (batch_num + 1) * batch_size]
    targets = y[batch_num * batch_size : (batch_num + 1) * batch_size]
    return features, targets


def train(
    train_x, train_y, val_x, val_y, model, epochs, batch_size, optimizer, criterion
):
    """TODO"""
    num_batches = len(train_x) // batch_size
    history = {}
    model.train()
    print("epochs = ", epochs)
    for epoch in range(1, epochs + 1):
        print(f" ===== Epoch: {epoch}/{epochs} =====")
        loss_sum = 0
        for batch in range(num_batches):
            data, targets = get_batch(train_x, train_y, batch_size, batch)
            optimizer.zero_grad()  # zero the gradient buffers
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            loss_sum += loss
            optimizer.step()  # Does the update
            if batch % 20 == 0:
                print(f"Batch {batch+1}/{num_batches} | loss = {loss:2.5f}")
        history.setdefault("train_loss", []).append(
            (loss_sum / num_batches).cpu().detach().numpy()
        )
        val_loss = val_eval(val_x, val_y, model, criterion)
        print(f"Epoch = {epoch} | val_loss = {val_loss:2.5f}")
        history.setdefault("val_loss", []).append(val_loss.cpu().detach().numpy())
    return model, history


def val_eval(val_x, val_y, model, criterion):
    """TODO"""
    model.eval()
    preds = model(val_x)
    val_loss = criterion(preds, val_y)
    return val_loss


def evaluate(
    model: nn.Module, eval_data: torch.Tensor, targets: torch.Tensor, criterion
) -> float:
    """TODO"""
    model.eval()  # turn on evaluation mode
    with torch.no_grad():
        output = model(eval_data)
        loss = criterion(output, targets).item()
    return loss


def main():
    """TODO"""
    path = "../../data/B0005.csv"
    num_data = int(1e7)
    df_discharge = get_clean_data(
        path, int(num_data)
    )  # 10: 11140, 50: 113930, 100:316545, 159: 556018
    feature_names = [
        "cycle",
        "voltage_measured",
        "current_measured",
        "temperatrue_measured",
        "capcity_during_discharge",
        "capacity",
    ]
    test_size = 0.2
    window_size = 5
    (dev_x, dev_y), (test_x, test_y), _, y_scaler = load_supervised_data(
        df_discharge, test_size, feature_names, window_size
    )
    print(f"dev_x = {dev_x.shape}, dev_y = {dev_y.shape}")
    print(f"test_x = {test_x.shape}, test_y = {test_y.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    dev_x = torch.from_numpy(dev_x).type(torch.float32).to(device)
    dev_y = torch.from_numpy(dev_y).type(torch.float32).to(device)
    test_x = torch.from_numpy(test_x).type(torch.float32).to(device)
    test_y = torch.from_numpy(test_y).type(torch.float32).to(device)
    # Set hyperparameters
    epochs = 25
    input_shape = dev_x.shape[1:]
    d_model = 8
    nhead = 2
    num_layers = 2
    output_size = 1
    dropout = 0.2
    batch_size = 64
    num_folds = 5
    # Define model
    model = Transformer(
        input_shape, d_model, nhead, num_layers, output_size, dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train (Cross-validation)
    histories = []
    models = []
    k_fold = KFold(n_splits=num_folds)
    k_fold.get_n_splits(dev_x)
    cross_val = 1
    if cross_val:
        for fold, (train_index, val_index) in enumerate(k_fold.split(dev_x)):
            train_x, val_x = dev_x[train_index], dev_x[val_index]
            train_y, val_y = dev_y[train_index], dev_y[val_index]
            print(f"ʕ •ᴥ•ʔ -- Fold {fold} -- ʕ •ᴥ•ʔ")
            model, history = train(
                train_x,
                train_y,
                val_x,
                val_y,
                model,
                epochs,
                batch_size,
                optimizer,
                criterion,
            )
            models.append(model)
            histories.append(history)

        plot_train_val_loss(histories)

    model = Transformer(
        input_shape, d_model, nhead, num_layers, output_size, dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    model, history = train(
        dev_x, dev_y, dev_x, dev_y, model, epochs, batch_size, optimizer, criterion
    )

    model_path = "./supervised_model"
    torch.save(model.state_dict(), model_path)
    # model.load_state_dict(torch.load(model_path))

    # Evaluate
    print("evaluate")
    model.eval()
    test_loss = evaluate(model, test_x, test_y, criterion)
    pred = model(test_x)
    pred = pred.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()
    pred_inv = y_scaler.inverse_transform(pred)
    test_inv = y_scaler.inverse_transform(test_y)

    parity_plot(test_inv, pred_inv)
    print(f"test loss = {test_loss}")

    """
    models = []
    for cycle in cycles:
        model_path = f"./supervised_model_{cycle}"
        model = Transformer(input_shape, d_model, nhead, num_layers, 
                            output_size, dropout).to(device)
        model.load_state_dict(torch.load(model_path))
        models.append(model)
    df_discharge = get_clean_data(path, int(1e7))
    df_feature = df_discharge[feature_names]
    future_cycle = 160
    #plot_future_capacity(df_feature, input_shape, model, 
                          X_scaler, y_scaler, future_cycle, window_size)
    plot_future_capacities(df_feature, input_shape, model, X_scaler, y_scaler, 
                           future_cycle, window_size)
    """


if __name__ == "__main__":
    main()
