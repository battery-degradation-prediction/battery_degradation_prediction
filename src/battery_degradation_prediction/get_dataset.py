from torch.utils.data import DataLoader
from battery_degradation_prediction.preprocessing import get_clean_data
from battery_degradation_prediction.load_data import load_data

class BateryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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
    test_size = 0.2
    dev_x, dev_y, test_x, test_y, X_scaler, y_scaler = load_data(
        df_discharge, test_size, feature_names
    )
    train_dataset = BateryDataset()
    train_dataloader = DataLoader((dev_x, dev_y), batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")


if __name__ == "__main__":
    main()

