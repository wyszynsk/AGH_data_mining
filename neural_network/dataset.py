import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

class CSVDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_dataloaders(train_csv_file, test_csv_file, batch_size=64):
   
    train_df = pd.read_csv(train_csv_file)
    test_df = pd.read_csv(test_csv_file)

    
    y_train = train_df.iloc[:, 0].values   # first column as labels
    X_train = train_df.iloc[:, 1:].values  # all columns except the first one as features

    y_test = test_df.iloc[:, 0].values    
    X_test = test_df.iloc[:, 1:].values   

    # normalization 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # creating datasets
    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)

    # creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
