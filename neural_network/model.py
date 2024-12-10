import torch
import torch.nn as nn

class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # 28x28 images (784 pixels)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)   # 10 classes (0 to 9)

    def forward(self, x):
        x = x.view(-1, 784)  # flatten the images
        x = torch.relu(self.fc1(x))  # apply ReLU activation
        x = torch.relu(self.fc2(x))  # apply ReLU activation
        x = self.fc3(x)  # output layer
        return x
