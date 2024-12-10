import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import FullyConnectedNN

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images) #forward pass
        loss = criterion(outputs, labels)
        loss.backward() #backward pass, computing gradients
        optimizer.step() #adjusting parameters
        running_loss += loss.item()
    return running_loss / len(loader)

def test(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1) #gets the class with the highest probability
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return test_loss / len(loader), accuracy

if __name__ == "__main__":
    #parameters
    train_csv_file = 'fashion-mnist_train.csv'  
    test_csv_file = 'fashion-mnist_test.csv'    
    batch_size = 64
    learning_rate = 0.001
    epochs = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load data
    train_loader, test_loader = get_dataloaders(train_csv_file, test_csv_file, batch_size)

    #model, loss, optimizer
    model = FullyConnectedNN().to(device)
    criterion = nn.CrossEntropyLoss() #loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #updasing weights

    #training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    
    model_path = 'fashion_mnist_fc.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
