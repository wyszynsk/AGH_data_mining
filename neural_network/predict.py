import torch
from dataset import get_dataloaders
from model import FullyConnectedNN
import matplotlib.pyplot as plt

def visualize_predictions(model, loader, device, num_samples=6):
    
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # perform predictions
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # convert images to numpy for visualization
    images = images.cpu().numpy()

    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')  
   
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    #convert images to numpy for visualization
    images = images.cpu().numpy()

    #plot the predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')  #28x28 image
        ax.set_title(f"True: {labels[i].item()}\nPred: {preds[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    train_csv_file = 'fashion-mnist_train.csv'  
    test_csv_file = 'fashion-mnist_test.csv'
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #loading data
    _, test_loader = get_dataloaders(train_csv_file, test_csv_file, batch_size)

    
    model = FullyConnectedNN()
    model.load_state_dict(torch.load('fashion_mnist_fc.pth', map_location=device, weights_only=True))

    model.to(device)

    #visualize predictions
    visualize_predictions(model, test_loader, device)
