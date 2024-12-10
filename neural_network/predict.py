import torch
from dataset import get_dataloaders
from model import FullyConnectedNN
import matplotlib.pyplot as plt
import math

def visualize_predictions(model, loader, device, num_samples=12, cols=4):
   
    model.eval()
    images_shown = 0
    rows = math.ceil(num_samples / cols)  #calculate rows based on num_samples and cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

    
    axes = axes.flatten()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        
        for i in range(len(images)):
            if images_shown < num_samples:
                ax = axes[images_shown]
                ax.imshow(images[i].cpu().numpy().reshape(28, 28), cmap='gray')
                ax.set_title(f"True: {labels[i].item()}\nPred: {preds[i].item()}")
                ax.axis('off')
                images_shown += 1
            if images_shown == num_samples:
                break
        if images_shown == num_samples:
            break

    
    for i in range(images_shown, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    train_csv_file = 'fashion-mnist_train.csv'
    test_csv_file = 'fashion-mnist_test.csv'
    batch_size = 64

    #device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    _, test_loader = get_dataloaders(train_csv_file, test_csv_file, batch_size)

    
    model = FullyConnectedNN()
    state_dict = torch.load('fashion_mnist_fc.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)

    visualize_predictions(model, test_loader, device)
