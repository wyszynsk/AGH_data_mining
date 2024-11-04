import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#FF9999', '#00FFFF'])



def acctual_vs_pred(X_test, y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for actual classes with custom colormap
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=custom_cmap, s=100)
    ax1.set_title('Actual Classes')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')

    # Plot for predicted classes with custom colormap
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=custom_cmap, s=100)
    ax2.set_title('Predicted Classes')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')

    # Show the plots
    plt.tight_layout()
    plt.show()


def plot(X, y):
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='salmon', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='cyan', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Classification Dataset')
    plt.legend()
    plt.show()