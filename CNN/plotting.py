from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torchvision import datasets, transforms

import random




def plot_loss(results:dict , save_folder:Path, show:bool=False):
    train_losses = results["losses_train"]
    test_losses = results["losses_test"]

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xticks(epochs)
    plt.xticks(rotation=90)  # Rotate x-axis tick labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss (NLL)')
    plt.legend()
    plt.title('Train and Test Loss')

    plt.savefig(save_folder/"loss_plot.png")
    if show:
        plt.show()
    plt.close()

def plot_confusion_matrix(epoch:int, results:dict, save_folder:Path, show:bool=False):
    "Plots the confusion matrices for both test and train for a given epoch"
    # Some boilerplate
    classes = [str(i) for i in range(10)]

    # Show a confusion matrix for the train set
    train_targ = results["confusion"]["train_t"][epoch-1]
    train_pred = results["confusion"]["train_p"][epoch-1]
    cm = confusion_matrix(train_targ, train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Train Confusion Matrix Epoch {epoch}")
    plt.savefig(save_folder/f"train_confusion_matrix_epoch_{epoch}.png")
    if show:
        plt.show()
    plt.close()

    # Do the same for the test set
    test_targ = results["confusion"]["test_t"][epoch-1]
    test_pred = results["confusion"]["test_p"][epoch-1]

    cm = confusion_matrix(test_targ, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix Epoch {epoch}")
    plt.savefig(save_folder/f"test_confusion_matrix_epoch_{epoch}.png")
    if show:
        plt.show()

    plt.close()


def visualize_activations(model, input_image):
    # Set the model to evaluation mode
    model.eval()

    # Make the image a batch of size 1 by adding a batch dimension
    input_image = input_image.unsqueeze(0)  # Shape: (1, 28, 28)

    # Pass the image through the model
    with torch.no_grad():  # We don't need gradients for this operation
        _, conv1_activations, conv2_activations, maxpools = model(input_image)
    
    print(len(conv1_activations[0]))
    print(len(conv2_activations[0]))
    print(len(maxpools[0]))

    # Visualizing the activations
    fig, ax = plt.subplots(2, 15, figsize=(20, 7))

    # Plot the activations of the first convolutional layer
    for i in range(conv1_activations.size(1)):  # Iterate over the 30 filters
        row = 0 if i < 15 else 1
        if row == 1:
            col = i - 15
        else:
            col = i
        
        ax[row, col].imshow(conv1_activations[0, i].cpu().detach().numpy(), cmap='viridis')
        ax[row, col].axis('off')
        ax[row, col].set_title(f'Filter {i+1}')

    fig.suptitle("Activation of the first layer")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(4, 15, figsize=(20, 7))
    # Plot the activations of the second convolutional layer
    for i in range(conv2_activations.size(1)):  # Iterate over the 60 filters
        row = i // 15
        col = i % 15
        # print(row, col)
        ax2[row, col].imshow(conv2_activations[0, i].cpu().detach().numpy(), cmap='viridis')
        ax2[row, col].axis('off')
        ax2[row, col].set_title(f'Filter {i+1}')
    
    fig2.suptitle("Activation of the second layer")
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(4, 15, figsize=(20, 7))
    # Plot the activations of the second convolutional layer
    for i in range(maxpools.size(1)):  # Iterate over the 60 filters
        row = i // 15
        col = i % 15
        # print(row, col)
        ax3[row, col].imshow(maxpools[0, i].cpu().detach().numpy(), cmap='viridis')
        ax3[row, col].axis('off')
        ax3[row, col].set_title(f'Filter {i+1}')
    
    fig3.suptitle("Maxpools")
    plt.tight_layout()
    plt.show()
    return None

def get_single_image(download_folder: Path, img_index:int, force_random=False):
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Create data loaders
    dataset1 = datasets.MNIST(download_folder, train=True, transform=transform)
    dataset2 = datasets.MNIST(download_folder, train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=False)
    # Get a batch of images and labels from the train_loader
    images, labels = next(iter(train_loader))

    # Select a single image from the batch
    picked_image = images[0]  # Shape: (1, 28, 28)

    if force_random:
        picked_image = images[random.randint(0, len(images)-1)]

    return picked_image



    
    
    
    pass