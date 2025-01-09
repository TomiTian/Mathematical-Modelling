from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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