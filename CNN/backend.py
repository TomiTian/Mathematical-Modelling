import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pathlib import Path

import pickle


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    all_preds = []
    all_targets = []
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Collect predictions for confusion matrix
        preds = output.argmax(dim=1, keepdim=True).squeeze()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    train_loss /= len(train_loader.dataset)
    return np.array(all_preds), np.array(all_targets), train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            preds = output.argmax(dim=1, keepdim=True).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            correct += preds.eq(target.view_as(preds)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return np.array(all_preds), np.array(all_targets), test_loss


def calculate_accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    total = len(targets)
    accuracy = 100. * correct / total
    return accuracy

def main(save_folder: Path, download_folder: Path):
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    
    # Create data loaders
    dataset1 = datasets.MNIST(download_folder, train=True, transform=transform)
    dataset2 = datasets.MNIST(download_folder, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []
    test_losses = []
    
    train_targs, train_preds = [], []
    test_targs, test_preds = [], []

    for epoch in range(1, args.epochs + 1):
        tr_p, tr_t, train_loss = train(args, model, device, train_loader, optimizer, epoch)
        te_p, te_t, test_loss = test(model, device, test_loader)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_targs.append(tr_t)
        train_preds.append(tr_p)

        test_targs.append(te_t)
        test_preds.append(te_p)

    # Calculate final accuracies
    final_train_accuracy = calculate_accuracy(train_preds[-1], train_targs[-1])
    final_test_accuracy = calculate_accuracy(test_preds[-1], test_targs[-1])

    # Compile the results
    results = { "losses_train": train_losses,
            "losses_test": test_losses,
            "epochs": args.epochs,
            "confusion": {"train_t": train_targs, "train_p": train_preds, 
                          "test_t": test_targs, "test_p": test_preds},
            "final_train_accuracy": final_train_accuracy,
            "final_test_accuracy": final_test_accuracy
    }
    
    # Save the weights
    if args.save_model:
        torch.save(model.state_dict(), save_folder / "cnn_weights.pt")
    
    # Save the results / summaries
    with open(save_folder / "results.pkl", 'wb') as f:
        pickle.dump(results, f)

    return results

if __name__ == '__main__':
    main()





def load_or_train_CNN(path_results: Path, download_folder:Path, force:bool=False):
    """We load the model (and results) if weights are available, otherwise we train (and test) the model
    
    * force: bool Train the model regardless of the existing weights.
    * path_results: Path Folder to save/retrieve the model weights
    * path_img: Path Folder to save the images to/from
    * path_data: Path Folder to save/retrieve the raw data to/from

    * Returns: dict with all results of either the trained or loaded model
    * Optionally returns dict with results of loaded model (re-tested)
    """
    
    # Check if weights exist
    weights_exist = (path_results / "cnn_weights.pt").exists()
    # Check if data exists
    results_exist = (path_results / "results.pkl").exists()

    # If either force, or weights/results do not exist, run the train/test loop

    if not weights_exist or not results_exist:
        print("Can not find Weights or results, will have to run the train/test loop")
        results = main(path_results, download_folder)
    elif force:
        print("Force is enabled, will run the train/test loop")
        results = main(path_results, download_folder)
    else:
        # Can load the weights and results from pickle etc.
        with open(path_results / "results.pkl", 'rb') as file:
            results = pickle.load(file)
            print("Loaded previous results from file")
            #! RETEST WITH LOADED WEIGHTS TO REPRODUCE ACCURACY

        model = Net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(path_results / "cnn_weights.pt", map_location=device))
        model.to(device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(download_folder, train=True, transform=transform)
        test_dataset = datasets.MNIST(download_folder, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # Run the forward propagation for the loaded model
        model.eval()
        with torch.no_grad():
            train_preds, train_targets = [], []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1, keepdim=True).squeeze()
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(target.cpu().numpy())

            test_preds, test_targets = [], []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1, keepdim=True).squeeze()
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(target.cpu().numpy())

        # Calculate accuracies
        calculated_train_accuracy = calculate_accuracy(np.array(train_preds), np.array(train_targets))
        calculated_test_accuracy = calculate_accuracy(np.array(test_preds), np.array(test_targets))

        # results['loaded_train_accuracy'] = loaded_train_accuracy
        # results['loaded_test_accuracy'] = loaded_test_accuracy
        print("Now let us check if accuracy is the same as the one we calculated from the weights")
        print("Loaded vs calculated")
        print(f"Train accuracy {results["final_train_accuracy"]} vs {calculated_train_accuracy}")
        print(f"Test accuracy {results["final_test"]} vs {calculated_test_accuracy}")

    return results