import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pathlib import Path
import time
import pickle


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, 1)
        self.conv2 = nn.Conv2d(30, 60, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8640, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        c1 = F.relu(x)
        x = self.conv2(c1)
        c2 = F.relu(x)
        c3 = F.max_pool2d(c2, 2)
        x = self.dropout1(c3)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, c1, c2, c3


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    all_preds = []
    all_targets = []
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ ,__, ___ = model(data)
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
            output, _ ,__, ___ = model(data)
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
    accuracy = 100 * correct / total
    return accuracy

def forward_prop_test_set(download_folder:Path, path_results: Path, mode:str):
    """Returns the accuracy of the model (best or final) on the test set
    and the amount of time it took to get all predictions
    
    To make it completely fair, reloads the model, re-loads the data etc.
    """
    start_time = time.perf_counter()
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mode == "best":
        model.load_state_dict(torch.load(path_results / "cnn_weights_best.pt", map_location=device, weights_only=True))
    elif mode == "final":
        model.load_state_dict(torch.load(path_results / "cnn_weights_final.pt", map_location=device, weights_only=True))
    # model.load_state_dict(torch.load(path_results / "cnn_weights.pt", map_location=device, weights_only=True))
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(download_folder, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


    # Run the forward propagation for the loaded model
    model.eval()
    with torch.no_grad():
        test_preds, test_targets = [], []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output, _ ,__, ___ = model(data)
            preds = output.argmax(dim=1, keepdim=True).squeeze()
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(target.cpu().numpy())

    end_time = time.perf_counter()
    # Calculate accuracy
    accuracy = calculate_accuracy(np.array(test_preds), np.array(test_targets))
    t = end_time - start_time
    
    # Even more sanity checking
    pre, tar, loss = test(model,device, test_loader)
    acc2 = calculate_accuracy(pre, tar)

    if acc2 != accuracy:
        print("WARNING: Accuracy calculated with 2 different methods does not match up")

    return accuracy, t, model




def main(save_folder: Path, download_folder: Path, parser: argparse.ArgumentParser):
    start_time = time.perf_counter()
    args, unknown = parser.parse_known_args()
    
    # Setting training and testing specific parameters
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Transform the data so it is normalized. 
    # 0.1307 and 0.3081 are the training mean and std of the MNIST dataset
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

    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = None

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

        # Check if the current test loss is the best we've seen so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            best_epoch = epoch

    end_time = time.perf_counter()

    # Save the optimal weights
    if args.save_model and best_model_state is not None:
        torch.save(best_model_state, save_folder / "cnn_weights_best.pt")
    
    # Save the final weights
    if args.save_model:
        torch.save(model.state_dict(), save_folder / "cnn_weights_final.pt")

    # We calculate the final accuracies for both set of weights.
    final_accuracy, t_final = forward_prop_test_set(download_folder, save_folder, "final")
    best_accuracy, t_best = forward_prop_test_set(download_folder, save_folder, "best")

    # Compile the results
    results = { "losses_train": train_losses,
            "losses_test": test_losses,
            "epochs": args.epochs,
            "confusion": {"train_t": train_targs, "train_p": train_preds, 
                          "test_t": test_targs, "test_p": test_preds},
            "final_epoch_accuracy": final_accuracy,
            "best_epoch_accuracy": best_accuracy,
            "best_epoch":best_epoch,
            "prep_time": end_time - start_time,
            "predict_time": t_final,

    }
    
    # Save the results / summaries
    with open(save_folder / "results.pkl", 'wb') as f:
        pickle.dump(results, f)

    return results, model






def load_or_train_CNN(path_results: Path, download_folder:Path, parser, force:bool=False):
    """We load the model (and results) if weights are available, otherwise we train (and test) the model
    
    * force: bool Train the model regardless of the existing weights.
    * path_results: Path Folder to save/retrieve the model weights
    * path_img: Path Folder to save the images to/from
    * path_data: Path Folder to save/retrieve the raw data to/from

    * Returns: dict with all results of either the trained or loaded model
    * might also return some re-tested results.
    """
    # Check if weights exist
    weights_exist = (path_results / "cnn_weights_best.pt").exists() and (path_results / "cnn_weights_final.pt").exists()
    # Check if data exists
    results_exist = (path_results / "results.pkl").exists()

    # If either force, or weights/results do not exist, run the train/test loop
    if not weights_exist or not results_exist:
        print("Can not find Weights or results, will have to run the train/test loop")
        results, model = main(path_results, download_folder, parser=parser)
    elif force:
        print("Force is enabled, will run the train/test loop")
        results, model = main(path_results, download_folder, parser=parser)
    
    else:
        # Can load the weights and results from pickle etc.
        with open(path_results / "results.pkl", 'rb') as file:
            results = pickle.load(file)
            print("Loaded previous results from file")
            #! RETEST WITH LOADED WEIGHTS TO REPRODUCE ACCURACY

        best_test_accuracy, t_best, model = forward_prop_test_set(download_folder, path_results, "best")
        final_test_accuracy, t_final, model = forward_prop_test_set(download_folder, path_results, "final")
        
        print("Now let us check if accuracy is the same as the one we calculated from the weights")
        print("     Loaded vs calculated")
        print(f"Best accuracy {results['best_epoch_accuracy']} vs {best_test_accuracy}")
        print(f"Final accuracy {results['final_epoch_accuracy']} vs {final_test_accuracy}")
        print(f"It took {t_final} seconds to calculate the predictions")
        print(f"That is an average of {t_final/10000} seconds per image")

    return results, model





