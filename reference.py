"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this code.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(indim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, outdim),
        )


    def forward(self, x):
        """
        x shape (batch_size, indim)
        """
        return self.model(x)


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length


def validate(loader):
    """Takes in a dataloader, then returns the model loss and accuracy on this loader"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()  # Assuming a classification task
    
    with torch.no_grad():  # No need to track gradients during validation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)  # Get class predictions
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples  # Fraction of correct predictions

    return avg_loss, accuracy

if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """
    indim = 60
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 500

    #dataset
    Xtrain = pd.read_csv("./data/X_train.csv")
    Ytrain = pd.read_csv("./data/y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns).to_numpy().astype(np.float32)
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("./data/X_test.csv")
    Ytest = pd.read_csv("./data/y_test.csv").to_numpy()
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), columns=Xtest.columns).to_numpy().astype(np.float32)
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Construct model
    model = SingleLayerMLP(indim, outdim, hidden_layer=hidden_dim).to(device)
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    # Training loop
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        avg_train_loss = running_loss / total
        train_accuracy = correct / total

        # Validate on test set
        test_loss, test_accuracy = validate(test_loader)

        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Print progress every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

    # Plot loss & accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.show()