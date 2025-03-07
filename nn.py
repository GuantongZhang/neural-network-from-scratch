"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input_cache = None

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        self.input_cache = x
        return torch.relu(x)

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        # outdim === indim
        return grad_wrt_out * (self.input_cache > 0)


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 * torch.rand((outdim, indim), dtype=torch.float64, requires_grad=True, device=device)
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, requires_grad=True, device=device)
        self.lr = lr
    
    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.input_cache = x
        return self.weights @ x + self.bias
    
    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        batch_size = grad_wrt_out.shape[1]  # For averaging over batch
        #compute grad_wrt_weights
        self.grad_wrt_weights = (grad_wrt_out @ self.input_cache.T)  # (outdim, batch_size) @ (batch_size, indim)
        #compute grad_wrt_bias
        self.grad_wrt_bias = grad_wrt_out.sum(dim=1, keepdim=True)  # Averaging over batch
        #compute & return grad_wrt_input
        grad_wrt_input = self.weights.T @ grad_wrt_out  # (indim, outdim) @ (outdim, batch_size)
        return grad_wrt_input
    
    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights -= self.lr * self.grad_wrt_weights
        self.bias -= self.lr * self.grad_wrt_bias


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        batch_size = logits.shape[1]  # For averaging over batch
        exp_logits = torch.exp(logits - torch.max(logits, dim=0, keepdim=True)[0])  # For numerical stability
        self.probs = exp_logits / exp_logits.sum(dim=0, keepdim=True)  # (num_classes, batch_size)
        
        # Compute cross-entropy loss
        loss = -torch.sum(labels * torch.log(self.probs + 1e-9)) / batch_size  # Avoid log(0)
        
        # Cache labels for backward pass
        self.labels = labels
        
        return loss


    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        grad_wrt_logits = (self.probs - self.labels) / batch_size
        return grad_wrt_logits
    
    def getAccu(self):
        """
        return accuracy here
        """
        pred_classes = torch.argmax(self.probs, dim=0)  # Get predicted class indices
        true_classes = torch.argmax(self.labels, dim=0)  # Get true class indices
        accuracy = torch.mean((pred_classes == true_classes).float())  # Compute mean accuracy
        return accuracy


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        # Initiation
        self.linear1 = LinearMap(indim, hidden_layer, lr)
        self.relu = ReLU()
        self.linear2 = LinearMap(hidden_layer, outdim, lr)
        self.lr = lr


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        self.x_cache = x  # Cache for backprop
        self.q = self.linear1.forward(x)  # First linear layer: W1x + b1
        self.h = self.relu.forward(self.q)  # ReLU activation
        self.o = self.linear2.forward(self.h)  # Second linear layer: W2h + b2
        return self.o  # (outdim, batch_size)


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        grad_wrt_h = self.linear2.backward(grad_wrt_out)  # Backprop through W2
        grad_wrt_q = self.relu.backward(grad_wrt_h)  # Backprop through ReLU
        self.linear1.backward(grad_wrt_q)  # Backprop through W1

    
    def step(self):
        """update model parameters"""
        self.linear1.step()
        self.linear2.step()


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

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

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
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("./data/X_test.csv")
    Ytest = pd.read_csv("./data/y_test.csv")
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), columns=Xtest.columns).to_numpy()  ##
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Construct the model
    model = SingleLayerMLP(indim, outdim, hidden_layer=hidden_dim, lr=lr)
    loss_fn = SoftmaxCrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Training loop
    for epoch in range(epochs):
        model.zerograd()  # Reset gradients at the start of each epoch
        total_train_loss, total_train_correct = 0, 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.T.to(device), batch_y.to(device)  # Convert to correct shape (indim, batch_size)
            batch_y_onehot = torch.tensor(labels2onehot(batch_y.numpy()), dtype=torch.float64, device=device).T  # Convert to one-hot (num_classes, batch_size)

            # Forward pass
            logits = model.forward(batch_X)
            loss = loss_fn.forward(logits, batch_y_onehot)
            
            # Backward pass
            grad_wrt_out = loss_fn.backward()
            model.backward(grad_wrt_out)
            model.step()  # Update parameters
            
            # Track training loss
            total_train_loss += loss.item()
            total_train_correct += loss_fn.getAccu().item() * batch_X.shape[1]
            total_samples += batch_X.shape[1]

        # Average loss and accuracy for the epoch
        train_losses.append(total_train_loss / len(train_loader))
        train_accuracies.append(total_train_correct / total_samples)

        # Evaluation on test set
        total_test_loss, total_test_correct = 0, 0
        total_test_samples = 0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.T.to(device), batch_y.to(device)
                batch_y_onehot = torch.tensor(labels2onehot(batch_y.numpy()), dtype=torch.float64, device=device).T
                
                logits = model.forward(batch_X)
                loss = loss_fn.forward(logits, batch_y_onehot)

                total_test_loss += loss.item()
                total_test_correct += loss_fn.getAccu().item() * batch_X.shape[1]
                total_test_samples += batch_X.shape[1]

        test_losses.append(total_test_loss / len(test_loader))
        test_accuracies.append(total_test_correct / total_test_samples)

        # Print progress every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

    # Plot training and testing loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label="Train Accuracy")
    plt.plot(range(epochs), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.show()