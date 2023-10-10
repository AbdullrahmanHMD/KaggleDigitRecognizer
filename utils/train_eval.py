# PyTorch imports:
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import optim

# Other imports:
from typing import Union
from tqdm import tqdm


def train_an_epoch(model : Module, data_loader : DataLoader, criterion : Module, optimizer : optim, device : torch.device):
    model.train()
    model = model.to(device=device)
    total_loss = 0
    for x, y in data_loader:
        x = x.to(device=device)
        y = y.type(torch.FloatTensor)
        y = y.to(device=device)

        optimizer.zero_grad()

        y_pred = model(x)
        _, y_pred = torch.max(y_pred, axis=1)

        loss = criterion(y_pred, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss


def evaluate(model : Module, data_loader : DataLoader, criterion : Module, device : torch.device):
    model.eval()
    model = model.to(device=device)

    total_loss, num_correct = 0, 0
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(device=device)
            y = y.type(torch.FloatTensor)
            y = y.to(device=device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            total_loss += loss.item()

            _, y_pred = torch.max(y_pred, axis=1)
            num_correct += (y_pred == y).sum().item()

    num_batches = data_loader.batch_size
    total_data_points = (num_batches - 1) * data_loader.batch_size + len(data_loader.dataset) % data_loader.batch_size
    accuracy = num_correct / total_data_points
    return {"loss" : total_loss, "accuracy" : accuracy}


def train(model : Module,
          train_loader : DataLoader,
          val_loader : DataLoader,
          criterion : Module,
          optimizer : optim,
          lr_scheduler : optim.lr_scheduler,
          epochs,
          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          verbose=True):

    total_val_loss, total_train_loss = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(epochs):
        train_an_epoch(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)

        train_evaluation_dict = evaluate(model=model, data_loader=train_loader, criterion=criterion, device=device)
        val_evaluation_dict = evaluate(model=model, data_loader=val_loader, criterion=criterion, device=device)
        train_loss, train_accuray = train_evaluation_dict["loss"], train_evaluation_dict["accuracy"]
        val_loss, val_accuracy = val_evaluation_dict["loss"], val_evaluation_dict["accuracy"]

        total_train_loss.append(train_loss)
        train_accuracies.append(train_accuray)

        total_val_loss.append(val_loss)
        val_accuracies.append(val_accuracy)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            status = f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train_acc: {100 * train_accuray:.2f}% | " \
f"Val_acc: {100 * val_accuracy:.2f}% | LR: {current_lr}"
            print(status)

    return total_train_loss, total_val_loss, train_accuracies, val_accuracies
