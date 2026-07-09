"""Training utilities for bone age regression models, including validation and early stopping."""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import copy
from tqdm import tqdm


class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = np.inf
        self.counter = 0
        self.best_model = None

    def step(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())

            return True  # improvement

        else:
            self.counter += 1

            return False

    def should_stop(self):
        return self.counter >= self.patience


def mae_months(pred, target, scaler):

    # inverse transform 
    pred = pred * scaler.scale_[0] + scaler.mean_[0]
    target = target * scaler.scale_[0] + scaler.mean_[0]

    return torch.mean(torch.abs(pred - target)).item()


def train_one_epoch(model, loader, optimizer, criterion, device, use_male=False):

    model.train()
    total_loss = 0.0

    for batch in tqdm(loader):

        image = batch["image"].to(device)
        y = batch["boneage"].float().to(device)

        if use_male:
            male = batch["male"].float().unsqueeze(1).to(device)
        else:
            male = None

        optimizer.zero_grad()

        pred = model(image, male=male)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device, scaler, use_male = False):

    model.eval()

    total_loss = 0
    total_mae = 0

    with torch.no_grad():

        for batch in loader:

            image = batch["image"].to(device)
            y = batch["boneage"].to(device)
            if use_male:
                male = batch["male"].float().unsqueeze(1).to(device)
            else:
                male = None

            pred = model(image, male=male)

            loss = criterion(pred, y)

            total_loss += loss.item()

            total_mae += mae_months(pred, y, scaler)

    return total_loss / len(loader), total_mae / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    scaler,
    device,
    epochs=50,
    patience=10,
    lr=1e-4,
    wd=1e-4,
    save_path="torch_model.pt",
    use_male=False
):

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss()

    early_stopper = EarlyStopping(patience=patience, min_delta=1e-4)

    best_val_loss = np.inf

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_male=use_male
        )

        val_loss, val_mae = validate(
            model, val_loader, criterion, device, scaler, use_male=use_male
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE (months): {val_mae:.2f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(model.state_dict(), save_path)

            print(f"+++ Saved best model (loss {val_loss:.4f})")


        early_stopper.step(val_loss, model)

        if early_stopper.should_stop():

            print("Early stopping triggered")

            break

    # restore best model
    model.load_state_dict(early_stopper.best_model)

    return model