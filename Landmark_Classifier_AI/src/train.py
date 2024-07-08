import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one training epoch.
    """
    if torch.cuda.is_available():
        model.cuda()

    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()  # Clear the gradients of all optimized variables
        output = model(data)  # Forward pass
        loss_value = loss(output, target)  # Calculate the loss
        loss_value.backward()  # Backward pass
        optimizer.step()  # Perform a single optimization step

        # Update average training loss
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss_value.item() - train_loss))

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch.
    """
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    valid_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss_value = loss(output, target)

            # Update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss_value.item() - valid_loss))

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    """
    Optimize the model over a number of epochs.
    """
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs} starting...")

        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        print(f"Training loss after epoch {epoch}: {train_loss}")

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)
        print(f"Validation loss after epoch {epoch}: {valid_loss}")

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or ((valid_loss_min - valid_loss) / valid_loss_min > 0.01):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    """
    Monitor test loss and accuracy.
    """
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    # Set the model to evaluation mode
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # Move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the loss
            loss_value = loss(output, target)

            # Update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.item() - test_loss))

            # Convert logits to predicted class
            pred = output.argmax(dim=1, keepdim=True)

            # Compare predictions to true label
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    print(f'Test Loss: {test_loss:.6f}\n')
    print(f'\nTest Accuracy: {100. * correct / total:.2f}% ({correct}/{total})')

    return test_loss


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)
    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"

