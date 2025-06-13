from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

import numpy as np

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

low_cut_hz = 4.0    # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
                    # Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

transforms = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(
        lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
        factor=1e6,
    ),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, transforms, n_jobs=-1)

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

import torch

from braindecode.models import EEGConformer
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_chans = windows_dataset[0][0].shape[0]
n_times = windows_dataset[0][0].shape[1]

# The ShallowFBCSPNet is a `nn.Sequential` model

model = EEGConformer(
    n_outputs=n_classes,
    n_chans=n_chans,
    n_times=n_times,
    n_filters_time=256,       # embeding size (big = increase power, slower training)
    att_depth=4,              # attention block amount
    att_heads=8,              # multi-head attention amount
    drop_prob=0.1,            # dropout CNN
    att_drop_prob=0.1,        # dropout attention
    final_fc_length="auto"    # FC fitting
)

print(model)


if cuda:
    model.cuda()

splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
test_set = splitted["1test"]    # Session evaluation

from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

lr = 3e-4
weight_decay = 0
batch_size = 16
n_epochs = 30

from tqdm import tqdm

# Define a method for training one epoch

def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: LRScheduler,
    epoch: int,
    device,
    print_batch_stats=True,
):
    model.train()  # Set the model to training mode
    train_loss, correct = 0.0, 0.0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}/{n_epochs}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
            )

    # Update the learning rate
    scheduler.step()

    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model(dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()  # Switch to evaluation mode
    test_loss, correct = 0.0, 0.0

    if print_batch_stats:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        batch_loss = loss_fn(pred, y).item()

        test_loss += batch_loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {batch_loss:.6f}"
            )

    test_loss /= n_batches
    correct /= size

    print(f"Test Accuracy: {100 * correct:.1f}%, Test Loss: {test_loss:.6f}\n")
    return test_loss, correct


# Define the optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)

# Define the loss function
# We used the NNLoss function, which expects log probabilities as input
# (which is the case for our model output)
loss_fn = torch.nn.CrossEntropyLoss()

# train_set and test_set are instances of torch Datasets, and can seamlessly be
# wrapped in data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_accuracy = train_one_epoch(
        train_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        epoch,
        device,
    )

    test_loss, test_accuracy = test_model(test_loader, model, loss_fn)

    print(
        f"Train Accuracy: {100 * train_accuracy:.2f}%, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Test Accuracy: {100 * test_accuracy:.1f}%, "
        f"Average Test Loss: {test_loss:.6f}\n"
    )

import lightning as L
from torchmetrics.functional import accuracy


class LitModule(L.LightningModule):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y, "multiclass", num_classes=4)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs - 1
        )
        return [optimizer], [scheduler]

trainer = L.Trainer(max_epochs=n_epochs)

# Create and train the LightningModule
lit_model = LitModule(model)
trainer.fit(lit_model, train_loader)

trainer.test(dataloaders=test_loader)