from braindecode.datasets import MOABBDataset
import numpy as np
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.preprocessing import create_windows_from_events
import torch
from braindecode.models import EEGConformer
from braindecode.util import set_random_seeds
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 


subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

low_cut_hz = 4.0
high_cut_hz = 38.0
factor_new = 1e-3
init_block_size = 1000

transforms = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    Preprocessor(lambda data, factor: np.multiply(data, factor), factor=1e6),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    Preprocessor(
        exponential_moving_standardize,
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

preprocess(dataset, transforms, n_jobs=-1)

trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
n_chans = windows_dataset[0][0].shape[0]
n_times = windows_dataset[0][0].shape[1]

model = EEGConformer(
    n_outputs=n_classes,
    n_chans=n_chans,
    n_times=n_times,
    n_filters_time=256,
    att_depth=4,
    att_heads=8,
    drop_prob=0.1,
    att_drop_prob=0.1,
    final_fc_length="auto"
)

if cuda:
    model.cuda()

splitted = windows_dataset.split("session")
train_set = splitted["0train"]
test_set = splitted["1test"]

lr = 3e-4
weight_decay = 0
batch_size = 16
n_epochs = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
loss_fn = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

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
    model.train()
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
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}/{n_epochs}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
            )

    scheduler.step()
    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model(dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    progress_bar = tqdm(enumerate(dataloader), total=n_batches) if print_batch_stats else enumerate(dataloader)

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

history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
}

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

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_accuracy)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_accuracy)

    print(
        f"Train Accuracy: {100 * train_accuracy:.2f}%, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Test Accuracy: {100 * test_accuracy:.1f}%, "
        f"Average Test Loss: {test_loss:.6f}\n"
    )

# Accuracy Plot
epochs_range = range(1, n_epochs + 1)
plt.figure()
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, history["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Loss Plot
plt.figure()
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

# Confuse Matrix
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for X, y, _ in test_loader:
        X = X.to(device)
        y = y.to(device)
        preds = model(X).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

confusion_mat = confusion_matrix(all_labels, all_preds)

label_dict = windows_dataset.datasets[0].window_kwargs[0][1]["mapping"]
labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]

plot_confusion_matrix(confusion_mat, class_names=labels)
plt.show()
