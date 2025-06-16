from braindecode.datasets import MOABBDataset
from numpy import multiply
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.preprocessing import create_windows_from_events
import torch
from braindecode.models import TCN
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

# Load dataset
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

# Preprocessing
low_cut_hz = 4.0
high_cut_hz = 38.0
factor_new = 1e-3
init_block_size = 1000
factor = 1e6

preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    Preprocessor(lambda data: multiply(data, factor)),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    Preprocessor(
        exponential_moving_standardize,
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

preprocess(dataset, preprocessors, n_jobs=-1)

n_times = 1000

# Model setup
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
n_chans = dataset[0][0].shape[0]


model = TCN(
    n_in_chans=n_chans,
    n_outputs=n_classes,
    n_blocks=4,
    n_filters=25,
    kernel_size=8,
    drop_prob=0.2,#drop_prob=0.5,
    add_log_softmax=True,
    n_times=n_times
    
)

if cuda:
    model.cuda()

# Windowing
trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=n_times,
    window_stride_samples=n_times,
    drop_last_window=False,
    preload=True,
)

splitted = windows_dataset.split("session")
train_set = splitted["0train"]
valid_set = splitted["1test"]

# Classifier
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 60

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
)

_ = clf.fit(train_set, y=None, epochs=n_epochs)

# Plot Results
results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
df = pd.DataFrame(
    clf.history[:, results_columns],
    columns=results_columns,
    index=clf.history[:, "epoch"],
)
df["train_misclass"] = 100 - df["train_accuracy"] * 100
df["valid_misclass"] = 100 - df["valid_accuracy"] * 100

# Accuracy Plot
plt.figure(figsize=(8, 3))
plt.plot(df.index, df["train_accuracy"], label="Train Accuracy", marker='o')
plt.plot(df.index, df["valid_accuracy"], label="Valid Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss Plot
plt.figure(figsize=(8, 3))
plt.plot(df.index, df["train_loss"], label="Train Loss", marker='o')
plt.plot(df.index, df["valid_loss"], label="Valid Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
confusion_mat = confusion_matrix(y_true, y_pred)
label_dict = windows_dataset.datasets[0].window_kwargs[0][1]["mapping"]
labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
plot_confusion_matrix(confusion_mat, class_names=labels)
plt.show()
