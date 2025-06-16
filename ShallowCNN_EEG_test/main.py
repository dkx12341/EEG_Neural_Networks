
from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

from numpy import multiply

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

low_cut_hz = 4.0  # low cut 
high_cut_hz = 38.0  # high cut
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

factor = 1e6

preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,
        
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

preprocess(dataset, preprocessors, n_jobs=-1)

#

n_times = 1000


import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans from dataset
n_chans = dataset[0][0].shape[0]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    n_times=n_times,
    final_conv_length=30,
)

print(model)


if cuda:
    _ = model.cuda()


model.to_dense_prediction_model()

n_preds_per_input = model.get_output_shape()[2]


from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)


windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=n_times,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)



splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
valid_set = splitted["1test"]  # Session evaluation


from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training import CroppedLoss

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0



batch_size = 64
#n_epochs = 2
n_epochs = 35
clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.cross_entropy,
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
# Model training for a specified number of epochs. ``y`` is None as it is already supplied
# in the dataset.
_ = clf.fit(train_set, y=None, epochs=n_epochs)


# Plot Results

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# # Extract loss and accuracy values
# results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
# df = pd.DataFrame(
#     clf.history[:, results_columns],
#     columns=results_columns,
#     index=clf.history[:, "epoch"],
# )

# df = df.assign(
#     train_misclass=100 - 100 * df.train_accuracy,
#     valid_misclass=100 - 100 * df.valid_accuracy,
# )

# fig, ax1 = plt.subplots(figsize=(8, 3))
# df.loc[:, ["train_loss", "valid_loss"]].plot(
#     ax=ax1, style=["-", ":"], marker="o", color="tab:blue", legend=False, fontsize=14
# )

# ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=14)
# ax1.set_ylabel("Loss", color="tab:blue", fontsize=14)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# df.loc[:, ["train_misclass", "valid_misclass"]].plot(
#     ax=ax2, style=["-", ":"], marker="o", color="tab:red", legend=False
# )
# ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=14)
# ax2.set_ylabel("Misclassification Rate [%]", color="tab:red", fontsize=14)
# ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
# ax1.set_xlabel("Epoch", fontsize=14)

# # where some data has already been plotted to ax
# handles = []
# handles.append(
#     Line2D([0], [0], color="black", linewidth=1, linestyle="-", label="Train")
# )
# handles.append(
#     Line2D([0], [0], color="black", linewidth=1, linestyle=":", label="Valid")
# )
# plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
# plt.tight_layout()




# # Accuracy Plot
# epochs_range = range(1, n_epochs + 1)
# plt.figure()
# plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
# plt.plot(epochs_range, history["test_acc"], label="Test Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Accuracy over Epochs")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Loss Plot
# plt.figure()
# plt.plot(epochs_range, history["train_loss"], label="Train Loss")
# plt.plot(epochs_range, history["test_loss"], label="Test Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss over Epochs")
# plt.legend()
# plt.grid(True)
# plt.show()

import pandas as pd

# Extract relevant metrics from clf.history
results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]

df = pd.DataFrame(
    clf.history[:, results_columns],
    columns=results_columns,
    index=clf.history[:, "epoch"],
)

# Also compute misclassification rate if desired
df["train_misclass"] = 100 - df["train_accuracy"] * 100
df["valid_misclass"] = 100 - df["valid_accuracy"] * 100


import matplotlib.pyplot as plt

epochs_range = df.index  # or: range(1, n_epochs + 1)

# Accuracy Plot
plt.figure(figsize=(8, 3))
plt.plot(epochs_range, df["train_accuracy"], label="Train Accuracy", marker='o', linestyle='-')
plt.plot(epochs_range, df["valid_accuracy"], label="Valid Accuracy", marker='o', linestyle=':')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Accuracy over Epochs", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss Plot
plt.figure(figsize=(8, 3))
plt.plot(epochs_range, df["train_loss"], label="Train Loss", marker='o', linestyle='-')
plt.plot(epochs_range, df["valid_loss"], label="Valid Loss", marker='o', linestyle=':')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Loss over Epochs", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()






from sklearn.metrics import confusion_matrix

from braindecode.visualization import plot_confusion_matrix

# generate confusion matrices
# get the targets
y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

# generating confusion matrix
confusion_mat = confusion_matrix(y_true, y_pred)

# add class labels
# label_dict is class_name : str -> i_class : int
label_dict = windows_dataset.datasets[0].window_kwargs[0][1]["mapping"]
# sort the labels by values (values are integer class labels)
labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]

# plot the basic conf. matrix
plot_confusion_matrix(confusion_mat, class_names=labels)
plt.plot()


print("a")
