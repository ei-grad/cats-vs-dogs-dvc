# Rye + DVC: Cats vs Dogs Keras Tutorial

This repository guides you through building a machine learning pipeline for classifying images from the **Kaggle Cats vs Dogs** dataset using a custom CNN model built from scratch with **Keras**. The project uses **DVC** (Data Version Control) to manage the pipeline stages, datasets, and model metrics, while **Rye** is used to manage the Python environment. Follow the steps in this `README.md` to create and execute the pipeline stages manually.

## Project Overview

The pipeline includes the following stages:
1. **Download**: Download the dataset.
2. **Prepare**: Filter out corrupted images and save good ones into a TensorFlow dataset.
3. **Train**: Train a custom CNN model using Keras.
4. **Validate**: Evaluate the model on the validation data.

Each stage is created manually, and the results are tracked with DVC. Additionally, the **Tracking Metrics with Git Branches** section explains how to use Git branches and DVC metrics to track and compare model performance across different experiments.

## Requirements

- [Git](https://git-scm.com/)
- [Rye](https://rye.astral.sh/) (for environment management)
- [DVC](https://dvc.org/doc/install)
- **Curl** for downloading the dataset
- **Keras**
- **Matplotlib** for visualization
- **Dvclive** for tracking metrics

## Setting Up the Project

### 1. Fork the Repository

First, fork this repository into your own GitHub account. After forking, clone the repository:

```bash
git clone https://github.com/your-username/cats-vs-dogs-dvc.git
cd cats-vs-dogs-dvc
```

### 2. Install Rye

If you don't have Rye is not installed, follow the installation instructions from the [official documentation](https://rye.astral.sh/guide/installation/).

### 3. Initialize the Python project and virtual environment

Rye [workspaces](https://rye.astral.sh/guide/workspaces/) and [virtual projects](https://rye.astral.sh/guide/virtual/) simplify dependency management for non-package projects. In this tutorial, since we don't have an actual Python package or library, running `rye init --virtual` ensures the project only syncs its dependencies (e.g., `keras`, `dvc`, `dvclive`). This setup is ideal for managing tools without creating a package. Additionally, packages or libraries can be added later in this workspace, sharing the same virtual environment.

Run the following command to initialize a virtual environment and install the required dependencies:

```bash
rye init --virtual
rye add dvc keras tensorflow matplotlib dvclive
```

This command creates a Python virtual environment and installs the necessary dependencies for the project.

After initializing the environment and installing the dependencies, commit the initial state of the project:

```bash
git add . && git commit -m "Initialized python project via Rye"
```

### 4. Activate the Virtual Environment

Activate the virtual environment with:

```bash
source .venv/bin/activate
```

This command modifies the shell's environment variables to use Python and packages installed in the `.venv` directory, isolating project dependencies from the global Python environment. If you start a new shell session, call this command to ensure the correct environment is used.

If DVC is not used directly in your scripts, you can install it in a separate dedicated virtual environment with:

```bash
rye install dvc
```

### 5. Initialize DVC

Next, initialize DVC in the project directory:

```bash
dvc init

# Enable automatic `git add` after DVC stages
dvc config core.autostage true

git add . && git commit -m "Initialized DVC"
```

## Pipeline Stages

### 1. Download the Dataset

Create a script `download.sh` to download:

```bash
#!/bin/bash

mkdir data && cd data
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```

Run it in the DVC stage named `download`:

```bash
dvc stage add --run -n download \
  -d download.sh \
  -o data/kagglecatsanddogs_5340.zip \
  bash download.sh
```

Commit the changes:

```bash
git add . && git commit -m "Added download stage"
```

### 2. Prepare the Data

Create a script `prepare.py` to filter out corrupted images and generate a dataset using Keras' `image_dataset_from_directory` utility. Apply data augmentation and prefetching for better performance:

```python
import shutil
import zipfile

import keras


with zipfile.ZipFile("data/kagglecatsanddogs_5340.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Create training and validation datasets
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "data/PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=(180, 180),
    batch_size=None,
)

# Save the processed datasets
# Skip corrupt images via `.ignore_errors()` (otherwise the
# `Dataset.save()` will hang)
print("Saving the train dataset...")
train_ds.ignore_errors().save("data/train_ds")
print("Saving the validation dataset...")
val_ds.ignore_errors().save("data/val_ds")

# Remove PetImages directory, as it's no longer needed
shutil.rmtree("data/PetImages")
```

Run the `prepare` stage:

```bash
dvc stage add --run -n prepare \
  -d prepare.py \
  -d data/kagglecatsanddogs_5340.zip \
  -o data/train_ds \
  -o data/val_ds \
  python prepare.py
```

Commit the changes:

```bash
git add . && git commit -m "Added prepare stage"
```

### 3. Train the Model

Create a script `train.py` to build and train a CNN using Keras and track metrics with Dvclive:

```python
import keras
from keras import layers
from dvclive import Live
from tensorflow.data import Dataset, AUTOTUNE


# Model creation
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    outputs = layers.Dense(units, activation=None)(x)

    return keras.Model(inputs, outputs)


# Data augmentation
def data_augmentation(images):
    for layer in [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]:
        images = layer(images)
    return images


batch_size = 128

# Load datasets
train_ds = Dataset.load("data/train_ds").batch(batch_size, drop_remainder=True)
val_ds = Dataset.load("data/val_ds").batch(batch_size, drop_remainder=True)

# Apply data augmentation and prefetching
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

learning_rate = 3e-4
epochs = 25

model = make_model(input_shape=(180, 180, 3), num_classes=2)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Track training metrics with Dvclive
with Live() as live:
    live.log_param("learning_rate", learning_rate)
    live.log_param("optimizer", model.optimizer.__class__.__name__)
    live.log_param("epochs", epochs)

    for epoch in range(epochs):
        history = model.fit(train_ds, validation_data=val_ds, epochs=1)

        acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        live.log_metric("train_accuracy", acc)
        live.log_metric("val_accuracy", val_acc)
        live.log_metric("train_loss", loss)
        live.log_metric("val_loss", val_loss)
        live.next_step()

model.save('model/cats_vs_dogs.keras')
```

Run the `train` stage:

```bash
dvc stage add --run -n train \
  -d train.py \
  -d data/train_ds \
  -o model/cats_vs_dogs.keras \
  python train.py
```

Commit the changes and trained model:

```bash
git add . && git commit -m "Added train stage"
```

### 4. Validate the Model

Create a script `validate.py` to evaluate the model on the validation data and save metrics:

```python
import keras
from tensorflow.data import Dataset
import numpy as np
import json

# Load model and validation dataset
model = keras.models.load_model('model/cats_vs_dogs.keras')
val_ds = Dataset.load('data/val_ds').batch(128, drop_remainder=True)

# Evaluate the model
loss, acc = model.evaluate(val_ds)

# Save metrics
metrics = {"accuracy": acc}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"Validation accuracy: {acc:.2f}")
```

Run the `validate` stage:

```bash
dvc stage add --run -n validate \
  -d validate.py \
  -d model/cats_vs_dogs.h5 \
  -d data/val_ds \
  -M metrics.json \
  python validate.py
```

Commit the changes:

```bash
dvc repro
git add . && git commit -m "Added validate stage"
```
