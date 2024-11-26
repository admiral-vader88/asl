import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelBinarizer

# Load and preprocess the data
base_path = "C:/Users/prana/OneDrive/Documents/slt2/Sign-Language-Detection-main"
train_df = pd.read_csv(os.path.join(base_path, "sign_mnist_train/sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(base_path, "sign_mnist_test/sign_mnist_test.csv"))

y_train = train_df.pop('label')
y_test = test_df.pop('label')

# One-hot encoding labels
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Reshape and normalize image data
x_train = train_df.values.reshape(-1, 28, 28, 1) / 255.0
x_test = test_df.values.reshape(-1, 28, 28, 1) / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# Model architecture
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')  # 24 classes for A-Y (excluding J and Z in dataset)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better performance
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

# Training the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    validation_data=(x_test, y_test),
    epochs=30,
    callbacks=callbacks
)

# Save the trained model
model.save('sign_language_model.h5')
