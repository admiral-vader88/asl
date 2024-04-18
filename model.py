import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

# Read dataset
base_path = "C:/Users/prana/OneDrive/Documents/slt2/Sign-Language-Detection-main"
train_df = pd.read_csv(os.path.join(base_path, "sign_mnist_train/sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(base_path, "sign_mnist_test/sign_mnist_test.csv"))

# Process labels
y_train = train_df.pop('label')
y_test = test_df.pop('label')
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Prepare dataset
x_train = train_df.values.reshape(-1, 28, 28, 1) / 255.0
x_test = test_df.values.reshape(-1, 28, 28, 1) / 255.0


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Build model
model = Sequential([
    Conv2D(75, (3,3), strides=1, padding='same', activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPool2D((2,2), strides=2, padding='same'),
    Conv2D(50, (3,3), strides=1, padding='same', activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPool2D((2,2), strides=2, padding='same'),
    Conv2D(25, (3,3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D((2,2), strides=2, padding='same'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(24, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]
history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                    epochs=20,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

# Save model
model.save('smnist.h5')
