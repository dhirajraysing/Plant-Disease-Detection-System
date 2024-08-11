import pandas as pd
import numpy as np
#pip install opencv-python
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#load dataset
data_path = './sugarcane_dataset'
categories = ['healthy', 'mosaic', 'redrot', 'yellow']

#prepare data
data = []
labels = []

for category in categories:
    class_num = categories.index(category)
    path = os.path.join(data_path, category)
    print(f"Processing category: {category}, Path: {path}")
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            print(f"Reading image: {img_path}")
            img_array = cv2.imread(img_path)
            if img_array is None:
                print(f"Failed to read image: {img_path}")
                continue
            img_array = cv2.resize(img_array, (128, 128))
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error reading image {img}: {e}")

#print(f"Total images loaded: {len(data)}")
#print(f"Total labels loaded: {len(labels)}")

# Convert to numpy arrays
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encoding
labels = to_categorical(labels, num_classes=len(categories))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save('sugarcane_leaf_disease_model.keras')