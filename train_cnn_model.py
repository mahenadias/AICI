import os
import pickle

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load face data and user data
with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

with open('data/users.pkl', 'rb') as f:
    user_data = pickle.load(f)

# Prepare data for training
X = []
y = []

for i, face_samples in enumerate(faces_data):
    for face in face_samples:
        X.append(face)
        y.append(i)  # Assign label i to all samples of person i

X = np.array(X)
y = np.array(y)

# Normalize pixel values (0-255) to (0-1)
X = X / 255.0

# Reshape X to fit the CNN model input shape (samples, height, width, channels)
X = X.reshape(X.shape[0], 64, 64, 1)  # Update to (64, 64, 1) as per resized face samples

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(user_data), activation='softmax'))  # Output layer matches number of people

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model into the 'data' folder
output_model_path = os.path.join('data', 'face_recognition_cnn.h5')
model.save(output_model_path)

print(f"Model trained and saved successfully at {output_model_path}!")