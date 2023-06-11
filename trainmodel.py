import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

# Define constants
IMG_HEIGHT, IMG_WIDTH = 480, 640
BATCH_SIZE = 2

# Define the directories
base_dir = 'pics'
resistors_dir = os.path.join(base_dir, 'resistors')
capacitors_dir = os.path.join(base_dir, 'capacitors')
none_dir = os.path.join(base_dir, 'none')  # new

# Use ImageDataGenerator to read and preprocess the data
image_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generate training data
train_generator = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=base_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='categorical',  # changed
                                               subset='training')

# Generate validation data
validation_generator = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                             directory=base_dir,
                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                             class_mode='categorical',  # changed
                                             subset='validation')

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Define the model
model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15  # adjust the number of epochs according to your needs
)


# Save the model
model.save('sensmodel.h5')  # changed
