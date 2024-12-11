import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import os
import shutil

# Define image dimensions and other parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Directory paths (replace with your dataset paths)
train_dir = 'images_train'
val_dir = 'images_val'
test_dir = 'images_test'

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')  # Multi-class classification

val_generator = val_test_datagen.flow_from_directory(val_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

# Build the CNN model
model = Sequential([
    Input((IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Multi-class classification
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
#model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=20,
                    callbacks=[early_stopping])

# Save the model
model.save('xray_classifier.keras')

# Evaluate the model on test data
test_generator = val_test_datagen.flow_from_directory(test_dir,
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='categorical')
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Load and predict on a new image
def predict_xray_image(image_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    #return prediction
    class_idx = np.argmax(prediction)
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[class_idx]

# Example usage

test_image_dir = "test"
image_files = sorted(os.listdir(test_image_dir))
result_dir = os.path.join(test_image_dir, 'results')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
for i, image_file in enumerate(image_files):
    image_path = os.path.join(test_image_dir, image_file)
    if os.path.isdir(image_path) is not True:
        predicted_class = predict_xray_image(image_path)
        print(f"The predicted class for the {image_file} is: {predicted_class}")
        result_path = os.path.join(result_dir, predicted_class)
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        shutil.copy(image_path, os.path.join(result_path, image_file))
   