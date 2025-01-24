import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import os
import shutil

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the CNN model.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(image_path):
    """
    Extract feature embeddings from an image using the CNN model.
    """
    preprocessed_image = preprocess_image(image_path)
    features = model.predict(preprocessed_image)
    return features.flatten()


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



# Define image directory

image_dir = "images"
image_cmp_dir = "test\\results\\front"

# Compare all images to image_5.jpg, front view
target_image = "image_5.jpg"
target_path = os.path.join(image_dir, target_image)


# Load pre-trained CNN model (VGG16) for feature extraction
model = VGG16(weights="imagenet", include_top=False)

# Load and process all images
image_features = {}
image_files = sorted(os.listdir(image_cmp_dir))

for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_cmp_dir, image_file)
    features = extract_features(image_path)
    image_features[image_file] = features
    print(f"Extracted features for {image_file}")


if not os.path.isfile(target_path):
    print(f"{target_image} not found in image directory.")
    
else:
    target_features = extract_features(target_path)
    similarities = {}

    for image_file, features in image_features.items():
        if image_file != target_image:
            sim = cosine_similarity(
                [target_features], [features]
            )[0][0]  # Compute cosine similarity
            similarities[image_file] = sim

    # Sort images by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    print(f"Images similar to {target_image}:")
    for image_file, similarity in sorted_similarities:
        print(f"{image_file}: {similarity:.4f}")