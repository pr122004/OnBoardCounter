import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 259

# Function to load and preprocess images
def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

# Load training images
train_adults_dir = 'D:\\OnBoardCounterBlurred\\TrainBlurred\\AdultBlurred'
train_children_dir = 'D:\\OnBoardCounterBlurred\\TrainBlurred\\ChildBlurred'

train_adult_images, train_adult_labels = load_and_preprocess_images(train_adults_dir, label=1)
train_child_images, train_child_labels = load_and_preprocess_images(train_children_dir, label=0)

train_images = train_adult_images + train_child_images
train_labels = train_adult_labels + train_child_labels

X_train = np.array(train_images)
y_train = np.array(train_labels)

# Load test images
test_adults_dir = 'D:\\OnBoardCounterBlurred\\TestBlurred\\AdultBlurred'
test_children_dir = 'D:\\OnBoardCounterBlurred\\TestBlurred\\ChildBlurred'

test_adult_images, test_adult_labels = load_and_preprocess_images(test_adults_dir, label=1)
test_child_images, test_child_labels = load_and_preprocess_images(test_children_dir, label=0)

test_images = test_adult_images + test_child_images
test_labels = test_adult_labels + test_child_labels

X_test = np.array(test_images)
y_test = np.array(test_labels)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)


datagen.fit(X_train)

# Base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in base_model.layers:
    layer.trainable = False
    # base_model.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),  # Reduce the number of neurons
    layers.Dropout(0.4),  # Adjust dropout rate
    layers.Dense(512, activation='relu'),  # Reduce the number of neurons
    layers.Dropout(0.4),  # Adjust dropout rate
    layers.Dense(256, activation='relu'),  # Reduce the number of neurons
    layers.Dropout(0.3),  # Adjust dropout rate
    layers.Dense(256, activation='relu'),  # Reduce the number of neurons
    layers.Dropout(0.3),  # Adjust dropout rate+---
    layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' activation for binary classification
])

model.compile(optimizer="adam",
              loss='binary_crossentropy',  # Use 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Training
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=10,
                    validation_data=(X_test, y_test),
                    class_weight={0: 1., 1: 1.},
                    callbacks=[reduce_lr, early_stop])


# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save('adult_child_classifier_augmented_transfer.h5')

# Plotting
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification report
classification_rep = classification_report(y_test, y_pred, target_names=['child', 'adult'])
print("Classification Report:\n", classification_rep)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['child', 'adult']))
plt.xticks(tick_marks, ['child', 'adult'], rotation=45)
plt.yticks(tick_marks, ['child', 'adult'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
plt.show()

