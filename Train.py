import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


IMG_HEIGHT = 64
IMG_WIDTH = 64


def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(label)
    return images, labels


train_adults_dir = 'D:\\OnBoardCounterDataSet\\train\\adults'
train_children_dir = 'D:\\OnBoardCounterDataSet\\train\\children'

train_adult_images, train_adult_labels = load_and_preprocess_images(train_adults_dir, label=1)
train_child_images, train_child_labels = load_and_preprocess_images(train_children_dir, label=0)

train_images = train_adult_images + train_child_images
train_labels = train_adult_labels + train_child_labels

X_train = np.array(train_images)
y_train = np.array(train_labels)


test_adults_dir = 'D:\\OnBoardCounterDataSet\\test\\adults'
test_children_dir = 'D:\\OnBoardCounterDataSet\\test\\children'

test_adult_images, test_adult_labels = load_and_preprocess_images(test_adults_dir, label=1)
test_child_images, test_child_labels = load_and_preprocess_images(test_children_dir, label=0)

test_images = test_adult_images + test_child_images
test_labels = test_adult_labels + test_child_labels

X_test = np.array(test_images)
y_test = np.array(test_labels)


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


def classify_image(image):
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0
    prediction = model.predict(img)
    return prediction[0][0]


history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))


test_loss, test_accuracy = model.evaluate(X_test, y_test)


model.save('adult_child_classifier_augmented_transfer.h5')


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()



y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculating classification report which includes precision, recall, F1-score
classification_rep = classification_report(y_test, y_pred, target_names=['child', 'adult'])
print("Classification Report:\n", classification_rep)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


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
