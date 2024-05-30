import cv2
import numpy as np
import torch
from sort import Sort
from tensorflow.keras.models import load_model

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the pre-trained classification model

classification_model = load_model('adult_child_classifier_augmented_transfer.h5')

# Constants for image resizing
IMG_HEIGHT = 224
IMG_WIDTH = 259

# Initialize SORT tracker
tracker = Sort()

# Set of tracked IDs
tracked_ids = set()

# Function to preprocess an image
# Function to classify a person image
# Function to classify a person image
def classify_person(person_img):
    if person_img.shape[0] == 0:
        print("Skipping classification for empty image.")
        return None
    print("Person image shape before preprocessing:", person_img.shape)
    # Preprocess the person image
    person_img = preprocess_image(person_img)
    print("Person image shape after preprocessing:", person_img.shape)
    if person_img is None:
        print("Skipping classification due to preprocessing error.")
        return None
    # Classify the person using the classification model
    prediction = classification_model.predict(np.expand_dims(person_img, axis=0))
    # Convert the prediction to a human-readable label
    label = 'Adult' if prediction[0][0] > 0.5 else 'Child'  # Assuming a threshold of 0.5
    return label

# Function to preprocess an image
def preprocess_image(img):
    print("Original image shape:", img.shape)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    print("Resized image shape:", img.shape)
    return img


# Function to count unique people and classify them
# Function to count unique people and classify them
# Function to count unique people and classify them
def count_and_classify_people(tracked_objects, frame):
    total_count = len(tracked_objects)
    adult_count = 0
    child_count = 0
    for obj in tracked_objects:
        track_id, x1, y1, x2, y2 = obj
        # Extract person region
        person_region = frame[int(y1):int(y2), int(x1):int(x2)]
        # Classify the person
        label = classify_person(person_region)
        # Increment counts based on classification result
        if label == 'Adult':
            adult_count += 1
        elif label == 'Child':
            child_count += 1

    # Adjust counts if some classifications were skipped
    if total_count == 1 and (adult_count == 0 and child_count == 0):
        if adult_count == 0:
            adult_count = 1
        elif child_count == 0:
            child_count = 1
    elif total_count == 2 and (adult_count == 0 and child_count == 0):
        if adult_count == 0:
            adult_count = 2
        elif child_count == 0:
            child_count = 2

    return total_count, adult_count, child_count


# Capture video feed from camera
cap = cv2.VideoCapture(0)

# Loop to process each frame in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv5 on the blurred frame
    results = yolo_model(frame)

    # Get bounding boxes, class labels, and scores of detected objects
    boxes = results.xyxy[0][:, :4].cpu().numpy()
    scores = results.xyxy[0][:, 4].cpu().numpy()

    # Prepare detections for SORT tracker
    detections = np.array([[*box, score] for box, score in zip(boxes, scores)])

    # Update SORT tracker with detections if there are any
    if detections.shape[0] > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = []

    # Count and classify unique people
    total_count, adult_count, child_count = count_and_classify_people(tracked_objects, frame)
    print(f"Total People Detected: {total_count}, Adults: {adult_count}, Children: {child_count}")

    # Draw bounding boxes and IDs on the frame
    for obj in tracked_objects:
        track_id, x1, y1, x2, y2 = obj
        color = (0, 255, 0)  # green color for bounding boxes
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the original frame with detected objects
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
