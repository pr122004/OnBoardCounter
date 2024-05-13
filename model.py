import cv2

from tensorflow.keras.models import load_model


model = load_model('adult_child_classifier_augmented_transfer.h5')


def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame.astype('float32') / 255.0
    return frame.reshape(1, 64, 64, 3)


cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:

    ret, frame = cap.read()

    input_frame = preprocess_frame(frame)

    prediction = model.predict(input_frame)

    if prediction[0][0] > 0.5:
        label = "Adult"
    else:
        label = "Child"

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()