import cv2
import numpy as np
import tensorflow as tf
import h5py
from uuid import uuid4

# Load the trained model
model = tf.keras.models.load_model('dataset.h5')

# Load the dataset.h5 file
# dataset = h5py.File('dataset.h5', 'r')
# x_test = dataset['x_test'][:]
# y_test = dataset['y_test'][:]
# dataset.close()

# Function to predict the number
def predict_number(image):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image /= 255

    # Perform the prediction
    predictions = model.predict(image)
    predicted_number = np.argmax(predictions[0])
    accuracy = predictions[0][predicted_number]

    return predicted_number, accuracy

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the square region of interest (ROI)
square_size = 200
square_color = (0, 255, 0)  # Green color (BGR format)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
roi_top_left = (int((frame_width - square_size) / 2), int((frame_height - square_size) / 2))
roi_bottom_right = (roi_top_left[0] + square_size, roi_top_left[1] + square_size)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Draw the square ROI
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, square_color, 2)

    # Extract the ROI
    roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Predict the number if ROI is not empty
    if roi.size > 0:
        predicted_number, accuracy = predict_number(roi)

        # Display the predicted number and accuracy
        cv2.putText(frame, f"Number: {predicted_number}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.2%}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save the recognized number as a JPG file
        if accuracy > 0.9:
            number_folder = f"dataset/{predicted_number}"
            image_name = f"{uuid4()}.jpg"
            cv2.imwrite(f"{number_folder}/{image_name}", roi)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
