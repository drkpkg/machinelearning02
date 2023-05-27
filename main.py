import os
import uuid

import cv2
import numpy as np
import tensorflow as tf

# Load the MNIST dataset
(_, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Load the trained machine learning model
model = tf.keras.models.load_model('dataset.h5')

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Preprocess the frame to be in the same format as the MNIST dataset
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    input_image = np.expand_dims(normalized, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)

    # Pass the preprocessed frame through the model to obtain predictions
    prediction = model.predict(input_image)[0]
    predicted_label = np.argmax(prediction)

    # Check the accuracy of the prediction
    accuracy = prediction[predicted_label] * 100

    contours, hierarchy = cv2.findContours(resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Show a rectangle around the digit in the frame
        x, y, w, h = cv2.boundingRect(contours[0])
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        # check if dataset folder exists dataset/{predicted_label}
        if not os.path.exists(f'dataset/{predicted_label}'):
            os.makedirs(f'dataset/{predicted_label}')
        # Add frame to the dataset
        if accuracy > 90:
            cv2.imwrite(f'dataset/{predicted_label}/{str(uuid.uuid4())}.jpg', resized[y:y + h, x:x + w])

    # Overlay the predicted label onto the frame
    cv2.putText(frame, f"Pred: {str(predicted_label)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, f"Acc: {str(round(accuracy, 2))}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the overlay and rectangle
    cv2.imshow('frame', frame)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
