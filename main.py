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

    # Find contours around the hand-written number
    contours, hierarchy = cv2.findContours(resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Find the contour with the largest area
        max_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box for the contour and draw a rectangle around it
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Overlay the predicted label onto the frame
    cv2.putText(frame, str(predicted_label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Display the frame with the overlay and rectangle
    cv2.imshow('frame', frame)

    # Check for user input to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
