import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('old.h5')

# Define the image dimensions (same as what you trained on)
IMG_WIDTH, IMG_HEIGHT = 640, 480

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the size the model was trained on
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # Prepare the image to feed into the model
    image = img_to_array(resized_frame)
    image = np.expand_dims(image, axis=0)

    # Predict the class of the image
    result = model.predict(image)

    # Interpret the result
    if np.argmax(result) == 0:
        prediction = 'resistor'
    elif np.argmax(result) == 1:
        prediction = 'capacitor'
    else:
        prediction = 'none'

    # Display the prediction on the frame
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
