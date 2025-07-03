import cv2
import numpy as np
import joblib

# Load the saved SVM model
model_path = 'C:/Users/admin/Desktop/AI Project/svm_classifier_model.pkl'
print("Loading the trained SVM model...")
loaded_model = joblib.load(model_path)
print("Model loaded successfully!")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Placeholder for class mapping
class_mapping = {0: "Person1", 1: "Person2", 2: "Person3"}

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

print("Starting video stream...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Placeholder for embeddings (replace this with actual embedding extraction)
        # Here, we simulate embeddings using random numbers
        face_embedding = np.random.rand(1, 150)  # Replace 150 with the correct number of features

        # Predict the label
        prediction = loaded_model.predict(face_embedding)
        predicted_name = class_mapping.get(prediction[0], "Unknown")

        # Display the name on the frame
        cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
