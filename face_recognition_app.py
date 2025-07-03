import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# Load the trained model
model_path = 'C:/Users/admin/Desktop/AI Project/svm_classifier_model.pkl'
loaded_model = joblib.load(model_path)

# Placeholder for class mapping
class_mapping = {0: "Alice", 1: "Bob", 2: "Charlie"}

# Define feature names
feature_names = [f"feature_{i}" for i in range(150)]  # Adjust based on your dataset

# Function to simulate feature extraction (replace with actual embedding extraction logic)
def extract_features(image):
    # For now, simulate with random values
    return pd.DataFrame(np.random.rand(1, 150), columns=feature_names)

# Streamlit UI
st.title("Facial Recognition System")

# Upload or capture image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
capture_image = st.button("Capture from Webcam")

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to numpy array
    image_array = np.array(image)

    # Extract features
    st.write("Extracting features...")
    features = extract_features(image_array)

    # Make predictions
    prediction = loaded_model.predict(features)
    predicted_name = class_mapping.get(prediction[0], "Unknown")
    
    # Display prediction
    st.write(f"Prediction: {predicted_name}")

elif capture_image:
    # Use webcam
    cap = cv2.VideoCapture(0)
    st.write("Starting webcam...")
    ret, frame = cap.read()
    if ret:
        # Display the captured frame
        st.image(frame, caption="Captured Image", use_column_width=True)
        
        # Extract features
        st.write("Extracting features...")
        features = extract_features(frame)

        # Make predictions
        prediction = loaded_model.predict(features)
        predicted_name = class_mapping.get(prediction[0], "Unknown")

        # Display prediction
        st.write(f"Prediction: {predicted_name}")
    else:
        st.write("Failed to capture image. Please try again.")

    cap.release()

st.write("Upload an image or capture a photo to get started!")
