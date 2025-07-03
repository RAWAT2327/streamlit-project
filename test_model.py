import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the saved model
model_path = 'C:/Users/admin/Desktop/AI Project/svm_classifier_model.pkl'
print("Loading the trained SVM model...")
loaded_model = joblib.load(model_path)
print("Model loaded successfully!")

# Simulate new test data (replace these with real embeddings or test data if available)
feature_names = [f"feature_{i}" for i in range(150)]  # Ensure 150 matches your dataset's feature count
new_data = pd.DataFrame([
    [0.5] * 150,  # Example data point 1
    [0.2] * 150   # Example data point 2
], columns=feature_names)

# Simulate true labels (replace these with actual labels if available)
true_labels = [1, 0]  # Example labels for the above data points

# Predict using the model
print("Making predictions on new data...")
predictions = loaded_model.predict(new_data)
print("Predictions:", predictions)

# Calculate accuracy (for real-world data with known labels)
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy on test data: {accuracy:.2f}")
