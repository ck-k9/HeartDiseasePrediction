import sys
import os

# Ensure Python finds the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_and_preprocess_data
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🔄 Loading and preprocessing data...")
X_train, X_test, y_train, y_test = load_and_preprocess_data("dataset/heart.csv")

print(f"✅ Data Loaded: X_train: {X_train.shape}, X_test: {X_test.shape}")

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("best_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("✅ Model saved as best_model.pkl")
