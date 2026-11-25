
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_data" / "netinsight_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading cleaned dataset...")

df = pd.read_csv(DATA_FILE)

# Separate input features and target
X = df.drop(columns=["Label", "Target"])
y = df["Target"]

# Keep only numeric features
X = X.select_dtypes(include=["number"])

# Handle invalid values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

print("Dataset shape:", X.shape)
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL TRAINING COMPLETED!")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and feature names
joblib.dump(model, MODEL_DIR / "netinsight_model.pkl")
joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")

print("\nModel saved successfully!")
print("Location:", MODEL_DIR)

