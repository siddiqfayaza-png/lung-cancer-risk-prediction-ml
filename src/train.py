import os
import pandas as pd
import joblib
from preprocessing import preprocess_pipeline

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create models folder
os.makedirs("models", exist_ok=True)

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_pipeline("data/raw/lung_cancer.csv")

# logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# xgboost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# stote models and predictions in a dictionary
models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
}

# Save BEST model
joblib.dump(rf_model, "models/final_model.pkl")

print("Training completed successfully.")
print("Final model saved as models/final_model.pkl")
