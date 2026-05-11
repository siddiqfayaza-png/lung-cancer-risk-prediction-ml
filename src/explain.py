import shap
import joblib

from src.preprocessing import preprocess_pipeline

# load model
model = joblib.load("models/final_model.pkl")

# processed data
X_train, X_test, y_train, y_test = preprocess_pipeline("../data/raw/lung_cancer.csv")

# SHAP explainer
explainer = shap.Explainer(model)

# explain predictions
shap_values = explainer(X_test, check_additivity=False)

# global interpretation
shap.plots.beeswarm(shap_values)

# local interpretation
shap.plots.waterfall(shap_values[0])
