# Import necessary libraries
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# Evaluate one model(Template)
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


# Evaluate all models
def evaluate_all(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        evaluate_model(name, y_test, y_pred)


# Plot confusion matrix for one model(Template)
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# ROC AUC Score for one model(Template)
def roc_score(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", score)
