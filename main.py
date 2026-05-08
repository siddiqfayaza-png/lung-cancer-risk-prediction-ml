import sys
sys.path.append("..")

from src.preprocessing import preprocess_pipeline

X_train, X_test, y_train, y_test = preprocess_pipeline("../data/raw/lung_cancer.csv")

print(X_train.shape)
print(y_train.value_counts())
