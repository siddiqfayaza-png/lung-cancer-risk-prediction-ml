import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Encode target variable
def encode_target(df):
    df["Survived"] = df["Survived"].map({"Yes": 1, "No": 0})
    return df


# Encode categorical features
def encode_features(df):
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


# Split features and target
def split_features_target(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y


# Train test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# Balance training data using SMOTE
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# Preprocessing pipeline
def preprocess_pipeline(file_path):

    # Load
    df = load_data(file_path)

    # Encode target
    df = encode_target(df)

    # Encode categorical features
    df = encode_features(df)

    # Split features and target
    X, y = split_features_target(df)

    # Train test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Balance only training data
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test
