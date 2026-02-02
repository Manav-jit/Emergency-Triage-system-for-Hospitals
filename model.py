
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("Loading data...")
    try:
        # Load the dataset
        df = pd.read_csv(
            "data.csv",
            sep=";",          # correct delimiter
            encoding="latin1",# correct encoding
            decimal=","       # handle 5,00 -> 5.00
        )
    except FileNotFoundError:
        print("Error: data.csv not found in the current directory.")
        return

    # Define features and target
    FEATURES = [
        "Sex", "Age", "Arrival mode", "Injury",
        "Mental", "Pain", "NRS_pain",
        "SBP", "DBP", "HR", "RR", "BT", "Saturation"
    ]

    TARGET = "KTAS_expert"

    # Select columns and drop rows with missing target
    df = df[FEATURES + [TARGET]]
    df = df.dropna(subset=[TARGET])
    
    # Pre-pipeline cleaning for specific columns
    # Convert Saturation to numeric and fill value
    df["Saturation"] = pd.to_numeric(df["Saturation"], errors='coerce')
    df["Saturation"] = df["Saturation"].fillna(df["Saturation"].median())
    
    # Convert NRS_pain to numeric and fill value
    df["NRS_pain"] = pd.to_numeric(df["NRS_pain"], errors='coerce')
    df["NRS_pain"] = df["NRS_pain"].fillna(df["NRS_pain"].median())

    # Transform Target Variable
    def ktas_to_risk(x):
        if x in [1, 2]:
            return "High"
        elif x == 3:
            return "Medium"
        else:
            return "Low"

    df["risk_level"] = df[TARGET].apply(ktas_to_risk)
    df = df.drop(columns=[TARGET])

    # Clean vital signs - force to numeric
    vitals = ["SBP", "DBP", "HR", "RR", "BT"]
    for col in vitals:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Missing values in vitals after coercion:")
    print(df[vitals].isnull().sum())

    # --- Feature Engineering ---
    print("Performing feature engineering...")
    
    # Pulse Pressure = SBP - DBP
    df["Pulse_Pressure"] = df["SBP"] - df["DBP"]
    
    # Shock Index = HR / SBP
    # Avoid division by zero if SBP is 0 or NaN
    df["Shock_Index"] = df["HR"] / df["SBP"].replace(0, np.nan)

    # Prepare X and y
    X = df.drop(columns=["risk_level"])
    y = df["risk_level"]

    # Define numeric features (including new ones) and categorical features
    numeric_features = [
        "Age", "NRS_pain",
        "SBP", "DBP", "HR", "RR", "BT", "Saturation",
        "Pulse_Pressure", "Shock_Index"
    ]

    categorical_features = [
        "Sex", "Arrival mode", "Injury",
        "Mental", "Pain"
    ]

    # Build Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Define Base Classifier Pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Hyperparameter Tuning ---
    print("Starting Hyperparameter Tuning with GridSearchCV...")
    
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
    
    # Train
    grid_search.fit(X_train, y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Select best model
    best_model = grid_search.best_estimator_

    # Evaluation
    print("\nEvaluating best model on test set:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        plt.figure(figsize=(8,6))
        labels = sorted(y.unique())
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - Enchanced Model')
        plt.tight_layout()
        plt.savefig('confusion_matrix_tuned.png')
        print("Confusion matrix saved to 'confusion_matrix_tuned.png'")
    except Exception as e:
        print(f"Warning: Could not save plot. Error: {e}")

if __name__ == "__main__":
    main()
