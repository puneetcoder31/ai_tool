# tool_no_cv.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42  # make results reproducible


def load_data(path):
    return pd.read_csv(path)


def get_feature_columns(df, target):
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, scale_non_tree=True):
    # For simplicity: numeric -> median impute (+scale optionally)
    if scale_non_tree:
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler())])
    else:
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])

    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]) if cat_cols else 'drop'

    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop', sparse_threshold=0)

    return preproc


def get_models():
    # Default hyperparameters (no tuning). Use probability for SVC off for speed (but enable predict_proba if needed).
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'SVM': SVC(probability=False, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier()
    }


def run_single_split_compare(df, target, test_size=0.2, random_state=RANDOM_STATE):
    if target not in df.columns:
        raise ValueError("Target column not found in dataframe")

    # Prepare X and y
    X = df.drop(columns=[target]).copy()
    y_raw = df[target].copy()

    # Encode target if non-numeric
    label_map = None
    if y_raw.dtype == object or not np.issubdtype(y_raw.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        label_map = dict(enumerate(le.classes_))
    else:
        y = y_raw.values

    # Train/test split (stratify if classification)
    strat = y if len(np.unique(y)) <= 20 and (np.array(y).dtype != float) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=strat)

    num_cols, cat_cols = get_feature_columns(df, target)

    models = get_models()
    results = []

    for name, model in models.items():
        # Decide whether to scale numeric features for this model type:
        # Trees don't need scaling; linear/SVM/KNN benefit from scaling.
        scale_non_tree = not any(k in name.lower() for k in ['random forest', 'forest', 'decision tree', 'gradient boosting'])
        preproc = build_preprocessor(num_cols, cat_cols, scale_non_tree=scale_non_tree)

        pipe = Pipeline([('preproc', preproc), ('est', model)])

        # Fit on train
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            clf_report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
        except Exception as e:
            acc = None
            clf_report = f"Error fitting/predicting: {e}"
            cm = None

        results.append({
            'Model': name,
            'Test_accuracy': acc,
            'Classification_report': clf_report,
            'Confusion_matrix': cm,
            'Pipeline': pipe
        })

        print(f"{name} -> Test accuracy: {acc}")

    # Choose best by Test_accuracy (highest)
    valid_results = [r for r in results if r['Test_accuracy'] is not None]
    best = max(valid_results, key=lambda r: r['Test_accuracy']) if valid_results else None

    # Save best pipeline for later use
    if best:
        joblib.dump({'pipeline': best['Pipeline'], 'target': target, 'label_map': label_map}, 'best_model_no_cv.joblib')
        print(f"\nBest model: {best['Model']} (accuracy = {best['Test_accuracy']:.4f})")
        print("Saved pipeline to best_model_no_cv.joblib")
    else:
        print("No successful model trainings.")

    # Print a tidy summary table
    try:
        import pandas as pd
        df_summary = pd.DataFrame([{'Model': r['Model'], 'Test_accuracy': r['Test_accuracy']} for r in results])
        df_summary = df_summary.sort_values(by='Test_accuracy', ascending=False).reset_index(drop=True)
        print("\n--- Summary ---")
        print(df_summary.to_string(index=False))
    except Exception:
        pass

    # Optionally print detailed report for best model
    if best:
        print(f"\n--- Classification report for best model ({best['Model']}) ---")
        print(best['Classification_report'])
        print("\nConfusion matrix:")
        print(best['Confusion_matrix'])
        if label_map:
            print("\nLabel mapping (encoded -> original):")
            pprint(label_map)

    return results, best


def interactive_main():
    path = input("Enter path to CSV dataset: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        sys.exit(1)

    df = load_data(path)
    print("\nColumns:", df.columns.tolist())
    target = input("Enter the target column name: ").strip()
    if target not in df.columns:
        print("Target not found.")
        sys.exit(1)

    print("\nRunning single train/test split comparison (no CV). This will pick the model with highest test accuracy.")
    run_single_split_compare(df, target)


if __name__ == "__main__":
    interactive_main()
