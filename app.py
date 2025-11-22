# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
st.set_page_config(page_title="AutoML (Streamlit)", layout="wide")

# ------------ helpers ------------
def detect_task_simple(df, target):
    y = df[target]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return 'regression'
    else:
        return 'classification'

def get_feature_cols(df, target):
    # drop both original target and any special __target__ column if present
    X = df.drop(columns=[target, '__target__'], errors='ignore')
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols, for_tree=True):
    if for_tree:
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    else:
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    transformers = []
    if num_cols:
        transformers.append(('num', num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        transformers.append(('cat', cat_pipe, cat_cols))

    # If no transformers, create a passthrough (shouldn't happen usually)
    if not transformers:
        preproc = ColumnTransformer([], remainder='drop')
    else:
        preproc = ColumnTransformer(transformers, remainder='drop')

    return preproc

def compute_classification_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = None
    # compute binary AUC only when probability array present and binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            # y_proba could be (n_samples, n_classes)
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = None
    return acc, f1, auc

def plot_confusion(y_true, y_pred, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig

def show_feature_importance(pipe, feature_names, top_n=20):
    try:
        est = pipe.named_steps['est']
        if hasattr(est, 'feature_importances_'):
            fi = est.feature_importances_
            df = pd.DataFrame({'feature': feature_names, 'importance': fi}).sort_values('importance', ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(6, min(0.4*len(df),6)))
            sns.barplot(x='importance', y='feature', data=df, ax=ax)
            ax.set_title("Feature importances")
            plt.tight_layout()
            return fig
    except Exception:
        return None

# ------------ UI ------------
st.title("AutoML (single-file) — Upload CSV, pick target, train models")

with st.sidebar:
    st.markdown("## Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    fast_mode = st.checkbox("Fast mode (fewer CV trials)", value=True)
    single_split_only = st.checkbox("Single train/test split only (no CV)", value=False)
    compare_by = st.selectbox("Compare models by", ['cv_score','test_metric'], index=0, help="cv_score uses CV primary metric; test_metric uses hold-out test metric")

if uploaded_file is None:
    st.info("Upload a CSV to get started")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write("Data shape:", df.shape)
st.dataframe(df.head(5))

cols = df.columns.tolist()
target = st.selectbox("Select target column", options=cols)

mode_opts = ['auto','binary','multiclass','regression']
mode = st.selectbox("Mode (auto / binary / multiclass / regression)", options=mode_opts, index=0)

# threshold for binary (if chosen) - use reasonable defaults
min_val = float(df[target].min()) if pd.api.types.is_numeric_dtype(df[target]) else 0.0
max_val = float(df[target].max()) if pd.api.types.is_numeric_dtype(df[target]) else 1.0
threshold = st.number_input("Binary threshold (if using binary mode):", min_value=min_val, max_value=max_val, value=min_val if min_val==max_val else (min_val + (max_val-min_val)/2), step=1.0)

run_button = st.button("Train models")

if not run_button:
    st.stop()

# ------------- Prepare data -------------
st.info("Preparing data...")
df_clean = df.copy()

# Decide mode
if mode == 'auto':
    selected_mode = detect_task_simple(df_clean, target)
else:
    selected_mode = mode

st.write("Selected mode:", selected_mode)

if selected_mode == 'binary':
    # require numeric target for thresholding
    if not pd.api.types.is_numeric_dtype(df_clean[target]):
        st.error("Binary mode requires a numeric target for thresholding.")
        st.stop()
    df_clean['__target__'] = (df_clean[target] >= threshold).astype(int)
elif selected_mode in ('multiclass', 'classification'):
    df_clean['__target__'] = df_clean[target]
elif selected_mode == 'regression':
    df_clean['__target__'] = df_clean[target]
else:
    df_clean['__target__'] = df_clean[target]

# Features and target
# drop original target and __target__ safely when building features
X = df_clean.drop(columns=[target, '__target__'], errors='ignore')
y = df_clean['__target__']

num_cols, cat_cols = get_feature_cols(df_clean, target)

# Prepare train/test split
stratify = y if (selected_mode != 'regression' and y.nunique() > 1) else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=RANDOM_STATE, stratify=stratify)

st.write("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- Candidate models ----------
if selected_mode == 'regression':
    candidates = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }
else:
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier()
    }

st.write("Models to evaluate:", list(candidates.keys()))

# ---------- Train & evaluate ----------
st.info("Training models... this may take a bit.")
results = []
for name, estimator in candidates.items():
    st.write("Training:", name)
    tree_like = any(k in name.lower() for k in ['random forest','forest','decision tree','gradient boosting'])
    preproc = build_preprocessor(num_cols, cat_cols, for_tree=tree_like)
    pipe = Pipeline([('preproc', preproc), ('est', estimator)])
    record = {'Model': name}

    # If single-split-only, fit on X_train and evaluate on X_test (no CV)
    if single_split_only:
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            if selected_mode == 'regression':
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                record.update({'Test_R2': r2, 'Test_RMSE': rmse, 'Pipeline': pipe})
            else:
                y_proba = None
                try:
                    if hasattr(pipe, 'predict_proba'):
                        y_proba = pipe.predict_proba(X_test)
                except Exception:
                    y_proba = None
                acc, f1, auc = compute_classification_metrics(y_test, y_pred, y_proba)
                record.update({'Test_accuracy': acc, 'Test_f1_macro': f1, 'Test_auc': auc, 'Pipeline': pipe})
        except Exception as e:
            st.error(f"Error training {name}: {e}")
            continue

    else:
        # use CV on training set for model selection
        try:
            # choose CV scoring
            if selected_mode == 'regression':
                scoring = 'r2'
                cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            else:
                scoring = 'roc_auc' if y_train.nunique() == 2 else 'f1_macro'
                if y_train.nunique() == 2:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                else:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            # cross_val_score executes preprocess because pipe includes preproc
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_score = float(np.mean(scores))
            record['CV_score'] = cv_score

            # fit on training to measure test metrics
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = None
            try:
                if hasattr(pipe, 'predict_proba'):
                    y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None

            if selected_mode == 'regression':
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                record.update({'Test_R2': r2, 'Test_RMSE': rmse, 'Pipeline': pipe})
            else:
                acc, f1, auc = compute_classification_metrics(y_test, y_pred, y_proba)
                record.update({'Test_accuracy': acc, 'Test_f1_macro': f1, 'Test_auc': auc, 'Pipeline': pipe})
        except Exception as e:
            st.error(f"Error during CV for {name}: {e}")
            continue

    results.append(record)

# Build results DataFrame and choose best
if selected_mode == 'regression':
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k in ['Model','CV_score','Test_R2','Test_RMSE']} for r in results])
    sort_col = 'CV_score' if compare_by == 'cv_score' else 'Test_R2'
    if sort_col not in df_res.columns:
        # fallback
        sort_col = 'Test_R2' if 'Test_R2' in df_res.columns else 'CV_score'
    df_res = df_res.sort_values(by=sort_col, ascending=False)
else:
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k in ['Model','CV_score','Test_accuracy','Test_f1_macro','Test_auc']} for r in results])
    sort_col = 'CV_score' if compare_by == 'cv_score' else 'Test_accuracy'
    if sort_col not in df_res.columns:
        sort_col = 'Test_accuracy' if 'Test_accuracy' in df_res.columns else 'CV_score'
    df_res = df_res.sort_values(by=sort_col, ascending=False)

st.subheader("Model comparison")
st.dataframe(df_res.fillna('-'))

# Show best model details
if not results or df_res.empty:
    st.warning("No successful model runs.")
    st.stop()

# pick best model based on df_res first row
best_model_name = df_res.iloc[0]['Model']
best = next((r for r in results if r['Model'] == best_model_name), results[0])

st.subheader("Best model (by selected metric)")
st.write(best['Model'])
if selected_mode == 'regression':
    st.write("Test R2:", best.get('Test_R2'), "Test RMSE:", best.get('Test_RMSE'))
else:
    st.write("Test accuracy:", best.get('Test_accuracy'), "Test F1 (macro):", best.get('Test_f1_macro'), "CV_score:", best.get('CV_score'))
    # confusion matrix
    try:
        y_pred_best = best['Pipeline'].predict(X_test)
        unique_labels = np.unique(y_test)
        fig = plot_confusion(y_test, y_pred_best, labels=unique_labels)
        st.pyplot(fig)
        st.text("Classification report (best model on test):")
        st.text(classification_report(y_test, y_pred_best, zero_division=0))
    except Exception as e:
        st.write("Could not plot confusion matrix:", e)

# Feature importances if available
try:
    # get feature names after preprocessing
    preproc = best['Pipeline'].named_steps.get('preproc', None)
    feature_names = []
    if len(num_cols) > 0:
        feature_names.extend(num_cols)
    if len(cat_cols) > 0 and preproc is not None:
        try:
            # Try to get OneHot encoder inside the pipeline if present
            cat_transformer = None
            if hasattr(preproc, 'named_transformers_') and 'cat' in preproc.named_transformers_:
                cat_transformer = preproc.named_transformers_['cat']
            if cat_transformer is not None and hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                ohe = cat_transformer.named_steps['onehot']
                cats = ohe.get_feature_names_out(cat_cols).tolist()
                feature_names.extend(cats)
            else:
                feature_names.extend(cat_cols)
        except Exception:
            feature_names.extend(cat_cols)

    fi_fig = show_feature_importance(best['Pipeline'], feature_names)
    if fi_fig is not None:
        st.pyplot(fi_fig)
except Exception as e:
    st.write("Feature importance not available:", e)

# Save/download best pipeline
try:
    artifact_buf = io.BytesIO()
    # Use pickle to serialize pipeline into bytes buffer (robust for download)
    pickle.dump(best['Pipeline'], artifact_buf, protocol=pickle.HIGHEST_PROTOCOL)
    artifact_buf.seek(0)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"best_pipeline_{now}.pkl"
    st.download_button("Download best pipeline (pickle)", data=artifact_buf, file_name=fn, mime="application/octet-stream")
except Exception as e:
    st.write("Could not prepare pipeline for download:", e)

st.success("Done")
