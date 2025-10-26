"""
train_model.py - Robust training pipeline with XGBoost, safe SMOTE/SMOTENC handling, and fallback.

This version:
- uses the powerful XGBoost classifier
- chooses a safe k_neighbors for SMOTENC based on actual training counts
- if SMOTENC fails, falls back to RandomOverSampler to a reasonable target size
- retains safe calibration logic (calibrate only when enough samples per class)
- saves artifacts required by the Flask app
"""
import os, json, joblib, warnings
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

# optional imblearn imports
try:
    from imblearn.over_sampling import SMOTENC, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTENC = None
    RandomOverSampler = None
    IMBLEARN_AVAILABLE = False

# optional xgboost import
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

DATA_PATH = 'data/Disease_symptom_and_patient_profile_dataset.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# -------- CONFIG --------
USE_TOP_N = True
TOP_N = 25
USE_GROUPING = True
USE_SMOTE = True
USE_CALIBRATION = True
RANDOM_STATE = 42
# ------------------------

FEATURE_COLUMNS = ['Fever','Cough','Fatigue','DB','Age','Gender','BP','CL']

GROUP_MAP = {
    # extend this mapping if you want grouping
    'Hypertension': 'Cardio', 'Coronary Artery Disease': 'Cardio',
    'Asthma': 'Respiratory', 'Bronchitis': 'Respiratory', 'Pneumonia': 'Respiratory',
    'Diabetes': 'Metabolic', 'Hyperthyroidism': 'Endocrine', 'Hypothyroidism': 'Endocrine',
    'Influenza': 'Infectious', 'Measles': 'Infectious', 'Rubella': 'Infectious', 'Malaria': 'Infectious'
}

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}.")
    df = pd.read_csv(path)
    # normalize long column names to short
    rename_map = {'Difficulty Breathing':'DB', 'Blood Pressure':'BP', 'Cholesterol Level':'CL'}
    for long, short in rename_map.items():
        if long in df.columns and short not in df.columns:
            df[short] = df[long]
    return df

def basic_sanity(df):
    required = ['Disease'] + FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df = df[required].dropna().reset_index(drop=True)
    return df

def keep_top_n_classes(df, n=10):
    counts = df['Disease'].value_counts()
    top = counts.nlargest(n).index.tolist()
    df['Disease'] = df['Disease'].apply(lambda x: x if x in top else 'Other')
    return df, top, counts.to_dict()

def map_to_groups(df):
    df = df.copy()
    df['Disease'] = df['Disease'].apply(lambda x: GROUP_MAP.get(x, x))
    return df

def handle_rare_classes(df, min_count=2):
    counts = df['Disease'].value_counts()
    rare = counts[counts < min_count].index.tolist()
    metadata = {}
    if not rare:
        metadata['rare_merged'] = []
        return df, metadata
    df['Disease_orig'] = df['Disease'].astype(str)
    df['Disease'] = df['Disease'].apply(lambda x: 'Other' if x in rare else x)
    new_counts = df['Disease'].value_counts()
    metadata['rare_merged'] = rare
    metadata['other_count'] = int(new_counts.get('Other', 0))
    if new_counts.get('Other',0) < min_count:
        df = df[df['Disease'] != 'Other'].reset_index(drop=True)
        metadata['other_dropped'] = True
    else:
        metadata['other_dropped'] = False
    return df, metadata

def encode_features_and_target(df):
    encoders = {}
    df_enc = df.copy()
    cat_cols = ['Fever','Cough','Fatigue','DB','Gender','BP','CL']
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = {'classes': le.classes_.tolist()}
    le_t = LabelEncoder()
    df_enc['Disease_lbl'] = le_t.fit_transform(df_enc['Disease'].astype(str))
    encoders['Disease'] = {'classes': le_t.classes_.tolist()}
    X = df_enc[FEATURE_COLUMNS]
    y = df_enc['Disease_lbl']
    return X, y, encoders, le_t, df_enc

def safe_classification_report(y_true, y_pred, le_target):
    labels_present = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    try:
        target_names = le_target.inverse_transform(labels_present)
    except Exception:
        all_names = list(le_target.classes_)
        target_names = [all_names[i] if i < len(all_names) else str(i) for i in labels_present]
    report = classification_report(y_true, y_pred, labels=labels_present, target_names=target_names, zero_division=0)
    return report, labels_present, target_names

def apply_resampling_with_fallback(X_train, y_train, metadata):
    """
    Try SMOTENC with a safe k_neighbors. If it fails, fallback to RandomOverSampler
    upsampling minorities to a reasonable target_count (median of class counts or at least 3).
    Returns X_train_res, y_train_res and updates metadata.
    """
    train_counts = Counter(y_train.tolist())
    counts_list = np.array(list(train_counts.values()))
    min_count = int(counts_list.min()) if counts_list.size else 0
    max_count = int(counts_list.max()) if counts_list.size else 0
    median_count = int(np.median(counts_list)) if counts_list.size else 0

    print("Training class counts (sample):", dict(list(train_counts.items())[:12]))
    print("Minimum training examples for any class:", min_count)

    # If imblearn not available, skip and return original
    if not IMBLEARN_AVAILABLE:
        print("imblearn not available; install 'imbalanced-learn' to enable SMOTE. Skipping resampling.")
        metadata['smote_applied'] = False
        return X_train, y_train

    # Choose safe k_neighbors: cannot exceed (min_count - 1). Ensure at least 1.
    k_neighbors = max(1, min(5, min_count - 1)) if min_count >= 2 else None

    if k_neighbors is None:
        print("Not enough samples for SMOTE (min_count < 2). Falling back to RandomOverSampler.")
        metadata['smote_applied'] = False
    else:
        # categorical feature indices for SMOTENC: all features except Age (index of 'Age' in FEATURE_COLUMNS)
        categorical_indices = [i for i, c in enumerate(FEATURE_COLUMNS) if c != 'Age']
        try:
            sm = SMOTENC(categorical_features=categorical_indices, random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print(f"Applied SMOTENC with k_neighbors={k_neighbors}. New training shape: {X_res.shape}")
            metadata['smote_applied'] = True
            metadata['smote_k_neighbors'] = int(k_neighbors)
            return X_res, y_res
        except Exception as e:
            print("SMOTENC failed with error:", e)
            print("Falling back to RandomOverSampler (safe duplication strategy).")

    # FALLBACK: RandomOverSampler to bring minority classes up to a moderate target_count
    try:
        # target_count: median of counts, but at least 3
        target_count = max(3, median_count)
        sampling_strategy = {}
        for cls, cnt in train_counts.items():
            if cnt < target_count:
                sampling_strategy[int(cls)] = int(target_count)
        if not sampling_strategy:
            print("No classes need oversampling (already above target). Skipping RandomOverSampler.")
            metadata['smote_applied'] = False
            return X_train, y_train

        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        print(f"Applied RandomOverSampler to reach target_count={target_count}. New training shape: {X_res.shape}")
        metadata['smote_applied'] = True
        metadata['smote_fallback'] = 'RandomOverSampler'
        metadata['smote_target_count'] = int(target_count)
        return X_res, y_res
    except Exception as e:
        print("RandomOverSampler fallback failed:", e)
        metadata['smote_applied'] = False
        return X_train, y_train

def train_and_save(X, y, encoders, le_target, df_encoded, metadata):
    # train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    except Exception as e:
        print("Stratified split failed; falling back to non-stratified:", e)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=None)

    # Apply SMOTE/oversampling with fallback if requested
    if USE_SMOTE:
        X_train, y_train = apply_resampling_with_fallback(X_train, y_train, metadata)
    else:
        metadata['smote_applied'] = False

    # compute per-class counts in training set (after resampling)
    train_counts = Counter(y_train.tolist())
    positive_counts = [c for c in train_counts.values() if c > 0]
    min_count = min(positive_counts) if positive_counts else 0
    print("Post-resampling min training count:", min_count)

    # base classifier
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not found. Please install it with 'pip install xgboost'")
    
    print("Using XGBoost classifier.")
    base_clf = XGBClassifier(
        objective='multi:softprob',  # for multiclass classification
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,     # avoids a deprecation warning
        eval_metric='mlogloss'       # common metric for multiclass problems
    )
    metadata['model_type'] = 'XGBoost'


    # Decide calibration strategy:
    if USE_CALIBRATION:
        if min_count >= 3:
            cv_used = 3
            clf_to_fit = CalibratedClassifierCV(base_clf, cv=cv_used)
            print(f"Calibration: using CalibratedClassifierCV(cv={cv_used})")
        elif min_count == 2:
            cv_used = 2
            clf_to_fit = CalibratedClassifierCV(base_clf, cv=cv_used)
            print(f"Calibration: using CalibratedClassifierCV(cv={cv_used}) (limited by class counts)")
        else:
            clf_to_fit = base_clf
            print("Not enough examples for calibration (min_count < 2). Skipping calibration.")
    else:
        clf_to_fit = base_clf
        print("Calibration disabled by config; using base classifier.")

    # Fit classifier
    clf_to_fit.fit(X_train, y_train)

    # Evaluate
    y_pred = clf_to_fit.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy: %.4f" % acc)

    # top-k if probabilities available
    try:
        y_proba = clf_to_fit.predict_proba(X_test)
        if y_proba.shape[1] >= 3:
            try:
                top3 = top_k_accuracy_score(y_test, y_proba, k=3)
                print("Top-3 accuracy: %.4f" % top3)
                metadata['top3_accuracy'] = float(top3)
            except Exception:
                pass
    except Exception:
        y_proba = None

    report_text, labels_present, target_names = safe_classification_report(y_test, y_pred, le_target)
    print(report_text)
    metadata['num_classes_encoder'] = len(le_target.classes_)
    metadata['num_classes_in_test'] = int(len(labels_present))
    metadata['accuracy'] = float(acc)

    # Save artifacts
    joblib.dump(clf_to_fit, os.path.join(MODELS_DIR, 'model.pkl'))
    with open(os.path.join(MODELS_DIR, 'encoders.json'), 'w', encoding='utf-8') as f:
        json.dump(encoders, f, indent=2, ensure_ascii=False)
    with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump({'features': FEATURE_COLUMNS}, f, indent=2)
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    df_for_knn = df_encoded.copy()
    df_for_knn['Disease_lbl'] = df_for_knn['Disease_lbl'].astype(int)
    joblib.dump(df_for_knn, os.path.join(MODELS_DIR, 'train_data.pkl'))

    print("Saved model and artifacts to models/")

def main():
    warnings.filterwarnings("ignore")
    df = load_data(DATA_PATH)
    df = basic_sanity(df)
    metadata = {'original_num_rows': int(df.shape[0])}

    if USE_GROUPING:
        df = map_to_groups(df)
        metadata['grouping_used'] = True
    else:
        metadata['grouping_used'] = False

    if USE_TOP_N:
        df, top_list, counts = keep_top_n_classes(df, n=TOP_N)
        metadata['kept_top_n'] = int(TOP_N)
        metadata['top_n_list'] = top_list
    else:
        metadata['kept_top_n'] = None

    df, rare_meta = handle_rare_classes(df, min_count=2)
    metadata.update(rare_meta)

    if df.shape[0] < 20:
        print("WARNING: Very few samples remain after cleaning")

    X, y, encoders, le_target, df_encoded = encode_features_and_target(df)
    train_and_save(X, y, encoders, le_target, df_encoded, metadata)

if __name__ == '__main__':
    main()