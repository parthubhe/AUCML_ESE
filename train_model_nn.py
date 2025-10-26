"""
train_model.py - Robust training pipeline using a Keras Neural Network.

This version:
- uses a feedforward neural network (built with TensorFlow/Keras)
- adds StandardScaler for neural network data preprocessing
- uses SciKerasClassifier to make the Keras model compatible with scikit-learn
- retains safe SMOTE/SMOTENC and calibration logic
- saves artifacts required by the Flask app
"""
import os, json, joblib, warnings
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# --- NEURAL NETWORK IMPORTS ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from scikeras.wrappers import KerasClassifier
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False


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
    'Hypertension': 'Cardio', 'Coronary Artery Disease': 'Cardio',
    'Asthma': 'Respiratory', 'Bronchitis': 'Respiratory', 'Pneumonia': 'Respiratory',
    'Diabetes': 'Metabolic', 'Hyperthyroidism': 'Endocrine', 'Hypothyroidism': 'Endocrine',
    'Influenza': 'Infectious', 'Measles': 'Infectious', 'Rubella': 'Infectious', 'Malaria': 'Infectious'
}

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}.")
    df = pd.read_csv(path)
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
    train_counts = Counter(y_train.tolist())
    counts_list = np.array(list(train_counts.values()))
    min_count = int(counts_list.min()) if counts_list.size else 0
    median_count = int(np.median(counts_list)) if counts_list.size else 0
    print("Training class counts (sample):", dict(list(train_counts.items())[:12]))
    print("Minimum training examples for any class:", min_count)
    if not IMBLEARN_AVAILABLE:
        print("imblearn not available; install 'imbalanced-learn' to enable SMOTE. Skipping resampling.")
        metadata['smote_applied'] = False
        return X_train.to_numpy(), y_train.to_numpy()
    k_neighbors = max(1, min(5, min_count - 1)) if min_count >= 2 else None
    if k_neighbors is None:
        print("Not enough samples for SMOTE (min_count < 2). Falling back to RandomOverSampler.")
        metadata['smote_applied'] = False
    else:
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
    try:
        target_count = max(3, median_count)
        sampling_strategy = {}
        for cls, cnt in train_counts.items():
            if cnt < target_count:
                sampling_strategy[int(cls)] = int(target_count)
        if not sampling_strategy:
            print("No classes need oversampling (already above target). Skipping RandomOverSampler.")
            metadata['smote_applied'] = False
            return X_train.to_numpy(), y_train.to_numpy()
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
        return X_train.to_numpy(), y_train.to_numpy()

def create_nn_model(n_features_in_, n_classes_):
    """Dynamically creates a Keras model."""
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features_in_,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes_, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- THE KEY FIX IS HERE ---
if NN_AVAILABLE:
    class RobustKerasClassifier(KerasClassifier):
        """
        A subclass of KerasClassifier that explicitly sets the _estimator_type
        as a class attribute. This ensures that scikit-learn's `clone` operation
        preserves the type, preventing 'Got a regressor' errors in wrappers
        like CalibratedClassifierCV.
        """
        _estimator_type = "classifier"

def train_and_save(X, y, encoders, le_target, df_encoded, metadata):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    except Exception as e:
        print("Stratified split failed; falling back to non-stratified:", e)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=None)

    if USE_SMOTE:
        X_train_resampled, y_train_resampled = apply_resampling_with_fallback(X_train, y_train, metadata)
    else:
        X_train_resampled, y_train_resampled = X_train.to_numpy(), y_train.to_numpy()
        metadata['smote_applied'] = False

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test.to_numpy())

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("Saved StandardScaler to models/scaler.pkl")

    train_counts = Counter(y_train_resampled.tolist())
    min_count = min(train_counts.values()) if train_counts else 0
    print("Post-resampling min training count:", min_count)

    if not NN_AVAILABLE:
        raise ImportError("TensorFlow/Keras/SciKeras not found. Please run 'pip install tensorflow scikeras'")

    print("Using Robust Keras Neural Network classifier.")

    def make_model():
        return create_nn_model(
            n_features_in_=X_train_final.shape[1],
            n_classes_=len(le_target.classes_)
        )

    # Use the new robust subclass
    base_clf = RobustKerasClassifier(
        model=make_model,
        epochs=50,
        batch_size=32,
        verbose=0,
        random_state=RANDOM_STATE,
        loss="sparse_categorical_crossentropy"
    )
    metadata['model_type'] = 'NeuralNetwork'

    if USE_CALIBRATION:
        if min_count >= 3:
            cv_used = 3
            # This should now work correctly without any other changes
            clf_to_fit = CalibratedClassifierCV(base_clf, cv=cv_used)
            print(f"Calibration: using CalibratedClassifierCV(cv={cv_used})")
        else:
            clf_to_fit = base_clf
            print("Not enough examples for reliable calibration (min_count < 3). Skipping calibration.")
    else:
        clf_to_fit = base_clf
        print("Calibration disabled by config; using base classifier.")

    print("Fitting the final model...")
    clf_to_fit.fit(X_train_final, y_train_resampled)
    print("Model fitting complete.")

    y_pred = clf_to_fit.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy: %.4f" % acc)

    try:
        y_proba = clf_to_fit.predict_proba(X_test_scaled)
        if y_proba.shape[1] >= 3:
            top3 = top_k_accuracy_score(y_test, y_proba, k=3)
            print("Top-3 accuracy: %.4f" % top3)
            metadata['top3_accuracy'] = float(top3)
    except Exception as e:
        print(f"Top-3 accuracy could not be calculated: {e}")

    report_text, labels_present, target_names = safe_classification_report(y_test, y_pred, le_target)
    print(report_text)
    metadata.update({
        'num_classes_encoder': len(le_target.classes_),
        'num_classes_in_test': int(len(labels_present)),
        'accuracy': float(acc)
    })

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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    if NN_AVAILABLE:
        tf.random.set_seed(RANDOM_STATE)
    
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