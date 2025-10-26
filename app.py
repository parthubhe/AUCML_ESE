"""
app.py - Flask web app with improved prediction, SHAP explainability, KNN retrieval, admin page,
and Ollama LLM diagnostics integration.

Requires:
 - Ollama running locally (default: http://localhost:11434)
 - Model artifacts produced by train_model.py in models/
"""
import os, json, joblib, warnings, requests, textwrap
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
import markdown

# shap optional
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'dev-key-change-this'

MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
ENCODERS_PATH = os.path.join(MODELS_DIR, 'encoders.json')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_columns.json')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')
TRAIN_DATA_PATH = os.path.join(MODELS_DIR, 'train_data.pkl')

# Ollama config
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))  # seconds

FORM_FIELDS = [
    ('Fever', 'Fever'),
    ('Cough', 'Cough'),
    ('Fatigue', 'Fatigue'),
    ('Difficulty Breathing', 'DB'),
    ('Age', 'Age'),
    ('Gender', 'Gender'),
    ('Blood Pressure', 'BP'),
    ('Cholesterol Level', 'CL'),
]

# ----------------- Load artifacts -----------------
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None, None, None, None
    model = joblib.load(MODEL_PATH)
    with open(ENCODERS_PATH, 'r', encoding='utf-8') as f:
        encoders_raw = json.load(f)
        encoders = {k: {'classes': list(v['classes'])} for k, v in encoders_raw.items()}
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        feat = json.load(f)
    meta = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    train_df = None
    if os.path.exists(TRAIN_DATA_PATH):
        train_df = joblib.load(TRAIN_DATA_PATH)
    return model, encoders, feat, meta, train_df


model, encoders, feat, meta, train_df = load_artifacts()

# ----------------- Nearest Neighbors -----------------
nn = None
if train_df is not None and feat:
    try:
        X_knn = train_df[feat['features']].values
        nn = NearestNeighbors(n_neighbors=4, metric='euclidean').fit(X_knn)
    except Exception as e:
        print("KNN initialization failed:", e)
        nn = None

# ----------------- SHAP Explainer -----------------
explainer = None
def get_explainer(m):
    global explainer
    if explainer is None:
        try:
            if SHAP_AVAILABLE:
                explainer = shap.TreeExplainer(m)
            else:
                explainer = None
        except Exception:
            explainer = None
    return explainer

# ----------------- Ollama Integration -----------------
def call_ollama_with_context(prompt_text, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT):
    """
    Send prompt to local Ollama API and return the text response.
    Works with /api/generate (non-streamed).
    """
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", resp.text)
    except Exception as e:
        return f"[LLM call failed: {e}]"

# ----------------- Routes -----------------

@app.route('/', methods=['GET'])
def home():
    """Renders the new landing page."""
    return render_template('home.html')

@app.route('/form', methods=['GET'])
def index():
    """Renders the prediction form page."""
    choices = {}
    for label, col in FORM_FIELDS:
        if encoders and col in encoders:
            choices[col] = encoders[col]['classes']
        else:
            if col == 'Age':
                choices[col] = None
            elif col in ['Gender']:
                choices[col] = ['Male', 'Female']
            else:
                choices[col] = ['Yes', 'No']
    return render_template('index.html', form_fields=FORM_FIELDS, choices=choices, meta=meta)

@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    for label, col in FORM_FIELDS:
        val = request.form.get(col)
        if val is None or val == '':
            flash(f"Please provide a value for {label}")
            return redirect(url_for('index'))
        data[col] = val
    try:
        age = float(data['Age'])
    except:
        flash("Age must be numeric")
        return redirect(url_for('index'))

    if not model or not encoders or not feat:
        flash("Model artifacts not found. Please run training script.")
        return redirect(url_for('index'))

    feature_cols = feat['features']
    x_row = []
    for col in feature_cols:
        if col == 'Age':
            x_row.append(age)
        else:
            classes = encoders[col]['classes']
            try:
                idx = classes.index(data[col])
            except ValueError:
                found = None
                for i, c in enumerate(classes):
                    if str(c).lower() == str(data[col]).lower():
                        found = i
                        break
                idx = found if found is not None else 0
            x_row.append(idx)
    X_input = pd.DataFrame([x_row], columns=feature_cols)

    # predict probabilities
    try:
        probs = model.predict_proba(X_input)[0]
    except Exception:
        probs = model.predict_proba(X_input.values)[0]

    classes = encoders['Disease']['classes']
    pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    topk = pairs[:6]

    # SHAP explanation (if available)
    shap_bar = None
    expl = get_explainer(model)
    if expl is not None:
        try:
            sv = expl.shap_values(X_input)
            class_idx = int(np.argmax(probs))
            if isinstance(sv, list):
                svc = sv[class_idx][0]
            else:
                svc = sv[0]
            shap_pairs = list(zip(feature_cols, svc.tolist()))
            shap_pairs = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
            shap_bar = [{'feature': f, 'value': float(v)} for f, v in shap_pairs]
        except Exception:
            shap_bar = None

    # Similar-case retrieval
    similar_cases = []
    if nn is not None and train_df is not None:
        try:
            distances, idxs = nn.kneighbors(X_input.values, n_neighbors=4)
            idxs = idxs[0].tolist()
            for i in idxs[1:4]:
                row = train_df.iloc[i]
                similar_cases.append({
                    'Disease': row.get('Disease', 'Unknown'),
                    'Age': int(row.get('Age', 0)),
                    'Fever': row.get('Fever', '?'),
                    'Cough': row.get('Cough', '?'),
                    'Fatigue': row.get('Fatigue', '?')
                })
        except Exception:
            similar_cases = []

    chart = {'labels': classes, 'probs': probs.tolist()}

    # LLM diagnostic summary
    try:
        input_summary_lines = []
        input_summary_lines.append("Patient features:")
        for label, col in FORM_FIELDS:
            input_summary_lines.append(f"- {label}: {data[col]}")
        input_summary_lines.append("")
        input_summary_lines.append("Top model predictions (disease : probability):")
        for cls, p in pairs[:8]:
            input_summary_lines.append(f"- {cls} : {p*100:.1f}%")
        input_summary_lines.append("")
        input_summary_lines.append(f"Model metadata: {json.dumps(meta, ensure_ascii=False)}")
        llm_context = "\n".join(input_summary_lines)

        prompt = textwrap.dedent(f"""
You are a clinical-assistant style expert. Use the context below to perform a concise diagnostic assessment and provide a clear, actionable plan for next steps. The plan should include:
1) Brief possible diagnoses or areas of concern based on probabilities and features (short bullets).
2) Recommended immediate tests or examinations (non-invasive, explain why).
3) Lifestyle or monitoring recommendations (non-prescriptive, high-level).
4) Suggested medical specialties or referrals (if applicable).

**Important:** Provide high-level recommendations only — do NOT give prescriptive treatment or dosing. Include a short disclaimer at the end: this is informational, not medical advice.

CONTEXT:
{llm_context}

Provide output in clear bullet points and short paragraphs.
""")
        llm_response = call_ollama_with_context(prompt)
    except Exception as e:
        llm_response = f"[Failed to build LLM prompt: {e}]"
        
    # Convert LLM Markdown output to HTML for proper rendering
    html_output = markdown.markdown(llm_response)

    return render_template(
        'result.html',
        topk=topk,
        chart=chart,
        shap_bar=shap_bar,
        similar_cases=similar_cases,
        meta=meta,
        llm_output=html_output
    )

@app.route('/admin', methods=['GET'])
def admin():
    counts = {}
    if train_df is not None:
        counts = train_df['Disease'].value_counts().to_dict()
    return render_template('admin.html', meta=meta, counts=counts)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    app.run(debug=True)