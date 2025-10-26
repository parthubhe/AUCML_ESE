# Age-related Condition Predictor (Flask + ML)

This project converts the provided notebook into a Flask web app.  
Main features:
- Training script `train_model.py` that trains a RandomForest model to predict **Disease** from symptoms and demographics.
- `app.py` serves a simple web UI (Bootstrap + Chart.js) that collects user inputs and displays predicted disease probabilities.
- The UI shows top-K diseases and a horizontal bar chart of the full distribution.

How to run:
1. Place your CSV dataset at `data/Disease_symptom_and_patient_profile_dataset.csv`.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train_model.py`
4. Run the server: `flask run` (or `python app.py`)

Notes about novelty (to help with CA3 grade):
- The delivered project upgrades a simple binary "Results" model into a multi-class disease probability predictor.
- You can add SHAP explanations, age-group specific risk dashboards, or a nearest-neighbor example page to increase novelty.
