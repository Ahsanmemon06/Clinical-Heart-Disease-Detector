# 🫀 Heart Disease Risk Prediction Web App

An end-to-end clinical machine learning web application built with **Flask**, **Scikit-Learn**, and **Joblib** to assess coronary heart disease risk. The pipeline uses an optimized Support Vector Machine (RBF Kernel) with custom decision threshold calibration to maximize clinical sensitivity (Recall) and minimize missed diagnoses (False Negatives).

---

## 📌 Project Overview

In healthcare diagnostics, particularly cardiology, the cost of a **False Negative** (classifying a cardiac patient as healthy) is significantly higher than that of a **False Positive**. Standard off-the-shelf models operating at a default decision threshold of 0.5 often miss borderline positive cases.

This project:
1. Preprocesses clinical tabular data using an automated `ColumnTransformer` (StandardScaler + OneHotEncoder).
2. Benchmarks multiple classifiers (Logistic Regression, KNN, Decision Trees, Random Forest, XGBoost, and SVM).
3. Tunes an **RBF Support Vector Machine (SVM)** with `class_weight='balanced'` and lowers the decision threshold to **0.28**.
4. Deploys the full pipeline via a responsive, medical-grade Flask web dashboard.

---

## 📊 Model Benchmarking & Comparison

All models were evaluated on an identical test split ($N = 54$; Class 0: 33, Class 1: 21):

| Model | Accuracy | Class 1 Precision | Class 1 Recall (Sensitivity) | Class 1 F1-Score | False Negatives (Missed) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default Logistic Regression** | 87% | 100% | 67% | 0.80 | 7 |
| **KNN (k=5)** | 85% | 93% | 67% | 0.78 | 7 |
| **Decision Tree (depth=4)** | 83% | 93% | 62% | 0.74 | 8 |
| **Random Forest (Default)** | 81% | 87% | 62% | 0.72 | 8 |
| **Tuned Random Forest** | 87% | 94% | 71% | 0.81 | 6 |
| **XGBoost** | 87% | 94% | 71% | 0.81 | 6 |
| **Default SVM (threshold=0.5)** | 87% | 89% | 76% | 0.82 | 5 |
| **🎯 Tuned SVM (threshold=0.28)** | **80%** | **67%** | **85.7% (18/21)** | **0.75** | **3 (Lowest)** |

### 🔍 Key Clinical Takeaway
By calibrating the decision threshold to **0.28**, the model captured **18 out of 21 diseased patients** (dropping missed diagnoses from 5 to 3). In clinical screening, trading marginal precision for higher sensitivity is vital to patient safety.

---

## 🛠️ Tech Stack

- **Core:** Python 3.10+
- **Machine Learning:** Scikit-Learn, XGBoost, NumPy, Pandas
- **Model Serialization:** Joblib
- **Web Framework:** Flask, Jinja2
- **Frontend:** HTML5, CSS3, FontAwesome

---

## 📁 Project Directory Structure

```text
├── dataset_heart.csv            # Training & evaluation dataset
├── model_development.ipynb      # EDA, feature scaling, model benchmarking
├── Heart_disease_model.pkl      # Bundled Scikit-Learn Pipeline + optimal threshold
├── app.py                       # Flask application routing & inference logic
├── requirements.txt             # Project dependencies
├── .gitignore                   # Ignored files
├── templates/
│   └── index.html               # Responsive clinical interface
└── static/
    ├── css/
    │   └── style.css            # Custom medical UI styling
    └── images/
        └── heart.jpg            # Anatomical illustration

        
⚙️ Installation & Local Setup
1. Clone the repository:
git clone [https://github.com/Ahsanmemon06/Heart-Disease-Prediction-Flask.git](https://github.com/Ahsanmemon06/Heart-Disease-Prediction-Flask.git)
cd Heart-Disease-Prediction-Flask

2. Create and activate a virtual environment:
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

3. Install required packages:
Bash
pip install -r requirements.txt

4. Launch the web application:
Bash
python app.py

5. Open in browser:
Navigate to http://127.0.0.1:5000


🧪 Clinical Features & Form Inputs
The application processes 13 clinical biomarkers:

Age (Years)

Sex (Male: 1, Female: 0)

Chest Pain Type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic)

Resting Blood Pressure (mm Hg)

Serum Cholesterol (mg/dl)

Fasting Blood Sugar (> 120 mg/dl: 1, ≤ 120 mg/dl: 0)

Resting ECG (0: Normal, 1: ST-T Abnormality, 2: LVH)

Max Heart Rate Achieved (bpm)

Exercise Induced Angina (Yes: 1, No: 0)

Oldpeak (ST depression induced by exercise)

ST Segment Slope (1: Upsloping, 2: Flat, 3: Downsloping)

Major Vessels (Number of vessels colored by fluoroscopy, 0–3)

Thalassemia (3: Normal, 6: Fixed defect, 7: Reversible defect)
