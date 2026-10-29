from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the bundled model and threshold
model_bundle = joblib.load('Heart_disease_model.pkl')
pipeline = model_bundle['pipeline']
threshold = model_bundle['threshold']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    probability_score = None
    alert_type = None

    if request.method == 'POST':
        try:
            # Extract form inputs matching the exact dataset column names
            form_data = {
                'age': [int(request.form.get('age'))],
                'sex': [int(request.form.get('sex'))],
                'chest pain type': [int(request.form.get('chest_pain_type'))],
                'resting blood pressure': [int(request.form.get('resting_blood_pressure'))],
                'serum cholestoral': [int(request.form.get('serum_cholestoral'))],
                'fasting blood sugar': [int(request.form.get('fasting_blood_sugar'))],
                'resting electrocardiographic results': [int(request.form.get('resting_electrocardiographic_results'))],
                'max heart rate': [int(request.form.get('max_heart_rate'))],
                'exercise induced angina': [int(request.form.get('exercise_induced_angina'))],
                'oldpeak': [float(request.form.get('oldpeak'))],
                'ST segment': [int(request.form.get('st_segment'))],
                'major vessels': [int(request.form.get('major_vessels'))],
                'thal': [int(request.form.get('thal'))]
            }

            # Convert input to DataFrame
            input_df = pd.DataFrame(form_data)

            # Get predicted probability for Disease (Class 1)
            prob_disease = pipeline.predict_proba(input_df)[0, 1]
            probability_score = round(prob_disease * 100, 2)

            # Apply the tuned threshold
            if prob_disease >= threshold:
                prediction_text = "High Risk: Heart Disease Detected"
                alert_type = "danger"
            else:
                prediction_text = "Low Risk: No Heart Disease Detected"
                alert_type = "success"

        except Exception as e:
            prediction_text = f"Error in processing input: {str(e)}"
            alert_type = "warning"

    return render_template(
        'index.html',
        prediction=prediction_text,
        probability=probability_score,
        alert_type=alert_type
    )

if __name__ == '__main__':
    app.run(debug=True)