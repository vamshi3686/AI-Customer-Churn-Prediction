from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standar_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# IMPORTANT: same order as training
feature_columns = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'sim_column_BSNL', 'sim_column_Idea', 'sim_column_Reliancejio',
    'Contract_od'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            input_data = []

            for col in feature_columns:
                value = request.form.get(col)

                # default 0 if not provided
                if value is None or value == "":
                    input_data.append(0)
                else:
                    input_data.append(float(value))

            # Convert to numpy
            features_array = np.array([input_data])

            # Scale
            features_scaled = scaler.transform(features_array)

            # Predict
            pred = model.predict(features_scaled)[0]

            if pred == 1:
                prediction = "Customer will Churn ❌"
            else:
                prediction = "Customer will Stay ✅"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)