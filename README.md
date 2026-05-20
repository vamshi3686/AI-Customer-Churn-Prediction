# 📊 AI-Powered Customer Retention Prediction System

AI-Powered Customer Retention Prediction System is a Machine Learning-based web application that predicts customer churn using customer behavioral and subscription data.  
The application helps businesses identify customers who are likely to leave their services, enabling proactive retention strategies and better decision-making.

This project was built to explore practical applications of:
- Machine Learning
- Customer Analytics
- Data Preprocessing Pipelines
- Flask Web Deployment
- Predictive Business Intelligence

---

## 🚀 Features

- Customer churn prediction using AI
- Real-time prediction through web interface
- Data preprocessing automation
- Missing value handling
- Feature scaling and encoding
- Multiple Machine Learning model evaluation
- Pickle-based model deployment
- Logging system for monitoring and debugging
- Responsive Flask-based UI

---

## 🧠 Technologies Used

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

### Model Deployment
- Pickle

### Logging & Utilities
- Python Logging Module

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Powered-Customer-Retention-Prediction-System.git

cd AI-Powered-Customer-Retention-Prediction-System
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Machine Learning Workflow

## Project Pipeline

1. Data Collection  
2. Missing Data Handling  
3. Label Encoding  
4. Feature Scaling  
5. Feature Engineering  
6. Model Training  
7. Model Evaluation  
8. Churn Prediction  
9. Flask Deployment  

---

# ▶️ Run Application

```bash
python main.py
```

Then

```bash
python app.py
```

---

# 🌐 Open in Browser

```bash
http://127.0.0.1:5000
```

---

# 📂 Project Structure

```bash
AI-Powered-Customer-Retention-Prediction-System/
│
├── logs/
│   ├── all_models.log
│   ├── f_m.log
│   ├── feature_scaling.log
│   ├── label_encoder.log
│   ├── main.log
│   ├── missing_data.log
│   └── var_out.log
│
├── templates/
│   └── index.html
│
├── all_models.py
├── app.py
├── f_m.py
├── feature_scaling.py
├── label_encoder.py
├── logging_code.py
├── main.py
├── missing_data.py
├── var_out.py
│
├── Model.pkl
├── standar_scaler.pkl
├── roc_curve.png
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

# 📊 Dataset Information

Dataset Used:

## Telco Customer Churn Dataset

The dataset contains:
- Customer demographics
- Service subscriptions
- Payment methods
- Contract details
- Monthly charges
- Tenure information
- Churn status

---

# 📸 Application Preview

## 🔹 Home Page

<img width="100%" alt="Home Page" src="home.png">

The home page allows users to:
- Enter customer details
- Submit prediction requests
- Access churn analysis interface
- View prediction results instantly

---

## 🔹 Prediction Result

<img width="100%" alt="Prediction Result" src="prediction.png">

After prediction, the system displays:
- Churn prediction result
- Customer retention status
- Model-based prediction analysis
- Processed customer insights

---

# 📈 Model Performance

The project evaluates multiple Machine Learning algorithms such as:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

ROC Curve visualization is included for model performance comparison.

---

# 📸 How It Works

1. User enters customer information  
2. Input data gets preprocessed  
3. Label encoding and scaling are applied  
4. Trained ML model predicts churn probability  
5. Prediction result is displayed on the dashboard  

---

# 🎯 Future Improvements

- Deep Learning integration
- Cloud deployment using AWS/Render
- Interactive analytics dashboard
- REST API support
- Docker containerization
- Real-time customer monitoring
- Advanced feature engineering

---

# 👨‍💻 Developer

**Vamshi Krishna**

AI & Data Science Developer specializing in:
- Machine Learning
- Predictive Analytics
- Flask Development
- Intelligent AI Systems

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
