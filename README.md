### Student Performance Prediction System
A robust, console-based Machine Learning application designed to predict student exam scores using linear regression. This project demonstrates a complete ML pipeline: from data preprocessing and K-Fold Cross-Validation to a production-ready interactive interface.

### 🚀 Key Features
1. Automated Preprocessing: Handles data cleaning and feature selection (dropping non-predictive IDs).
2. Model Reliability: Implements 5-Fold Cross-Validation to ensure the model's performance is stable and not a result of overfitting on the 200-record dataset.
3. Modular Architecture: Separate scripts for training (train_model.py) and inference (predict.py), mimicking real-world deployment patterns.Interactive CLI: A user-friendly console interface for real-time predictions with input validation.

### 🛠️ Tech Stack
1. Language: Python 3.x
2. Libraries: scikit-learn (Modeling & Validation), pandas (Data Manipulation), numpy (Numerical Operations), joblib (Model Serialization)

### 📊 Dataset Overview
The model is trained on a dataset (200 records) containing:
1. Features: Hours Studied, Sleep Hours, Attendance %, and Previous Scores.
2. Target: Final Exam Score (Numerical).

### ⚙️ Setup & InstallationClone the repository:
```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
```

### Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### 📂 Project Structure
```bash
student/
├── train_model.py      # Data processing, CV, and model training
├── predict.py          # Interactive console application
├── dataset.csv         # Student performance data
├── requirements.txt    # Project dependencies
└── .gitignore          # Prevents pushing .pkl and venv files
```

### 📈 How to Run
1. Train the Model:This will run the cross-validation diagnostics and save the trained model as student_model.pkl.
```bash
python train_model.py
```

2. Run Predictions:Launch the interactive console to input student data and receive score predictions.
```bash
python predict.py
```

### 🛡️ Model Validation (Reliability)
To ensure the system is "ready to deploy," I utilized K-Fold Cross-Validation (k=5). This process splits the 200 records into 5 different subsets to verify that the R2 score remains consistent across different data slices, providing a more realistic estimate of performance on unseen data.