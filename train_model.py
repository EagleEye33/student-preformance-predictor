import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_system(file_path):
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found in the 'student' folder.")
        return

    # 2. Preprocessing
    # We drop 'student_id' because it's just a label, not a predictor
    X = df.drop(columns=['student_id', 'exam_score'])
    y = df['exam_score']

    # 3. Model Initialization
    model = LinearRegression()

    # 4. Cross-Validation (The Reliability Check)
    # We use 5 folds to ensure the model performs well on all parts of your 200 records
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kf, scoring='r2')

    print("\n" + "="*30)
    print(" MACHINE LEARNING DIAGNOSTICS")
    print("="*30)
    print(f"Average R2 Score: {np.mean(cv_results):.3f}")
    print(f"Model Stability (Std Dev): {np.std(cv_results):.3f}")
    print("="*30 + "\n")

    # 5. Final Training
    # We train on the FULL 200 records now that we know it's stable
    model.fit(X, y)

    # 6. Save Model and Column names (to ensure input order matches)
    model_data = {
        'model': model,
        'features': list(X.columns)
    }
    joblib.dump(model_data, 'student_model.pkl')
    print("✅ Model trained and saved as 'student_model.pkl'")

if __name__ == "__main__":
    train_system('student_exam_scores.csv')