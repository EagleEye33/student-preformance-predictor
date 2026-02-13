import joblib
import numpy as np

def run_console_app():
    # Load the saved model and feature list
    try:
        data = joblib.load('student_model.pkl')
        model = data['model']
        features = data['features']
    except FileNotFoundError:
        print("❌ Error: No trained model found. Run 'train_model.py' first.")
        return

    print("\n" + "*"*40)
    print(" STUDENT PERFORMANCE PREDICTOR (V1.0)")
    print("*"*40)

    while True:
        try:
            print("\nPlease enter student details:")
            
            # Dynamic input based on the features we used in training
            user_inputs = []
            for feature in features:
                val = float(input(f" -> {feature.replace('_', ' ').title()}: "))
                user_inputs.append(val)

            # Convert to numpy array for prediction
            input_array = np.array([user_inputs])
            
            # Predict
            prediction = model.predict(input_array)[0]
            
            # Final Output Logic
            print("-" * 20)
            print(f"PREDICTED EXAM SCORE: {prediction:.2f}%")
            
            if prediction >= 75:
                print("Feedback: Excellent! Likely to be a Top Performer.")
            elif prediction >= 40:
                print("Feedback: On track to pass. Keep up the consistency.")
            else:
                print("Feedback: WARNING - High risk of failure. Intervention needed.")
            print("-" * 20)

        except ValueError:
            print("⚠️ Invalid input. Please enter numbers only.")
        
        cont = input("\nPredict for another student? (y/n): ").lower()
        if cont != 'y':
            print("Exiting... Good luck with your studies!")
            break

if __name__ == "__main__":
    run_console_app()