import pandas as pd
import joblib

# Load the saved model
model_rf = joblib.load('titanic_rf_model.pkl')

# Get input from user with sample guidance
print("🚢 Titanic Survival Prediction")
print("Please enter the following passenger details:\n")

try:
    passenger_id = int(input("Enter Passenger ID [e.g., 892]: "))
    pclass = int(input("Enter Passenger Class (Upper=1, Middle=2, Lower=3) [e.g., 3]: "))
    sex = input("Enter Sex (male/female) [e.g., male]: ").strip().lower()
    age = float(input("Enter Age [e.g., 25]: "))
    sibsp = int(input("Enter Number of Siblings/Spouses (SibSp) [e.g., 0]: "))
    parch = int(input("Enter Number of Parents/Children (Parch) [e.g., 0]: "))
    fare = float(input("Enter Fare [e.g., 32.5]: "))
except ValueError:
    print("\n Invalid input. Please enter correct values.")
    exit()

# Validate inputs
if sex not in ['male', 'female'] or pclass not in [1, 2, 3]:
    print("\n Invalid choice for sex or passenger class.")
    exit()

# Create DataFrame with all expected columns
input_df = pd.DataFrame([{
    'PassengerId': passenger_id,
    'Pclass': pclass,
    'Sex': 1 if sex == 'female' else 0,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare
}])

# Predict and show result
prediction = model_rf.predict(input_df)[0]
print("\n🔍 Prediction Result:", "You Survived!" if prediction == 1 else "You Did Not Survive")






