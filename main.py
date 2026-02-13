import pandas as pd

# Load your specific dataset
df = pd.read_csv('student_exam_scores.csv') 

# See the first few rows
print(df.head())

# Check for missing values and data types
print(df.info())

# Summary statistics
print(df.describe())