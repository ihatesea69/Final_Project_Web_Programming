import pandas as pd

# Load the dataset
file_path = './Sleep_Efficiency_raw.csv'
sleep_data = pd.read_csv(file_path)

# Drop the 'ID' column if it's not needed
sleep_data_cleaned = sleep_data.drop(columns=['ID'])

# Convert 'Bedtime' and 'Wakeup time' to datetime format
sleep_data_cleaned['Bedtime'] = pd.to_datetime(sleep_data_cleaned['Bedtime'], errors='coerce')
sleep_data_cleaned['Wakeup time'] = pd.to_datetime(sleep_data_cleaned['Wakeup time'], errors='coerce')

# Fill missing values for numerical columns with the median
for column in ['Awakenings', 'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency']:
    sleep_data_cleaned[column].fillna(sleep_data_cleaned[column].median(), inplace=True)

# Convert categorical variables 'Gender' and 'Smoking status' to numerical codes
sleep_data_cleaned['Gender'] = sleep_data_cleaned['Gender'].map({'Male': 1, 'Female': 0})
sleep_data_cleaned['Smoking status'] = sleep_data_cleaned['Smoking status'].map({'Yes': 1, 'No': 0})

# Save the cleaned dataset
cleaned_file_path = './Sleep_Efficiency_cleaned.csv'
sleep_data_cleaned.to_csv(cleaned_file_path, index=False)

print("Cleaned data saved to:", cleaned_file_path)
