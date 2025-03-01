import pandas as pd

# Load dataset from the fitbit
file_path = "fitbit_merged.csv"
df = pd.read_csv(file_path)

# Data Cleaning - Remove Unnecessary Columns
columns_to_remove = ["Id", "ActivityDate", "Calories", "LoggedActivitiesDistance"]
df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

# Calculate Weekly Means
# Group every 7 rows together (weekly aggregation) and compute the mean
df_weekly_means = df_cleaned.groupby(df_cleaned.index // 7).mean()

# Define Healthy Reference Values for Each Metric
healthy_ranges = {
    "TotalSteps": 7000,  # Recommended daily step count (WHO)
    "VeryActiveMinutes": 30,  # Minimum recommended exercise time per day
    "FairlyActiveMinutes": 30,
    "LightlyActiveMinutes": 150,  # Weekly recommended light exercise
    "SedentaryMinutes": 600,  # Should not exceed this daily
    "Bp": 120,  # Healthy systolic blood pressure
    "heartrate": 60,  # Average resting heart rate
    "temperature": 37.0,  # Normal body temperature
    "oxygen": 95,  # Minimum acceptable blood oxygen level
    "sleep": 7.5,  # Minimum healthy sleep hours per night
    "weight": 58  # Healthy weight for reference (varies per person)
}

# Generate Health Advice Based on Weekly Trends
def generate_advice(df_means, healthy_ranges):
    advice_list = []
    
    for index, row in df_means.iterrows():
        advice = f"**Week {index+1} Health Advice:**\n"
        
        for metric, healthy_value in healthy_ranges.items():
            if metric in row:
                if row[metric] < healthy_value:
                    advice += f"- Your {metric} is below the recommended value ({row[metric]:.2f} vs {healthy_value}). Try to improve this by adjusting your routine. Ask the Chatbot for more information.\n"
                elif row[metric] > healthy_value:
                    advice += f"- Your {metric} is above the recommended value ({row[metric]:.2f} vs {healthy_value}). Consider making changes to balance this. Ask the Chatbot for more information\n"
        
        advice_list.append(advice)
    
    return advice_list

# Generate and Display Health Advice
weekly_health_advice = generate_advice(df_weekly_means, healthy_ranges)
for advice in weekly_health_advice:
    print(advice)
