import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import openai
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bleak import BleakScanner, BleakClient
import asyncio

# 1️⃣ Load Past Health Records
def load_health_records():
    health_data = pd.read_csv("C:\Users\adithi\Desktop\healthcare_dataset.csv")
    health_data["Date"] = pd.to_datetime(health_data["Date"])
    return health_data

# 2️⃣ Connect to Smartphone via Bluetooth & Fetch Real-Time Data
async def fetch_smartphone_data():
    devices = await BleakScanner.discover()
    target_device = None 
    
    for device in devices:
        print(f"Found Device: {device.name}, Address: {device.address}")
        if device.name == "Kireeti":
            target_device =  device.address
            break
        
    if not target_device:
        print("❌ Smartphone 'kireeti' not found. Ensure Bluetooth is enabled.")
        return None
    
    # Simulated real-time health data from smartphone sensors
    real_time_data = {
        "Heart Rate (bpm)": np.random.randint(60, 120),
        "Blood Sugar (mmol/L)": np.random.uniform(4.5, 8.5),
        "Steps": np.random.randint(2000, 15000),
        "Sleep Hours": np.random.uniform(4, 9)
    }
    return real_time_data

# 3️⃣ Train AI Model to Predict Health Risks
def train_health_model(health_data):
    features = ["Heart Rate (bpm)", "Blood Sugar (mmol/L)", "Steps", "Sleep Hours"]
    target = "Risk Level"
    
    health_data[target] = health_data[target].map({"Low": 0, "Medium": 1, "High": 2})
    
    X = health_data[features]
    y = health_data[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, scaler

# 4️⃣ Detect Health Anomalies
def detect_anomalies(real_time_data, past_data):
    alerts = []
    
    if real_time_data["Heart Rate (bpm)"] > 110:
        alerts.append("⚠️ High heart rate detected! Consider consulting a doctor.")
    if real_time_data["Blood Sugar (mmol/L)"] > 7.5:
        alerts.append("⚠️ Elevated blood sugar levels detected.")
    avg_steps = past_data["Steps"].mean()
    if real_time_data["Steps"] < avg_steps * 0.5:
        alerts.append("⚠️ Your activity levels are much lower than usual.")
    
    return alerts

# 5️⃣ Use OpenAI to Generate Lifestyle Recommendations
def get_health_recommendations(health_issues):
    api_key = "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"
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    A person has the following health conditions: {health_issues}.
    Suggest practical, easy-to-follow lifestyle habit changes to improve their health.
    """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a health and wellness expert."},
                  {"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# ✅ Run the Full Health AI Analysis
async def run_health_analysis():
    file_path = "C:\Users\Kiree\OneDrive\Desktop\health_data.csv"
    health_data = load_health_records(file_path)
    
    print("⏳ Connecting to Smartphone (Kireeti)...")
    real_time_data = await fetch_smartphone_data()
    if not real_time_data:
        print("❌ Unable to fetch data from smartphone.")
        return
    
    print(f"📡 Real-Time Smartphone Data: {real_time_data}")
    
    
    model, scaler = train_health_model(health_data)
    
    input_data = np.array([[real_time_data["Heart Rate (bpm)"], real_time_data["Blood Sugar (mmol/L)"],
                            real_time_data["Steps"], real_time_data["Sleep Hours"]]])
    input_scaled = scaler.transform(input_data)
    risk_prediction = model.predict(input_scaled)
    
    risk_levels = {0: "Low", 1: "Medium", 2: "High"}
    predicted_risk = risk_levels[risk_prediction[0]]
    
    print(f"\n🔮 Predicted Health Risk Level: {predicted_risk}")
    
    alerts = detect_anomalies(real_time_data, health_data)
    if alerts:
        print("\n⚠️ Health Alerts:")
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("\n✅ No health alerts detected.")
    
    health_conditions = "hypertension, obesity, sleep apnea"
    lifestyle_tips = get_health_recommendations(health_conditions)
    
    print("\n💡 AI-Powered Lifestyle Recommendations:")
    print(lifestyle_tips)

asyncio.run(run_health_analysis())
