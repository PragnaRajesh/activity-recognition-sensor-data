import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import RandomForestClassifier

# Load sensor data
df = pd.read_csv('sensor_data_sample.csv')
feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Feature extraction on full dataset for training
window_size = 10
X = pd.DataFrame()

for col in feature_cols:
    X[f'{col}_mean'] = df[col].rolling(window=window_size).mean()
    X[f'{col}_std'] = df[col].rolling(window=window_size).std()
    X[f'{col}_min'] = df[col].rolling(window=window_size).min()
    X[f'{col}_max'] = df[col].rolling(window=window_size).max()

X = X.dropna().reset_index(drop=True)
y = df['activity'][window_size - 1:].reset_index(drop=True)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize Tkinter window
root = tk.Tk()
root.title("Real-Time Activity Prediction")
root.geometry("500x300")

label_title = ttk.Label(root, text="Live Activity Recognition", font=("Arial", 18))
label_title.pack(pady=10)

sensor_frame = ttk.LabelFrame(root, text="Sensor Data")
sensor_frame.pack(pady=10)

sensor_labels = {}
for col in feature_cols:
    label = ttk.Label(sensor_frame, text=f"{col}: --")
    label.pack()
    sensor_labels[col] = label

prediction_label = ttk.Label(root, text="Predicted Activity: --", font=("Arial", 16), foreground="blue")
prediction_label.pack(pady=20)

buffer = []

def update_data(index=0):
    if index < len(df):
        row = df.iloc[index]
        for col in feature_cols:
            sensor_labels[col].config(text=f"{col}: {row[col]:.3f}")
        buffer.append([row[col] for col in feature_cols])

        if len(buffer) >= window_size:
            window_df = pd.DataFrame(buffer[-window_size:], columns=feature_cols)
            features = []
            for col in feature_cols:
                features.extend([
                    window_df[col].mean(),
                    window_df[col].std(),
                    window_df[col].min(),
                    window_df[col].max()
                ])
            prediction = model.predict([features])[0]
            prediction_label.config(text=f"Predicted Activity: {prediction}")

        root.after(500, update_data, index + 1)

update_data()

root.mainloop()
