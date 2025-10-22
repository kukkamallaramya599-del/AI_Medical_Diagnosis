# train_heart_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("data/heart_disease.csv")

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())

# ---------------------------
# 2. Clean and Encode Data
# ---------------------------

# Drop ID column if exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Convert 'sex' to numeric
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Convert other categorical string columns to numeric using get_dummies
categorical_cols = ['cp', 'thal', 'slope']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop any remaining non-numeric columns
for col in df.columns:
    if df[col].dtype == 'object':
        df = df.drop(col, axis=1)

# Features and Labels
X = df.drop('num', axis=1)
y = df['num']

# Convert target to binary classification (0 = no disease, 1 = any disease)
y = y.apply(lambda x: 0 if x == 0 else 1)

print("✅ Features and Labels ready")
print("X shape:", X.shape, "y shape:", y.shape)

# ---------------------------
# 3. Split Train-Test Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Scale Features
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 5. Build ANN Model
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------
# 6. Train Model
# ---------------------------
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
# 7. Evaluate Model
# ---------------------------
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")

# ---------------------------
# 8. Save Model and Scaler
# ---------------------------
model.save("models/heart_model.h5")
joblib.dump(scaler, "models/heart_scaler.pkl")

print("🎉 Model and Scaler saved successfully!")
