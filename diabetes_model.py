# train_diabetes_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("data/diabetes.csv")
print("✅ Dataset Loaded")
print(df.head())

# 2. Features and Labels
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']                # 0 = No diabetes, 1 = Diabetes

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build ANN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# 7. Evaluate model
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")

# 8. Save model and scaler
model.save("models/diabetes_model.h5")
joblib.dump(scaler, "models/diabetes_scaler.pkl")
print("🎉 Diabetes model and scaler saved successfully!")
