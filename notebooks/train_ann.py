import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Make sure the models folder exists
os.makedirs("models", exist_ok=True)

# 1. Load dataset (example: Pima Indians Diabetes)
data = pd.read_csv("data/diabetes.csv")   # put your dataset inside "data" folder

X = data.drop("Outcome", axis=1)   # Features
y = data["Outcome"]                # Target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build ANN model (3 layers: input, hidden, output)
model = Sequential()

# Input + Hidden layer 1 (12 neurons, relu activation)
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))

# Hidden layer 2 (8 neurons, relu activation)
model.add(Dense(8, activation='relu'))

# Output layer (1 neuron, sigmoid for binary classification)
model.add(Dense(1, activation='sigmoid'))

# 5. Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1)

# 7. Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# 8. Save model and scaler into models folder
model.save("models/ann_model.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model trained, evaluated, and saved in the models/ folder")
