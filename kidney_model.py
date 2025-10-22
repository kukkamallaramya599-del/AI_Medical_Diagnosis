import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("data/kidney_disease.csv")
print("✅ Dataset Loaded")
print(df.head())

# 2. Replace '?' with NaN and drop rows with missing target
df.replace('?', np.nan, inplace=True)
df.dropna(subset=['classification'], inplace=True)

# Drop ID column if exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# 3. Encode categorical columns
map_yes_no = {'yes':1, 'no':0, ' yes':1}  # some values have a space
for col in ['htn','dm','cad','pe','ane']:
    if col in df.columns:
        df[col] = df[col].map(map_yes_no)

map_abnormal = {'normal':0,'abnormal':1}
for col in ['rbc','pc']:
    if col in df.columns:
        df[col] = df[col].map(map_abnormal)

map_present = {'notpresent':0,'present':1}
for col in ['pcc','ba']:
    if col in df.columns:
        df[col] = df[col].map(map_present)

map_appet = {'good':0,'poor':1}
for col in ['appet']:
    if col in df.columns:
        df[col] = df[col].map(map_appet)

# 4. Convert all remaining object columns to numeric safely
for col in df.columns:
    if df[col].dtype == 'object' and col != 'classification':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN after conversion
df.dropna(inplace=True)

# 5. Features and Labels
X = df.drop('classification', axis=1)
y = df['classification'].map({'ckd':1,'notckd':0})

print("✅ Features and labels ready")
print("X shape:", X.shape, "y shape:", y.shape)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Build ANN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 9. Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# 10. Evaluate
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")

# 11. Save model and scaler
model.save("models/kidney_model.h5")
joblib.dump(scaler, "models/kidney_scaler.pkl")
print("🎉 Kidney model and scaler saved successfully!")
