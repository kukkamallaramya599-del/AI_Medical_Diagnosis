import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("ann_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("ann_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion successful! ann_model.tflite saved.")
