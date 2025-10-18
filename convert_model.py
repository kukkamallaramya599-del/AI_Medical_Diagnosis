import tensorflow as tf

# Load the legacy .h5 model
model = tf.keras.models.load_model("models/ann_model.h5", compile=False)

# Save it in TensorFlow SavedModel format (recommended)
model.save("models/ann_model_converted", save_format="tf")

print("✅ Model converted and saved at: models/ann_model_converted/")
