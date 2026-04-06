import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(128,128),
    batch_size=32
)

# Normalize data
train_ds = train_ds.map(lambda x, y: (x/255.0, y))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
#model.fit(train_ds, epochs=3)

print("Model training complete 🎉")
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Load test image (put any X-ray image path here)
img_path = "C:\\Users\\vansh\\Downloads\\test.jpg"   # change this later

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > prediction[0][1]:
    print("🩺 Pneumonia Detected")
else:
    print("✅ Normal")