import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. PARÁMETROS Y RUTAS DE ENTRADA/SALIDA
# ==========================================
DATASET_DIR = r"C:\dataset_residuos"
OUTPUT_DIR = r"./"
OUTPUT_MODEL_NAME = "modelo_residuos.tflite"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

print(f"Cargando dataset desde: {DATASET_DIR}")

# ==========================================
# 2. CARGA Y PARTICIONAMIENTO DEL DATASET
# ==========================================
# 80% para entrenamiento, 20% para validación
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

clases = train_ds.class_names
print(f"Clases detectadas en orden de índice: {clases}")

# Optimización de pipeline en memoria RAM
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. CONSTRUCCIÓN DE LA ARQUITECTURA
# ==========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1.0 / 127.5, offset=-1, input_shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(clases), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 4. ENTRENAMIENTO DE LA CABEZA DENSA
# ==========================================
print("\nIniciando entrenamiento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================================
# 5. CUANTIZACIÓN POST-ENTRENAMIENTO (INT8)
# ==========================================
print("\nIniciando proceso de cuantización a INT8...")

def representative_data_gen():
    for images, _ in train_ds.take(20):
        for img in images:
            yield [np.expand_dims(img.numpy(), axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# ==========================================
# 6. EXPORTACIÓN DEL ARCHIVO .TFLITE
# ==========================================
ruta_salida = os.path.join(OUTPUT_DIR, OUTPUT_MODEL_NAME)
with open(ruta_salida, "wb") as f:
    f.write(tflite_model_quant)

tamano_mb = len(tflite_model_quant) / (1024 * 1024)
print("=" * 60)
print(f"Modelo exportado con éxito en: {ruta_salida}")
print(f"Tamaño final del archivo: {tamano_mb:.2f} MB")
print("=" * 60)
