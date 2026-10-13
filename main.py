import time
import cv2
import numpy as np
from gpiozero import LED
from ai_edge_litert.interpreter import Interpreter

# ==========================================
# 1. HARDWARE: Configuración de GPIO
# ==========================================
# Pin físico 11 -> GPIO 17 (LED Verde: Reciclable)
# Pin físico 13 -> GPIO 27 (LED Rojo: No Reciclable)
led_verde = LED(17)
led_rojo = LED(27)

# ==========================================
# 2. MOTOR DE IA: Carga del Modelo Cuantizado
# ==========================================
MODEL_PATH = "modelo_residuos.tflite"

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error crítico al inicializar LiteRT: {e}")
    exit(1)

# Extracción de metadatos de tensores
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']        # [1, 224, 224, 3]
input_dtype = input_details[0]['dtype']        # np.uint8 o np.int8
output_scale, output_zero_point = output_details[0]['quantization']

# ==========================================
# 3. ENTRADA DE VIDEO: Cámara USB
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: No se detectó la cámara en /dev/video0.")
    exit(1)

print("=" * 60)
print(" SISTEMA DE CLASIFICACIÓN DE RESIDUOS ACTIVO")
print(" Presiona Ctrl + C en cualquier momento para finalizar")
print("=" * 60)

# ==========================================
# 4. BUCLE PRINCIPAL DE INFERENCIA
# ==========================================
try:
    while True:
        t_inicio = time.perf_counter()

        # Captura de cuadro
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Preprocesamiento: Color y resolución
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[2]))

        # Ajuste de tipo de dato para el tensor de entrada
        if input_dtype == np.uint8:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
        elif input_dtype == np.int8:
            input_data = np.expand_dims(img_resized.astype(np.int32) - 128, axis=0).astype(np.int8)
        else:
            input_data = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

        # Ejecución de la red neuronal
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Decuantización a probabilidades reales (0.0 a 1.0)
        if output_details[0]['dtype'] in [np.int8, np.uint8]:
            probabilidades = (output_data[0].astype(np.float32) - output_zero_point) * output_scale
        else:
            probabilidades = output_data[0]

        # Interpretación de salida binaria:
        # Índice 0: no_reciclable | Índice 1: reciclable
        clase_predicha = int(np.argmax(probabilidades))
        confianza = float(probabilidades[clase_predicha] * 100)

        # Control físico de actuadores
        if clase_predicha == 1:
            led_verde.on()
            led_rojo.off()
            estado_texto = "RECICLABLE   "
        else:
            led_verde.off()
            led_rojo.on()
            estado_texto = "NO RECICLABLE"

        # Métricas de rendimiento
        t_delta = time.perf_counter() - t_inicio
        latencia_ms = t_delta * 1000
        fps = (1.0 / t_delta) if t_delta > 0 else 0

        print(f"[{estado_texto}] Confianza: {confianza:5.1f}% | Latencia: {latencia_ms:5.1f} ms | FPS: {fps:4.1f}")
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupción manual detectada.")
finally:
    # Liberación segura de hardware
    led_verde.off()
    led_rojo.off()
    cap.release()
    print("Pines GPIO y cámara liberados.")
