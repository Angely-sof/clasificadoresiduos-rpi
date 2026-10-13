import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="modelo_residuos.tflite")
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()

dummy_input = np.zeros(in_det[0]['shape'], dtype=in_det[0]['dtype'])

tiempos = []
for _ in range(50):
    t_start = time.perf_counter()
    interpreter.set_tensor(in_det[0]['index'], dummy_input)
    interpreter.invoke()
    tiempos.append((time.perf_counter() - t_start) * 1000)

print(f"Latencia promedio de inferencia: {np.mean(tiempos):.2f} ms")
