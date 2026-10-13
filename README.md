# Sistema Embebido de Clasificación de Residuos (Fase 2)

**Integrantes:** Angely Burgos, Santiago Diaz, Juan Romero, Dilan Muñoz  
**Docente:** Joaquín F. Sánchez  
**Materia:** Arquitectura de Computadores — Universidad Sergio Arboleda  

Prototipo embebido de visión artificial para clasificación binaria de residuos en tiempo real (*Reciclable* vs. *No Reciclable*), operando de manera 100% local en una **Raspberry Pi 3 Modelo B+**.

---

## 1. Arquitectura y Entrenamiento del Modelo
- **Base:** MobileNetV2 preentrenada en ImageNet.
- **Dataset:** Más de 2.000 imágenes clasificadas en dos clases (`0: no_reciclable`, `1: reciclable`).
- **Optimización:** Cuantización post-entrenamiento a enteros de 8 bits (INT8 / UINT8) con dataset representativo.
- **Motor de Inferencia en Embebido:** Google LiteRT (`ai_edge_litert`).

---

## 2. Conexiones Físicas (GPIO)
- **LED Verde (Reciclable):** Pin físico 11 -> GPIO 17 (a través de resistencia limitadora de 330 Ω).
- **LED Rojo (No Reciclable):** Pin físico 13 -> GPIO 27 (a través de resistencia limitadora de 330 Ω).
- **Tierra (GND):** Pin físico 9 / 14 común a los cátodos de ambos LEDs.
- **Cámara:** Conexión USB 2.0 (`/dev/video0`).

---

## 3. Instalación y Ejecución en Raspberry Pi

### Requisitos del Sistema
Raspberry Pi OS Lite (64-bit):
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-numpy libatlas-base-dev libopenjp2-7 libtiff5

git clone [https://github.com/Angely-sof/clasificador-residuos-rpi.git](https://github.com/Angely-sof/clasificador-residuos-rpi.git)
cd clasificador-residuos-rpi
pip3 install -r requirements.txt
python3 main.py
