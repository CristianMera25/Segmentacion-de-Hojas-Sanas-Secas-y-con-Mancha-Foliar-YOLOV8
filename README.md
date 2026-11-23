# LeafSeg AI: Segmentación de Patologías Foliares en Tiempo Real mediante YOLOv8

Sistema de visión por computador basado en segmentación por instancias (`YOLOv8l-seg`) para la detección, clasificación y conteo automático de hojas sanas, secas y afectadas por mancha foliar en tiempo real.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Requisitos del Sistema y Librerías](#requisitos-del-sistema-y-librerías)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [Ejecución de la Aplicación](#ejecución-de-la-aplicación)
5. [Arquitectura y Metodología](#arquitectura-y-metodología)
6. [Conjunto de Datos (Dataset)](#conjunto-de-datos-dataset)
7. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
8. [Estructura del Repositorio](#estructura-del-repositorio)
9. [Autor](#autor)

---

## Descripción General

La inspección fitosanitaria manual en viveros y cultivos suele ser un proceso lento, subjetivo y difícil de escalar. La detección tardía de patologías foliares compromete la capacidad fotosintética de la planta, reduce el rendimiento del cultivo y genera pérdidas económicas.

Este proyecto integra un modelo de aprendizaje profundo (**YOLOv8l-seg**) dentro de una aplicación de escritorio multihilo capaz de procesar flujos de video en tiempo real, imágenes estáticas o grabaciones locales. Para cada hoja detectada, el sistema genera una máscara de segmentación a nivel de píxel, clasifica su estado sanitario en tres categorías (**Sana**, **Mancha**, **Seca**) y calcula en vivo el **Índice de Salud** del lote observado.

---

## Requisitos del Sistema y Librerías

### Requisitos de Hardware y Software
* **Sistema Operativo:** Windows 10/11, Linux (Ubuntu 20.04+) o macOS.
* **Python:** Versión 3.10 a 3.13 (se recomienda Python 3.12).
* **Aceleración por Hardware (Opcional pero recomendada):** Tarjeta gráfica NVIDIA compatible con CUDA para inferencia en tiempo real a alta tasa de cuadros por segundo (FPS). Si no se detecta una GPU compatible, la aplicación opera automáticamente en modo CPU.
* **Entrada de Video:** Cámara web integrada, cámara USB externa o transmisión mediante DroidCam.

### Librerías Utilizadas
El proyecto depende de las siguientes bibliotecas incluidas en el archivo `requirements.txt`:

* **`ultralytics`:** Framework principal utilizado para cargar los pesos del modelo `YOLOv8l-seg`, ejecutar las predicciones de segmentación por instancias y renderizar las máscaras sobre los fotogramas.
* **`torch` y `torchvision` (PyTorch):** Motor de cálculo tensorial que ejecuta la red neuronal profunda en GPU (mediante CUDA) o en CPU.
* **`opencv-python` (`cv2`):** Encargada de la captura de video desde la cámara o archivos multimedia, la conversión entre espacios de color (`BGR` a `LAB` y `RGB`), la aplicación del filtro de ecualización adaptativa **CLAHE** y el redimensionamiento de fotogramas.
* **`customtkinter`:** Biblioteca moderna basada en Tkinter utilizada para construir la interfaz gráfica de usuario (GUI) en modo oscuro, gestionar los paneles de control y actualizar los contadores en vivo.
* **`pillow` (`PIL`):** Puente de conversión entre las matrices numéricas de OpenCV y los objetos de imagen compatibles con la interfaz gráfica de CustomTkinter.
* **`numpy`:** Manejo eficiente de arreglos multidimensionales y operaciones vectorizadas sobre los tensores de salida y canales de imagen.

---

## Instalación y Configuración

Sigue estos pasos en tu terminal para preparar un entorno aislado y evitar conflictos con otras instalaciones de Python en tu equipo.

### 1. Clonar el repositorio
```bash
git clone https://github.com/CristianMera25/Segmentacion-de-Hojas-Sanas-Secas-y-con-Mancha-Foliar-YOLOV8
cd leaf-disease-segmentation-yolov8