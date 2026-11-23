# Segmentación de Hojas Sanas, Secas y con Mancha Foliar mediante YOLOv8

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
git clone https://github.com/CristianMera25/Segmentacion-de-Hojas-Sanas-Secas-y-con-Mancha-Foliar-YOLOV8.git
cd Segmentacion-de-Hojas-Sanas-Secas-y-con-Mancha-Foliar-YOLOV8
```

### 2. Crear y activar el entorno virtual
En **Windows (PowerShell o CMD)**:
```bash
python -m venv venv
venv\Scripts\activate
```

En **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```
*(Verifica que el prefijo `(venv)` aparezca al inicio de la línea de comandos antes de continuar).*

### 3. Instalar las dependencias base
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Habilitar aceleración por GPU NVIDIA (Opcional)
Por defecto, el comando anterior puede instalar la versión de PyTorch compilada únicamente para CPU. Si dispones de una tarjeta gráfica NVIDIA (por ejemplo, series RTX 3000 o RTX 4000) y deseas ejecutar la segmentación a más de 50 FPS, reinstala PyTorch con soporte CUDA:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
*(Si utilizas una versión anterior de Python como 3.10 o 3.11, puedes emplear el índice `cu121` o `cu124`).*

---

## Ejecución de la Aplicación

### Requisito indispensable: Archivo de pesos (`best.pt`)
Antes de iniciar el programa, asegúrate de que el archivo **`best.pt`** (que contiene los pesos entrenados de la red neuronal `YOLOv8l-seg`) se encuentre ubicado en la carpeta raíz del proyecto, al mismo nivel que `app.py`. Si el archivo no existe o tiene otro nombre, la aplicación no podrá inicializar el modelo.

### Comando de inicio
Con el entorno virtual `(venv)` activado, ejecuta:

```bash
python app.py
```

### Modos de operación en la interfaz
Una vez abierta la ventana principal de **LeafSeg AI**, puedes operar el sistema de tres maneras:

1. **Inferencia en Tiempo Real (Iniciar Cámara):**
   * Al pulsar **Iniciar Cámara**, el sistema realiza una detección automática de hardware. Si existe una cámara web integrada o USB activa, inicia la captura directamente. Si no hay cámara física disponible pero se encuentra activo el cliente de DroidCam en red local (`127.0.0.1:4747`), la aplicación enlaza automáticamente dicho flujo de video.
   * El procesamiento corre en una arquitectura desacoplada de tres hilos (captura de video, inferencia neuronal y actualización de interfaz), permitiendo visualizar las máscaras de segmentación, el conteo por categoría, el índice de salud porcentual y los FPS reales sin bloquear la ventana.
2. **Análisis de Imagen Estática (Imagen):**
   * Permite seleccionar archivos `.jpg`, `.jpeg`, `.png` o `.webp` desde el disco para obtener un diagnóstico instantáneo.
3. **Análisis de Grabaciones (Video):**
   * Permite cargar archivos `.mp4`, `.avi`, `.mov` o `.mkv` para evaluar lotes completos grabados previamente en campo.
4. **Exportación de Evidencia (Guardar Diagnóstico):**
   * Guarda en la carpeta raíz una fotografía en alta resolución del fotograma actual con todas las segmentaciones y etiquetas superpuestas bajo el formato `diagnostico_leafseg_<timestamp>.jpg`.

---

## Arquitectura y Metodología

El flujo de procesamiento de cada fotograma sigue una tubería (pipeline) diseñada para mantener coherencia exacta entre la fase de entrenamiento y la fase de inferencia en producción:

1. **Captura de Fotograma:** Lectura continua desde la fuente de video en formato `BGR`.
2. **Preprocesamiento Cromático (LAB + CLAHE):**
   * Conversión del espacio de color `BGR` a `LAB` (Luminosidad, canal A verde-rojo, canal B azul-amarillo).
   * Aplicación de ecualización adaptativa de histograma limitada por contraste (**CLAHE**, `clipLimit=2.0`, `tileGridSize=(8, 8)`) exclusivamente sobre el canal `L` (Luminosidad).
   * Reconstrucción a `BGR`. Este paso estandariza las variaciones bruscas de iluminación natural o artificial sin alterar los pigmentos verdes, amarillos o marrones de la hoja.
3. **Inferencia YOLOv8l-seg:** Evaluación con umbral de confianza de `0.45`, generando cajas delimitadoras, probabilidades de clase y polígonos de segmentación por instancia.
4. **Cálculo Fitosanitario:** Conteo individual por clase y cálculo del Índice de Salud mediante la relación entre hojas sanas y el total de hojas detectadas en escena.

---

## Conjunto de Datos (Dataset)

El conjunto de datos fue construido, curado y anotado manualmente específicamente para este proyecto mediante la herramienta *Smart Polygon* en Roboflow.

| Característica | Detalle |
| :--- | :--- |
| **Plataforma** | Roboflow Universe |
| **Workspace** | `uprojects` |
| **Proyecto** | `Yolov8 Leaf Segmentation` (Versión 1) |
| **Tipo de Anotación** | Segmentación por Instancia (Polígonos) |
| **Imágenes Originales** | 350 fotografías capturadas bajo diversas condiciones lumínicas |
| **Imágenes Totales (tras Augmentation)** | 838 imágenes |

### Definición de Clases y Distribución

| Clase | Descripción |
| :--- | :--- |
| **Sana** | Tejido foliar saludable, coloración uniforme sin necrosis |
| **Mancha** | Presencia de lesiones fúngicas, bacterianas o clorosis localizada |
| **Seca** | Tejido necrótico avanzado, deshidratación severa o senescencia |

| Conjunto | Cantidad de Imágenes |
| :--- | :---: |
| **Entrenamiento (Train)** | 732 |
| **Validación (Val)** | 71 |
| **Prueba (Test)** | 35 |
| **Total** | **838** |

### Técnicas de Aumento de Datos (Data Augmentation)
Para mejorar la capacidad de generalización frente a diferentes ángulos de cámara e iluminación en vivero, se aplicaron transformaciones geométricas y fotométricas: rotaciones, volteos horizontales, volteos verticales y variaciones controladas de brillo.

---

## Entrenamiento del Modelo

Si deseas replicar el entrenamiento o ajustar el modelo con nuevas muestras, puedes descargar el dataset desde Roboflow y entrenar la arquitectura base `yolov8l-seg.pt`:

```python
from roboflow import Roboflow
from ultralytics import YOLO

# Descarga del dataset anotado
rf = Roboflow(api_key="TU_API_KEY")
project = rf.workspace("uprojects").project("Yolov8 Leaf Segmentation")
dataset = project.version(1).download("yolov8")

# Entrenamiento del modelo de segmentación
model = YOLO("yolov8l-seg.pt")
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640
)
```

---

## Estructura del Repositorio

```text
Segmentacion-de-Hojas-Sanas-Secas-y-con-Mancha-Foliar-YOLOV8/
├── app.py                  # Código fuente de la aplicación de escritorio
├── best.pt                 # Pesos entrenados del modelo YOLOv8l-seg
├── requirements.txt        # Lista de dependencias de Python
├── .gitignore              # Exclusiones de control de versiones
├── LICENSE                 # Licencia MIT
└── README.md               # Documentación técnica del proyecto
```

---

## Autor

**Cristian Andrés Mera Giraldo**