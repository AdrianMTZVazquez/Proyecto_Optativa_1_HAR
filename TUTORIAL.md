# Tutorial del Proyecto: Adquisición y Clasificación de Actividad Humana

## 1. Introducción

Bienvenido al proyecto de Adquisición y Clasificación de Actividad Humana (HAR). Este sistema está diseñado para recolectar datos de sensores (acelerómetros) y utilizar técnicas de Machine Learning para identificar la actividad que está realizando una persona.

Este tutorial te guiará a través de dos aspectos principales:
1.  **Cómo gestionar el código del proyecto utilizando Git y GitHub/GitLab.**
2.  **Cómo configurar, utilizar y entender la interfaz principal de adquisición y clasificación de datos (`interfaz_DAQ_V3.py`).**

El proyecto consta de dos componentes de software principales:
*   `esp32_mpu6050_monitor.py`: Una interfaz para visualizar en tiempo real datos de un acelerómetro MPU6050 conectado a un ESP32, transmitidos vía Bluetooth Low Energy (BLE).
*   `interfaz_DAQ_V3.py`: Una interfaz más completa para la adquisición de datos utilizando hardware NI-DAQ, creación de datasets etiquetados, entrenamiento de modelos de Machine Learning y clasificación en tiempo real.

---

## 2. Gestión de Código con Git y GitHub/GitLab

Git es un sistema de control de versiones que te ayuda a rastrear cambios en tu código. GitHub o GitLab son plataformas que alojan repositorios Git en la nube, facilitando la colaboración.

### 2.1. Conceptos Básicos de Git

*   **Repositorio (Repository/Repo):** Una carpeta que contiene todo tu proyecto y el historial de cambios.
*   **Commit:** Una "instantánea" de los cambios guardados en tu repositorio.
*   **Branch (Rama):** Una línea de desarrollo independiente. La rama principal suele ser `main` o `master`.
*   **Push (Empujar):** Enviar tus commits locales a un repositorio remoto (ej. GitHub).
*   **Pull (Tirar):** Traer cambios desde un repositorio remoto a tu repositorio local.
*   **Clone (Clonar):** Crear una copia local de un repositorio remoto existente.

### 2.2. Configuración Inicial de Git (si es la primera vez)

Abre una terminal o Git Bash y configura tu nombre y correo:
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu.correo@example.com"
```

### 2.3. Crear un Repositorio Local para tu Proyecto

1.  Navega a la carpeta raíz de tu proyecto en la terminal:
    ```bash
    cd ruta/a/tu/Proyecto_Optativa_1_HAR
    ```
2.  Inicializa un repositorio Git:
    ```bash
    git init
    ```
    Esto crea una subcarpeta oculta `.git` que almacena toda la información de Git.

### 2.4. Crear un Archivo `.gitignore`

Es crucial para evitar subir archivos innecesarios (ej. temporales de Python, logs, datos sensibles). Crea un archivo llamado `.gitignore` en la raíz de tu proyecto con el siguiente contenido (puedes adaptarlo):

```
# Python
__pycache__/
*.py[cod]
*$py.class

# Entornos virtuales
venv/
env/
.venv/
*.env

# Archivos de IDE
.vscode/
.idea/

# Archivos de sistema operativo
.DS_Store
Thumbs.db

# Datos (si son muy grandes o sensibles, considera no subirlos o usar Git LFS)
# *.csv
# *.pkl
# dataset/

# Logs
*.log
```

### 2.5. Crear un Repositorio Remoto (ej. en GitHub)

1.  Ve a [GitHub](https://github.com) e inicia sesión.
2.  Haz clic en "New" o el símbolo "+" para crear un nuevo repositorio.
3.  Dale un nombre (ej. `Proyecto_Optativa_1_HAR`), una descripción opcional.
4.  Elige si será público o privado.
5.  **Importante:** NO inicialices el repositorio remoto con un `README`, `.gitignore` o licencia si ya tienes un proyecto local existente que quieres subir. Crea un repositorio vacío.
6.  Copia la URL del repositorio remoto (ej. `https://github.com/tu-usuario/Proyecto_Optativa_1_HAR.git`).

### 2.6. Conectar tu Repositorio Local con el Remoto

En tu terminal, dentro de la carpeta del proyecto:
```bash
git remote add origin https://github.com/tu-usuario/Proyecto_Optativa_1_HAR.git
```
(Reemplaza la URL con la tuya). `origin` es el nombre por defecto para tu repositorio remoto.

### 2.7. Flujo de Trabajo Básico: Add, Commit, Push

1.  **Verificar el estado:**
    ```bash
    git status
    ```
    Muestra qué archivos han sido modificados o son nuevos.

2.  **Añadir archivos al "staging area" (prepararlos para el commit):**
    *   Para añadir todos los archivos nuevos/modificados:
        ```bash
        git add .
        ```
    *   Para añadir un archivo específico:
        ```bash
        git add src/interfaz_DAQ_V3.py
        ```

3.  **Hacer un commit (guardar los cambios en tu historial local):**
    ```bash
    git commit -m "Mensaje descriptivo del commit"
    ```
    Ejemplo: `git commit -m "Feat: Implementación inicial de la interfaz DAQ"`

4.  **Empujar (push) tus commits al repositorio remoto:**
    La primera vez que hagas push a una nueva rama (o si la rama `main` no existe en el remoto):
    ```bash
    git push -u origin main
    ```
    (Si tu rama principal se llama `master`, usa `master` en lugar de `main`).
    Para pushes subsecuentes en la misma rama:
    ```bash
    git push
    ```

### 2.8. Diagrama Conceptual: Flujo de Git Básico

*   **Puedes crear este diagrama en draw.io:**
    *   **Caja 1: "Espacio de Trabajo (Working Directory)"**
        *   Descripción: Tus archivos locales tal como los editas.
        *   Archivos: `interfaz_DAQ_V3.py`, `README.md`, etc.
    *   **Caja 2: "Área de Staging (Index)"**
        *   Descripción: Archivos marcados para el próximo commit.
        *   Flecha desde "Espacio de Trabajo" a "Área de Staging" con la etiqueta "`git add`".
    *   **Caja 3: "Repositorio Local (.git)"**
        *   Descripción: Historial de commits en tu máquina.
        *   Flecha desde "Área de Staging" a "Repositorio Local" con la etiqueta "`git commit`".
    *   **Caja 4: "Repositorio Remoto (GitHub/GitLab)"**
        *   Descripción: Copia del repositorio en la nube.
        *   Flecha bidireccional entre "Repositorio Local" y "Repositorio Remoto".
            *   Flecha hacia Remoto: "`git push`"
            *   Flecha desde Remoto: "`git pull` / `git clone`"

---

## 3. Entendiendo y Utilizando la Interfaz `interfaz_DAQ_V3.py`

Esta interfaz es la herramienta principal para la adquisición de datos con hardware NI-DAQ, la creación de datasets etiquetados para HAR, el entrenamiento de modelos y la clasificación en tiempo real.

### 3.1. Configuración del Entorno

**A. Requisitos de Software:**
*   Python 3.7 o superior.
*   Las siguientes librerías de Python:
    *   `PyQt5` (para la interfaz gráfica)
    *   `numpy` (para manejo numérico)
    *   `pyqtgraph` (para gráficos en tiempo real)
    *   `nidaqmx` (para interactuar con hardware NI-DAQ)
    *   `scikit-learn` (opcional, para las funciones de Machine Learning)
    *   `pandas` (para manejo de datos, especialmente para guardar/cargar CSV)
    *   `joblib` (para guardar/cargar modelos de scikit-learn)

**B. Instalación de Dependencias:**
Se recomienda crear un entorno virtual para tu proyecto.
```bash
python -m venv env
# Activar el entorno:
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate
```
Luego, instala las librerías. Idealmente, tendrás un archivo `requirements.txt` con todas las dependencias:
```
PyQt5
numpy
pyqtgraph
nidaqmx
scikit-learn
pandas
joblib
```
Instálalas con:
```bash
pip install -r requirements.txt
```
Si no tienes `requirements.txt`, instálalas individualmente:
```bash
pip install PyQt5 numpy pyqtgraph nidaqmx scikit-learn pandas joblib
```

**C. Requisitos de Hardware:**
*   Un dispositivo NI-DAQ compatible (ej. cDAQ-9174 con módulos de entrada analógica).
*   Sensores de aceleración compatibles con tu hardware DAQ.

### 3.2. Arquitectura General del Proyecto

*   **Diagrama Conceptual 1: Arquitectura General (draw.io)**
    *   **Componente 1: Hardware de Adquisición**
        *   Opción A: "ESP32 + MPU6050" (conectado vía BLE)
        *   Opción B: "NI-DAQ + Sensores" (conectado vía USB/PXI)
    *   **Componente 2: Software de Monitoreo/Adquisición**
        *   Si Opción A: "`esp32_mpu6050_monitor.py`"
            *   Funciones: Conexión BLE, recepción de datos, visualización simple.
        *   Si Opción B: "`interfaz_DAQ_V3.py`"
            *   Funciones: Control NI-DAQ, visualización avanzada, grabación etiquetada, entrenamiento ML, clasificación.
    *   **Componente 3: "Dataset Almacenado"**
        *   Formato: Archivos CSV.
        *   Contenido: Datos crudos del sensor, timestamps, etiquetas de actividad.
        *   Flecha desde "`interfaz_DAQ_V3.py`" (Grabación) hacia "Dataset Almacenado".
    *   **Componente 4: "Modelo de ML Entrenado"**
        *   Formato: Archivo `.pkl` (o similar).
        *   Flecha desde "`interfaz_DAQ_V3.py`" (Entrenamiento) hacia "Modelo de ML Entrenado".
        *   Flecha desde "Dataset Almacenado" hacia "`interfaz_DAQ_V3.py`" (Entrenamiento).
    *   **Componente 5: "Usuario"**
        *   Interactúa con las interfaces gráficas.

### 3.3. Interfaz `interfaz_DAQ_V3.py` a Detalle

La interfaz se organiza en pestañas para diferentes funcionalidades.

#### 3.3.1. Pestaña de Adquisición

Aquí configuras y controlas la adquisición de datos desde el hardware NI-DAQ y grabas segmentos de datos etiquetados.

*   **Controles Principales:**
    *   **Botón "Iniciar/Detener Adquisición":** Comienza o para la lectura de datos del NI-DAQ.
    *   **"Frecuencia (Hz)":** Selector para la tasa de muestreo de los sensores.
    *   **"Modo IEPE":** Checkbox para habilitar la alimentación IEPE si tus sensores lo requieren.
    *   **"Sensibilidad (mV/g)":** Configuración de la sensibilidad del sensor.
*   **Visualización de Datos:**
    *   **Gráfico "Señales de Aceleración":** Muestra las señales de los acelerómetros en el dominio del tiempo en tiempo real.
    *   **Gráfico "Espectro de Frecuencia":** Muestra la Transformada Rápida de Fourier (FFT) de las señales en tiempo real.
*   **Panel "Grabación de Actividades":**
    *   **"Actividad":** Dropdown para seleccionar la etiqueta de la actividad que se va a grabar (ej. "Caminar", "Correr", "Quieto"). Estas actividades se definen en la lista `ACTIVITIES` en el código.
    *   **"Duración (s)":** Tiempo en segundos durante el cual se grabará la actividad seleccionada.
    *   **Botón "Iniciar/Detener Grabación":** Comienza o para la grabación del segmento de datos con la etiqueta seleccionada.
    *   **Barra de Progreso:** Muestra el progreso de la grabación actual.
*   **Funcionamiento de la Grabación:**
    1.  Inicia la adquisición general de datos.
    2.  Selecciona una actividad del dropdown.
    3.  Establece la duración de la grabación.
    4.  Presiona "Iniciar Grabación".
    5.  Durante el tiempo especificado, los datos crudos de los sensores, junto con un timestamp para cada muestra y la etiqueta de la actividad, se almacenan temporalmente.
    6.  Al finalizar la grabación (automáticamente o al presionar "Detener Grabación"), estos datos se guardan o se añaden a un archivo CSV. El nombre del archivo podría ser configurable o predefinido (ej. `dataset_actividades.csv`).
    7.  Cada fila en el CSV representaría una muestra de todos los canales del acelerómetro, más una columna para el timestamp y una columna para la etiqueta de actividad.

*   **Diagrama Conceptual 2: Flujo de Adquisición y Grabación Etiquetada (draw.io)**
    *   **Entradas:**
        *   "Configuración del Usuario" (Frecuencia, Actividad, Duración)
        *   "Señal del Hardware NI-DAQ"
    *   **Proceso Central: `interfaz_DAQ_V3.py` (Pestaña Adquisición)**
        *   Módulo `AdquisicionAceleracionThread`: Lee datos del DAQ.
        *   Módulo `Actualización de Gráficos`: Muestra datos en tiempo real.
        *   Módulo `Lógica de Grabación` (`toggle_grabacion`, `actualizar_grabacion`):
            *   Recibe datos del buffer de adquisición.
            *   Asocia la etiqueta de `combo_actividad`.
            *   Genera timestamps.
            *   Controla la duración con `spin_duracion`.
    *   **Salidas:**
        *   "Visualización en Gráficos (Tiempo Real y FFT)"
        *   "Archivo `dataset_actividades.csv`" (Columnas: timestamp, accX, accY, accZ, ..., etiqueta_actividad)

#### 3.3.2. Pestaña de Entrenamiento

Esta pestaña te permite cargar los datos grabados, extraer características y entrenar un modelo de Machine Learning.

*   **Controles Principales:**
    *   **Botón "Cargar Datos":** Permite seleccionar y cargar un archivo CSV (como `dataset_actividades.csv`) que contiene los datos etiquetados.
    *   **Tabla de Datos:** Muestra un resumen o las primeras filas de los datos cargados.
    *   **Selector de Tipo de Modelo:** (ej. SVM, Random Forest, etc.)
    *   **Configuración de Extracción de Características:** (ej. tamaño de ventana, solapamiento, tipo de características a extraer - estadísticas, FFT, etc.)
    *   **Botón "Entrenar Modelo":** Inicia el proceso de entrenamiento.
    *   **Indicador de Progreso/Estado del Entrenamiento.**
    *   **Botón "Guardar Modelo":** Guarda el modelo entrenado (ej. como un archivo `.pkl`).
*   **Funcionamiento del Entrenamiento:**
    1.  Carga el dataset CSV.
    2.  Los datos se segmentan en ventanas.
    3.  Se extraen características de cada ventana utilizando `FeatureExtractor`.
    4.  Las características y sus etiquetas correspondientes se utilizan para entrenar el modelo seleccionado (`ActivityClassifier`).
    5.  Se muestra información sobre el rendimiento del modelo (opcional).
    6.  El modelo entrenado se puede guardar para su uso posterior.

*   **Diagrama Conceptual 3: Flujo de Entrenamiento del Modelo (draw.io)**
    *   **Entradas:**
        *   "Archivo `dataset_actividades.csv`"
        *   "Configuración del Usuario" (Tipo de modelo, parámetros de características)
    *   **Proceso Central: `interfaz_DAQ_V3.py` (Pestaña Entrenamiento)**
        *   Módulo `Carga de Datos`: Lee y preprocesa el CSV.
        *   Módulo `Segmentación y Extracción de Características` (`FeatureExtractor`):
            *   Aplica ventanas a los datos.
            *   Calcula características.
        *   Módulo `Entrenamiento del Modelo` (`ActivityClassifier.train`):
            *   Utiliza características y etiquetas.
            *   Entrena el clasificador (ej. SVM).
    *   **Salidas:**
        *   "Modelo Entrenado (`modelo_har.pkl`)"
        *   "Métricas de Rendimiento (Opcional)"

#### 3.3.3. Pestaña de Clasificación

Permite cargar un modelo previamente entrenado y realizar clasificación en tiempo real utilizando los datos que se están adquiriendo del NI-DAQ.

*   **Controles Principales:**
    *   **Botón "Cargar Modelo":** Permite seleccionar y cargar un archivo de modelo entrenado (ej. `modelo_har.pkl`).
    *   **Botón "Iniciar/Detener Clasificación":** Comienza o para el proceso de clasificación en tiempo real.
    *   **Visualización del Resultado:** Muestra la actividad predicha por el modelo en tiempo real.
*   **Funcionamiento de la Clasificación:**
    1.  Se carga un modelo entrenado.
    2.  Se inicia la adquisición de datos (si no está ya activa desde la pestaña de Adquisición).
    3.  Los datos crudos se procesan en ventanas y se extraen características de la misma manera que durante el entrenamiento.
    4.  Estas características se pasan al modelo cargado, que predice la actividad.
    5.  La actividad predicha se muestra en la interfaz.

*   **Diagrama Conceptual 4: Flujo de Clasificación en Tiempo Real (draw.io)**
    *   **Entradas:**
        *   "Señal del Hardware NI-DAQ (Tiempo Real)"
        *   "Modelo Entrenado (`modelo_har.pkl`)" (Cargado por el usuario)
    *   **Proceso Central: `interfaz_DAQ_V3.py` (Pestaña Clasificación)**
        *   Módulo `AdquisicionAceleracionThread`: Lee datos del DAQ.
        *   Módulo `Segmentación y Extracción de Características` (`FeatureExtractor`): Procesa datos en tiempo real.
        *   Módulo `Predicción del Modelo` (`ActivityClassifier.predict`): Usa el modelo cargado.
    *   **Salidas:**
        *   "Etiqueta de Actividad Predicha (Mostrada en GUI)"

---

## 4. Contribuyendo al Proyecto (Opcional, si es un equipo)

Si trabajas en equipo o quieres que otros contribuyan:

*   **Forking (Bifurcación):** Los colaboradores pueden "bifurcar" tu repositorio en GitHub, creando su propia copia.
*   **Branches (Ramas):** Crear ramas para nuevas características o correcciones (`git checkout -b nombre-rama`).
*   **Pull Requests (Solicitudes de Extracción):** Una vez que los cambios están listos en su rama, pueden enviar un "Pull Request" al repositorio original para que revises e integres sus cambios.

---

## 5. Conclusión

Este tutorial te ha proporcionado las bases para gestionar el código del proyecto con Git y para entender y operar la interfaz `interfaz_DAQ_V3.py`. ¡Esperamos que sea de gran utilidad para tus experimentos de Reconocimiento de Actividad Humana!

No dudes en expandir este tutorial con más detalles específicos de tu implementación o problemas comunes que encuentres.
