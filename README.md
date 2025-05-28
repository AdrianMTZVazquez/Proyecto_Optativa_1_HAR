# Proyecto de Clasificación de Movimientos Corporales con ESP32 y MPU6050

## 1. Introducción

Este proyecto tiene como objetivo desarrollar una interfaz gráfica de usuario (GUI) para la adquisición de datos de un sensor acelerómetro MPU6050 conectado a un ESP32 mediante Bluetooth Low Energy (BLE). La aplicación permite la visualización en tiempo real de los datos del acelerómetro, la recolección de datos para entrenamiento de modelos de aprendizaje automático, el entrenamiento de dichos modelos y la clasificación en tiempo real de actividades físicas.

## 2. Objetivos del Proyecto

*   Establecer comunicación BLE entre un ESP32 (con sensor MPU6050) y una aplicación de escritorio Python.
*   Visualizar en tiempo real los datos de aceleración (ejes X, Y, Z) y su Transformada Rápida de Fourier (FFT).
*   Implementar una funcionalidad para registrar los datos crudos del sensor en formato CSV, incluyendo etiquetas de actividad para facilitar el posterior entrenamiento de modelos.
*   Desarrollar una interfaz para la recolección de segmentos de datos etiquetados correspondientes a diferentes actividades físicas (ej. Quieto, Caminar, Correr).
*   Integrar capacidades para entrenar modelos de clasificación de actividades (ej. SVM, Random Forest) utilizando los datos recolectados.
*   Permitir guardar y cargar los modelos entrenados.
*   Realizar clasificación de actividad en tiempo real utilizando el modelo cargado.
*   Proporcionar una interfaz de usuario intuitiva y organizada mediante pestañas para las diferentes funcionalidades.

## 3. Arquitectura y Tecnologías

*   **Hardware**:
    *   ESP32 (microcontrolador con BLE).
    *   MPU6050 (sensor acelerómetro y giroscopio).
*   **Software (Aplicación de Escritorio)**:
    *   **Lenguaje**: Python 3.x
    *   **GUI**: PyQt5
    *   **Comunicación BLE**: `bleak` library
    *   **Procesamiento de Datos y Gráficas**: `numpy`, `scipy`, `pyqtgraph`
    *   **Machine Learning**: `scikit-learn` (opcional, con fallback a un clasificador dummy si no está instalado)
    *   **Manejo Asíncrono**: `asyncio`
*   **Estructura del Código Principal (`esp32_mpu6050_monitor.py`)**:
    *   `BLEDevice`: Clase para manejar la lógica de conexión, desconexión y streaming de datos BLE.
    *   `FeatureExtractor`: Clase con métodos estáticos para extraer características en el dominio del tiempo y la frecuencia de las señales del acelerómetro.
    *   `ActivityClassifier`: Clase para entrenar modelos de clasificación y realizar predicciones. Maneja diferentes tipos de modelos y la carga/guardado de los mismos.
    *   `MainWindow`: Clase principal de la GUI (hereda de `QMainWindow`). Gestiona la interfaz de usuario, la interacción con las otras clases, los temporizadores para actualización de gráficas y tareas asíncronas, y la lógica de las diferentes pestañas (Monitorización, Entrenamiento, Clasificación).

## 4. Funcionalidades Implementadas

La interfaz se organiza en tres pestañas principales:

### 4.1. Pestaña "Monitorización"

*   **Conexión BLE**: Botones para escanear/conectar y desconectar del dispositivo ESP32.
*   **Streaming de Datos**: Botones para iniciar/detener el envío de datos desde el ESP32.
*   **Visualización en Tiempo Real**:
    *   Gráfica de datos de aceleración (Ax, Ay, Az) en el dominio del tiempo.
    *   Gráfica de la magnitud de la FFT de las señales de aceleración.
*   **Registro CSV**:
    *   Botón para iniciar/detener el guardado de datos crudos del acelerómetro (`timestamp`, `ax`, `ay`, `az`) y la etiqueta de actividad seleccionada en la pestaña "Entrenamiento" en un archivo CSV.
    *   Los archivos se guardan en la carpeta `csv_logs/` con nombres basados en la fecha y hora.
    *   Indicador de estado del registro CSV.
*   **Registro de Estado**: Un área de texto que muestra mensajes de estado, errores y eventos importantes de la aplicación.

### 4.2. Pestaña "Entrenamiento"

*   **Recolección de Datos de Entrenamiento**:
    *   Selección de la actividad a registrar (ej. Quieto, Caminar, Correr, Agacharse, Saltar).
    *   Especificación del número de segmentos de datos a recolectar para la actividad seleccionada.
    *   Botón para iniciar/detener la recolección de datos.
    *   Barra de progreso para visualizar el avance de la recolección.
    *   Los datos recolectados se almacenan internamente para el entrenamiento.
*   **Entrenamiento y Gestión del Modelo**:
    *   Selección del tipo de modelo de Machine Learning a utilizar (ej. SVM, Random Forest - actualmente con fallback a dummy si scikit-learn no está).
    *   Botón "Entrenar Modelo": Utiliza los segmentos de datos recolectados para entrenar el modelo seleccionado.
    *   Botón "Guardar Modelo": Guarda el modelo entrenado en un archivo (`.pkl`) en la carpeta `har_models/`.
    *   Botón "Cargar Modelo": Carga un modelo previamente guardado.
*   **Registro de Entrenamiento**: Área de texto que muestra el progreso y resultados de la recolección y entrenamiento.

### 4.3. Pestaña "Clasificación en Tiempo Real"

*   **Estado de Clasificación**: Muestra la actividad predicha por el modelo cargado en tiempo real.
*   **Probabilidades (si aplica)**: Muestra las probabilidades asociadas a cada clase predicha (funcionalidad dependiente del tipo de clasificador).

## 5. Requisitos e Instalación

1.  **Hardware**:
    *   Un ESP32 programado para leer datos del MPU6050 y transmitirlos vía BLE. El firmware del ESP32 debe exponer un servicio BLE con una característica que notifique los datos del acelerómetro.
    *   Sensor MPU6050 conectado correctamente al ESP32.
2.  **Software**:
    *   Python 3.7 o superior.
    *   Las siguientes librerías de Python. Se recomienda crear un entorno virtual:
        ```bash
        pip install PyQt5 bleak numpy scipy pyqtgraph scikit-learn
        ```
        *Nota: `scikit-learn` es opcional para la funcionalidad básica de clasificación, pero necesaria para entrenar y usar modelos reales. Si no está instalada, se usará un clasificador dummy.*
3.  **Configuración del ESP32**:
    *   Asegúrate de que los UUIDs del servicio y la característica BLE en el script `esp32_mpu6050_monitor.py` (específicamente en la clase `BLEDevice`) coincidan con los definidos en el firmware de tu ESP32.
        *   `ACCEL_SERVICE_UUID`
        *   `ACCEL_CHAR_UUID`

## 6. Modo de Uso

1.  **Conectar Hardware**: Asegúrate de que tu ESP32 con el MPU6050 esté encendido y visible por BLE.
2.  **Ejecutar la Aplicación**:
    Navega al directorio `src/` y ejecuta el script:
    ```bash
    python esp32_mpu6050_monitor.py
    ```
3.  **Conectar al ESP32**:
    *   En la pestaña "Monitorización", haz clic en "Escanear y Conectar". La aplicación buscará dispositivos con el nombre "ESP32_Accelerometer" (configurable en el código).
    *   Una vez conectado, el estado se actualizará.
4.  **Iniciar Streaming**:
    *   Haz clic en "Iniciar Streaming" para comenzar a recibir y visualizar datos.
5.  **Registro CSV (Opcional)**:
    *   Si deseas guardar los datos crudos, ve a la pestaña "Entrenamiento", selecciona la actividad que estás realizando.
    *   Vuelve a "Monitorización" y haz clic en "Iniciar Registro CSV".
    *   Haz clic en "Detener Registro CSV" cuando termines. Los datos se guardarán en la carpeta `csv_logs/`.
6.  **Recolectar Datos para Entrenamiento**:
    *   Ve a la pestaña "Entrenamiento".
    *   Selecciona la "Actividad" que vas a realizar.
    *   Define el número de "Segmentos a recolectar".
    *   Haz clic en "Iniciar Recolección". Realiza la actividad mientras la aplicación recolecta los datos.
    *   Repite para todas las actividades que desees incluir en tu modelo.
7.  **Entrenar Modelo**:
    *   En la pestaña "Entrenamiento", selecciona el "Tipo de Modelo".
    *   Haz clic en "Entrenar Modelo". El progreso se mostrará en el log.
8.  **Guardar/Cargar Modelo**:
    *   Usa "Guardar Modelo" para guardar tu modelo entrenado.
    *   Usa "Cargar Modelo" para cargar un modelo existente.
9.  **Clasificación en Tiempo Real**:
    *   Ve a la pestaña "Clasificación en Tiempo Real".
    *   Si un modelo está cargado y entrenado, y los datos se están transmitiendo, verás la actividad predicha.

## 7. Diagrama de Flujo General (Conceptual)

```mermaid
graph TD
    A[Inicio de Aplicación] --> B{Interfaz Principal};
    B -- Pestaña Monitorización --> C[Controles BLE];
    C --> D[Conectar/Desconectar ESP32];
    D -- Conectado --> E[Iniciar/Detener Streaming];
    E -- Streaming Activo --> F[Recepción de Datos (ax,ay,az)];
    F --> G[Actualizar Gráficas Tiempo Real (Tiempo y FFT)];
    F --> H{¿Registro CSV Activo?};
    H -- Sí --> I[Guardar Datos en CSV con Etiqueta de Actividad];
    B -- Pestaña Entrenamiento --> J[Seleccionar Actividad y Segmentos];
    J --> K[Iniciar/Detener Recolección de Datos];
    K -- Datos Recolectados --> L[Almacenar Segmentos Etiquetados];
    L --> M[Seleccionar Tipo de Modelo];
    M --> N[Entrenar Modelo];
    N --> O[Guardar/Cargar Modelo Entrenado];
    B -- Pestaña Clasificación --> P[Cargar Modelo Entrenado si no está cargado];
    P -- Modelo Cargado --> Q{¿Streaming Activo?};
    Q -- Sí --> R[Extraer Features de Ventana Actual];
    R --> S[Predecir Actividad con Modelo];
    S --> T[Mostrar Actividad Predicha];
```

## 8. Futuras Mejoras

*   Implementación de más tipos de modelos de Machine Learning.
*   Optimización del rendimiento para altas tasas de muestreo.
*   Mejoras en la robustez de la conexión BLE.
*   Exportación/Importación de configuraciones de entrenamiento.
*   Visualización más detallada de las métricas del modelo entrenado.

---
*Este proyecto se basa en los conceptos y objetivos iniciales discutidos en [Proyecto de Optativa 1: Clasificación de movimientos corporales usando un acelerómetro y modelos de IA](URL_DE_TU_NOTION_AQUI) (Por favor, reemplaza con el enlace correcto si puedes compartirlo o resumir su contenido relevante).*
