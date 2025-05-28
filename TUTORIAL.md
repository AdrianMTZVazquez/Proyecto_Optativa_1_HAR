# Tutorial: Monitor de Acelerómetro ESP32-MPU6050 con BLE

## 1. Introducción

Bienvenido al proyecto de monitorización de datos del acelerómetro MPU6050 desde un ESP32. Este sistema está diseñado para conectar un ESP32 (que ejecuta MicroPython y lee un sensor MPU6050) a una aplicación de escritorio Python vía Bluetooth Low Energy (BLE) para visualizar los datos del acelerómetro en tiempo real.

Este tutorial te guiará a través de dos aspectos principales:
1.  **Cómo gestionar el código del proyecto utilizando Git y GitHub/GitLab.**
2.  **Cómo configurar, utilizar y entender la interfaz `esp32_mpu6050_monitor.py` y su componente clave para la comunicación BLE, la clase `BLEDevice`.**

El proyecto se centra principalmente en el script `esp32_mpu6050_monitor.py`, que proporciona una interfaz gráfica para:
*   Escanear y conectarse a un dispositivo ESP32 específico.
*   Recibir datos del acelerómetro (ax, ay, az) transmitidos por BLE.
*   Visualizar estos datos en gráficos en tiempo real.
*   Controlar el inicio y la detención del flujo de datos desde el ESP32.

---


## 2. Entendiendo y Utilizando la Interfaz `esp32_mpu6050_monitor.py`

Esta aplicación de escritorio Python te permite conectar y visualizar datos de un acelerómetro MPU6050 que está siendo leído por un ESP32 y transmitido vía Bluetooth Low Energy (BLE).

### 2.1. Configuración del Entorno

**A. Requisitos de Software (Python y Librerías):**
*   Python 3.7 o superior.
*   Las siguientes librerías de Python:
    *   `PyQt5`: Para la interfaz gráfica de usuario (GUI).
    *   `pyqtgraph`: Para los gráficos en tiempo real.
    *   `bleak`: Para la comunicación Bluetooth Low Energy asíncrona.
    *   `numpy`: Para el manejo eficiente de arrays de datos (aunque su uso directo en este script es mínimo, es común en el ecosistema).

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
Luego, instala las librerías. Si tienes un archivo `requirements.txt` similar a este:
```
PyQt5
pyqtgraph
bleak
numpy
```
Instálalas con:
```bash
pip install -r requirements.txt
```
O individualmente:
```bash
pip install PyQt5 pyqtgraph bleak numpy
```

**C. Requisitos de Hardware:**
*   Un ESP32 que tenga **cargado y ejecutándose el script `src_py/main.py`** (incluido en la carpeta `src_py` de este proyecto). Este script es esencial ya que se encarga de leer los datos del sensor MPU6050 y gestionar toda la comunicación BLE desde el ESP32. El ESP32 debe estar programado para:
    *   Lea datos de un sensor MPU6050 conectado vía I2C.
    *   Anuncie un servicio BLE con un nombre específico (definido como `ESP32_DEVICE_NAME` en `esp32_mpu6050_monitor.py`).
    *   Tenga una característica BLE para transmitir los datos del acelerómetro (UUID definido como `TX_CHAR_UUID`).
    *   Tenga una característica BLE para recibir comandos (UUID definido como `RX_CHAR_UUID`, ej. 'start', 'stop').
*   Un adaptador Bluetooth en tu PC compatible con BLE.

### 2.2. Funcionamiento Detallado de `esp32_mpu6050_monitor.py` (Diagramas Mermaid)

Esta sección detalla el funcionamiento interno de la aplicación `esp32_mpu6050_monitor.py` utilizando diagramas de secuencia Mermaid para ilustrar los flujos clave.

#### 2.2.1. Componentes Principales e Inicialización

Este diagrama muestra la creación e inicialización de los componentes principales de la aplicación: `MainWindow` (la interfaz gráfica) y `BLEDevice` (el manejador de la comunicación BLE), junto con el bucle de eventos `asyncio` que gestiona las operaciones asíncronas.

```mermaid
sequenceDiagram
    participant User
    participant PythonScript as esp32_mpu6050_monitor.py
    participant MainWindow
    participant BLEDevice
    participant AsyncioLoop as Bucle asyncio

    User->>PythonScript: Ejecuta el script
    PythonScript->>+MainWindow: main_window = MainWindow()
    MainWindow->>MainWindow: setup_ui() (botones, gráficos, etc.)
    MainWindow->>+BLEDevice: self.ble_device = BLEDevice(self.loop)
    BLEDevice->>BLEDevice: Inicializa atributos (cliente=None, señales PyQt, etc.)
    MainWindow->>+AsyncioLoop: self.timer_async = QTimer()
    MainWindow->>AsyncioLoop: self.timer_async.timeout.connect(self.run_async_tasks)
    MainWindow->>AsyncioLoop: self.timer_async.start(50)
    Note right of AsyncioLoop: El bucle asyncio se ejecuta<br/>periódicamente de forma no bloqueante.
    deactivate AsyncioLoop
    deactivate BLEDevice
    deactivate MainWindow
```

#### 2.2.2. Proceso de Conexión BLE

Este diagrama ilustra la secuencia de eventos cuando el usuario inicia una conexión BLE con el ESP32.

```mermaid
sequenceDiagram
    participant User
    participant MainWindow
    participant BLEDevice
    participant AsyncioLoop as Bucle asyncio
    participant ESP32Device as ESP32 (Periférico BLE)

    User->>+MainWindow: Clic en botón "Conectar"
    MainWindow->>MainWindow: connect_ble_device()
    MainWindow->>+BLEDevice: self.ble_device.connect()
    BLEDevice->>BLEDevice: connect() (método síncrono)
    BLEDevice->>+AsyncioLoop: self.loop.create_task(self.connect_async())
    AsyncioLoop->>+BLEDevice: Ejecuta connect_async()
    BLEDevice->>BLEDevice: Emite status_update_signal("Escaneando...")
    BLEDevice->>ESP32Device: BleakScanner.discover()
    alt Dispositivo Encontrado
        ESP32Device-->>BLEDevice: Devuelve dispositivo (dirección MAC)
        BLEDevice->>BLEDevice: self.device_address = dirección_mac
        BLEDevice->>BLEDevice: Emite status_update_signal("Conectando a ESP32...")
        BLEDevice->>+ESP32Device: self.client = BleakClient(dirección_mac)
        BLEDevice->>ESP32Device: await self.client.connect()
        ESP32Device-->>BLEDevice: Conexión establecida
        BLEDevice->>BLEDevice: self.is_connected = True
        BLEDevice->>ESP32Device: await self.client.start_notify(TX_CHAR_UUID, self._notification_handler)
        ESP32Device-->>BLEDevice: Notificaciones activadas
        BLEDevice->>BLEDevice: Emite connected_signal()
        BLEDevice->>BLEDevice: Emite status_update_signal("Conectado")
    else Dispositivo No Encontrado o Error de Conexión
        BLEDevice->>BLEDevice: Emite status_update_signal("Error: No se pudo conectar")
        BLEDevice->>BLEDevice: self.is_connected = False
    end
    deactivate ESP32Device
    deactivate BLEDevice
    deactivate AsyncioLoop
    deactivate MainWindow
```

#### 2.2.3. Proceso de Streaming de Datos (Inicio, Recepción y Detención)

Este diagrama muestra cómo se inicia el flujo de datos desde el ESP32, cómo se reciben y procesan los datos, y cómo se detiene el flujo.

```mermaid
sequenceDiagram
    participant User
    participant MainWindow
    participant BLEDevice
    participant AsyncioLoop as Bucle asyncio
    participant ESP32Device as ESP32 (Periférico BLE)

    User->>+MainWindow: Clic en botón "Iniciar Streaming"
    MainWindow->>MainWindow: start_stop_streaming()
    MainWindow->>+BLEDevice: self.ble_device.start_streaming()
    BLEDevice->>BLEDevice: start_streaming() (método síncrono)
    BLEDevice->>+AsyncioLoop: self.loop.create_task(self.start_streaming_async())
    AsyncioLoop->>+BLEDevice: Ejecuta start_streaming_async()
    alt Cliente Conectado y No en Streaming
        BLEDevice->>+ESP32Device: await self.client.write_gatt_char(RX_CHAR_UUID, b'start')
        ESP32Device-->>BLEDevice: Comando 'start' recibido
        BLEDevice->>BLEDevice: self.is_streaming = True
        BLEDevice->>BLEDevice: Emite streaming_started_signal()
        BLEDevice->>BLEDevice: Emite status_update_signal("Streaming iniciado")
    else Cliente No Conectado o Ya en Streaming
        BLEDevice->>BLEDevice: Emite status_update_signal("Error: No se puede iniciar streaming")
    end
    deactivate ESP32Device
    deactivate BLEDevice
    deactivate AsyncioLoop

    Note over ESP32Device, MainWindow: ESP32 comienza a enviar datos periódicamente...

    loop Recepción de Datos
        ESP32Device-->>+BLEDevice: Envía notificación BLE (datos ax, ay, az)
        BLEDevice->>BLEDevice: _notification_handler(sender, data)
        BLEDevice->>BLEDevice: (ax, ay, az) = struct.unpack('<fff', data)
        BLEDevice->>BLEDevice: Emite data_received_signal(ax, ay, az)
        BLEDevice-->>+MainWindow: data_received_signal(ax, ay, az)
        MainWindow->>MainWindow: handle_new_data(ax, ay, az)
        MainWindow->>MainWindow: Actualiza buffers de datos y gráficos
        deactivate MainWindow
        deactivate BLEDevice
    end

    User->>+MainWindow: Clic en botón "Detener Streaming"
    MainWindow->>MainWindow: start_stop_streaming() (ahora para detener)
    MainWindow->>+BLEDevice: self.ble_device.stop_streaming()
    BLEDevice->>BLEDevice: stop_streaming() (método síncrono)
    BLEDevice->>+AsyncioLoop: self.loop.create_task(self.stop_streaming_async())
    AsyncioLoop->>+BLEDevice: Ejecuta stop_streaming_async()
    alt Cliente Conectado y en Streaming
        BLEDevice->>+ESP32Device: await self.client.write_gatt_char(RX_CHAR_UUID, b'stop')
        ESP32Device-->>BLEDevice: Comando 'stop' recibido
        BLEDevice->>BLEDevice: self.is_streaming = False
        BLEDevice->>BLEDevice: Emite streaming_stopped_signal()
        BLEDevice->>BLEDevice: Emite status_update_signal("Streaming detenido")
    else Cliente No Conectado o No en Streaming
        BLEDevice->>BLEDevice: Emite status_update_signal("Error: No se puede detener streaming")
    end
    deactivate ESP32Device
    deactivate BLEDevice
    deactivate AsyncioLoop
    deactivate MainWindow
```

#### 2.2.4. Proceso de Desconexión BLE

Este diagrama detalla la secuencia de eventos cuando el usuario se desconecta del dispositivo BLE o cierra la aplicación.

```mermaid
sequenceDiagram
    participant User
    participant MainWindow
    participant BLEDevice
    participant AsyncioLoop as Bucle asyncio
    participant ESP32Device as ESP32 (Periférico BLE)

    alt Usuario Clic en "Desconectar" o Cierra Aplicación
        User->>+MainWindow: Clic en botón "Desconectar" (o evento de cierre de ventana)
        MainWindow->>MainWindow: disconnect_ble_device() (o closeEvent())
        MainWindow->>+BLEDevice: self.ble_device.disconnect()
        BLEDevice->>BLEDevice: disconnect() (método síncrono)
        BLEDevice->>+AsyncioLoop: self.loop.create_task(self.disconnect_async())
        AsyncioLoop->>+BLEDevice: Ejecuta disconnect_async()
        alt Cliente Conectado
            alt Streaming Activo
                BLEDevice->>BLEDevice: Llama internamente a stop_streaming_async()
                BLEDevice->>+ESP32Device: Envía comando 'stop'
                ESP32Device-->>BLEDevice: Comando 'stop' recibido
                BLEDevice->>BLEDevice: self.is_streaming = False
                deactivate ESP32Device
            end
            BLEDevice->>+ESP32Device: await self.client.stop_notify(TX_CHAR_UUID)
            ESP32Device-->>BLEDevice: Notificaciones detenidas
            BLEDevice->>ESP32Device: await self.client.disconnect()
            ESP32Device-->>BLEDevice: Desconexión completada
            BLEDevice->>BLEDevice: self.is_connected = False
            BLEDevice->>BLEDevice: self.client = None
            BLEDevice->>BLEDevice: Emite disconnected_signal()
            BLEDevice->>BLEDevice: Emite status_update_signal("Desconectado")
        else Cliente No Conectado
            BLEDevice->>BLEDevice: Emite status_update_signal("Ya desconectado")
        end
        deactivate ESP32Device
        deactivate BLEDevice
        deactivate AsyncioLoop
        MainWindow->>MainWindow: Actualiza GUI (desconectado)
        deactivate MainWindow
    end
```

### 2.3. Reconocimiento de Actividad Humana (HAR) en la Aplicación

La aplicación `esp32_mpu6050_monitor.py` no solo visualiza datos del acelerómetro, sino que también incluye un sistema completo para el Reconocimiento de Actividad Humana (HAR). Esto implica recolectar datos, extraer características, entrenar un modelo de clasificación y utilizarlo para predecir actividades en tiempo real.

#### 2.3.1. Recolección de Datos y Etiquetado para Entrenamiento

Este diagrama muestra cómo el usuario recolecta datos para una actividad específica, que luego se usarán para entrenar el modelo.

```mermaid
sequenceDiagram
    participant User
    participant MainWindowGUI as "MainWindow (GUI)"
    participant BLEDevice
    participant ESP32

    User->>MainWindowGUI: Selecciona Actividad (ej. "Caminar")
    User->>MainWindowGUI: Define No. Segmentos a Colectar
    User->>MainWindowGUI: Clic "Iniciar Recolección Entrenamiento"
    MainWindowGUI->>MainWindowGUI: is_collecting_training_data = true
    MainWindowGUI->>MainWindowGUI: current_training_activity = "Caminar"
    MainWindowGUI->>MainWindowGUI: collected_segments = 0
    MainWindowGUI->>MainWindowGUI: Limpia buffers de entrenamiento
    MainWindowGUI->>MainWindowGUI: Log("Iniciando recolección para 'Caminar'...")
    
    alt BLE No Conectado o Streaming Inactivo
        MainWindowGUI->>MainWindowGUI: Log("Error: Conectar y activar streaming.")
        MainWindowGUI->>MainWindowGUI: is_collecting_training_data = false
    else BLE Conectado y Streaming Activo
        Note over MainWindowGUI, ESP32: Usuario realiza la actividad "Caminar"
        loop Mientras recolecta y segmentos < objetivo
            ESP32-->>BLEDevice: Envía datos (ax, ay, az)
            BLEDevice-->>MainWindowGUI: data_received_signal(datos)
            MainWindowGUI->>MainWindowGUI: handle_new_data(datos)
            MainWindowGUI->>MainWindowGUI: Añade datos a buffer de segmento
            
            alt Segmento completo (FEATURE_WINDOW_SAMPLES)
                MainWindowGUI->>MainWindowGUI: Extrae segmento del buffer
                MainWindowGUI->>MainWindowGUI: Almacena segmento y etiqueta
                MainWindowGUI->>MainWindowGUI: collected_segments++
                MainWindowGUI->>MainWindowGUI: Log("Segmento X/Y recolectado.")
                MainWindowGUI->>MainWindowGUI: Desplaza buffer (superposición)
            end
        end

        alt Recolección Finalizada/Detenida
            User->>MainWindowGUI: (Opcional) Clic "Detener Recolección"
            MainWindowGUI->>MainWindowGUI: is_collecting_training_data = false
            MainWindowGUI->>MainWindowGUI: Log("Recolección de entrenamiento finalizada.")
            MainWindowGUI->>MainWindowGUI: (Opcional) Habilita "Guardar Datos"
        end
    end
```

#### 2.3.2. Extracción de Características

Una vez recolectados los segmentos de datos crudos, se extraen características significativas que el modelo de ML pueda utilizar.

```mermaid
sequenceDiagram
    participant MainWindowGUI as "MainWindow (GUI)"
    participant FeatureExtractor
    
    MainWindowGUI->>MainWindowGUI: (Después de recolectar datos o al cargar datos)
    MainWindowGUI->>FeatureExtractor: features_list = self.feature_extractor.extract_all_features(self.training_data_buffer, self.SAMPLE_RATE)
    FeatureExtractor->>FeatureExtractor: Para cada segmento en training_data_buffer:
    FeatureExtractor->>FeatureExtractor:  window_features = []
    FeatureExtractor->>FeatureExtractor:  time_features = extract_time_domain_features(segmento)
    FeatureExtractor->>FeatureExtractor:  freq_features = extract_frequency_domain_features(segmento, sample_rate)
    FeatureExtractor->>FeatureExtractor:  window_features.extend(time_features)
    FeatureExtractor->>FeatureExtractor:  window_features.extend(freq_features)
    FeatureExtractor-->>MainWindowGUI: Devuelve lista de vectores de características (features_list)
    MainWindowGUI->>MainWindowGUI: self.X_train = np.array(features_list)
    MainWindowGUI->>MainWindowGUI: self.y_train = np.array(self.training_labels_buffer)
    MainWindowGUI->>MainWindowGUI: Log("Características extraídas de los datos recolectados.")
```

#### 2.3.3. Entrenamiento del Modelo de Clasificación

Con las características extraídas (X_train) y sus etiquetas (y_train), se entrena un modelo de clasificación.

```mermaid
sequenceDiagram
    participant User
    participant MainWindowGUI as "MainWindow (GUI)"
    participant ActivityClassifier
    participant StandardScaler as "sklearn.preprocessing.StandardScaler"

    User->>MainWindowGUI: Selecciona tipo de modelo (SVM, RandomForest, etc.) en ComboBox
    User->>MainWindowGUI: Ajusta parámetros del modelo (si aplica)
    User->>MainWindowGUI: Clic en botón "Entrenar Modelo"
    
    alt Datos de Entrenamiento No Disponibles
        MainWindowGUI->>MainWindowGUI: Log("Error: Recolecte o cargue datos de entrenamiento primero.")
    else Datos de Entrenamiento Disponibles (self.X_train, self.y_train)
        MainWindowGUI->>MainWindowGUI: Log(f"Iniciando entrenamiento del modelo {self.classifier.model_type}...")
        MainWindowGUI->>ActivityClassifier: self.classifier.model_type = tipo_seleccionado
        MainWindowGUI->>ActivityClassifier: self.classifier.train(self.X_train, self.y_train)
        ActivityClassifier->>StandardScaler: scaler = StandardScaler()
        ActivityClassifier->>StandardScaler: X_scaled = scaler.fit_transform(X_train)
        ActivityClassifier->>ActivityClassifier: Crea instancia del modelo sklearn (ej. SVC())
        ActivityClassifier->>ActivityClassifier: model.fit(X_scaled, y_train)
        ActivityClassifier->>ActivityClassifier: self.scaler = scaler
        ActivityClassifier->>ActivityClassifier: self.model = model
        ActivityClassifier-->>MainWindowGUI: Entrenamiento completado
        MainWindowGUI->>MainWindowGUI: Log("Modelo entrenado exitosamente.")
        MainWindowGUI->>MainWindowGUI: (Opcional) Evalúa el modelo (accuracy, confusion matrix) y muestra resultados.
        MainWindowGUI->>MainWindowGUI: Habilita botón "Guardar Modelo"
    end
```

#### 2.3.4. Guardado y Carga del Modelo Entrenado

Los modelos entrenados pueden guardarse para uso futuro y cargarse al iniciar la aplicación.

```mermaid
sequenceDiagram
    participant User
    participant MainWindowGUI as "MainWindow (GUI)"
    participant ActivityClassifier

    alt Guardar Modelo
        User->>MainWindowGUI: Clic en "Guardar Modelo"
        MainWindowGUI->>MainWindowGUI: Abre diálogo para seleccionar ruta de archivo (ej. "activity_model.pkl")
        alt Usuario selecciona archivo y confirma
            MainWindowGUI->>ActivityClassifier: self.classifier.save(ruta_archivo)
            ActivityClassifier->>ActivityClassifier: Abre archivo en modo binario escritura ('wb')
            ActivityClassifier->>ActivityClassifier: pickle.dump({'model': self.model, 'scaler': self.scaler, 'model_type': self.model_type}, file)
            ActivityClassifier-->>MainWindowGUI: Modelo guardado
            MainWindowGUI->>MainWindowGUI: Log(f"Modelo guardado en {ruta_archivo}")
        else Usuario cancela
            MainWindowGUI->>MainWindowGUI: Log("Guardado de modelo cancelado.")
        end
    end

    alt Cargar Modelo (al inicio o manualmente)
        User->>MainWindowGUI: (Opcional) Clic en "Cargar Modelo"
        MainWindowGUI->>MainWindowGUI: (Al inicio) Intenta cargar "activity_model.pkl" por defecto
        MainWindowGUI->>MainWindowGUI: (Manual) Abre diálogo para seleccionar archivo de modelo
        alt Usuario selecciona archivo y confirma (o archivo por defecto existe)
            MainWindowGUI->>ActivityClassifier: self.classifier.load(ruta_archivo)
            ActivityClassifier->>ActivityClassifier: Abre archivo en modo binario lectura ('rb')
            ActivityClassifier->>ActivityClassifier: data = pickle.load(file)
            ActivityClassifier->>ActivityClassifier: self.model = data['model']
            ActivityClassifier->>ActivityClassifier: self.scaler = data['scaler']
            ActivityClassifier->>ActivityClassifier: self.model_type = data.get('model_type', 'desconocido')
            ActivityClassifier-->>MainWindowGUI: Modelo cargado
            MainWindowGUI->>MainWindowGUI: Log(f"Modelo '{self.classifier.model_type}' cargado desde {ruta_archivo}")
        else Archivo no existe o usuario cancela
            MainWindowGUI->>MainWindowGUI: Log("No se cargó el modelo.")
        end
    end
```

#### 2.3.5. Predicción/Clasificación en Tiempo Real

Una vez que hay un modelo cargado o entrenado, la aplicación puede clasificar la actividad en tiempo real a medida que llegan nuevos datos del sensor.

```mermaid
sequenceDiagram
    participant ESP32
    participant BLEDevice
    participant MainWindowGUI as "MainWindow (GUI)"
    participant FeatureExtractor
    participant ActivityClassifier

    Note over ESP32, MainWindowGUI: Streaming de datos activo y modelo HAR cargado/entrenado.
    
    loop Actualización Periódica (Plot Timer)
        MainWindowGUI->>MainWindowGUI: update_plots_and_classification()
        
        alt Suficientes datos en buffer para una ventana de características
            MainWindowGUI->>MainWindowGUI: Prepara ventana de datos (ej. self.ax_buffer, self.ay_buffer, self.az_buffer)
            Note right of MainWindowGUI: Usa los últimos FEATURE_WINDOW_SAMPLES de los buffers principales.
            MainWindowGUI->>FeatureExtractor: current_features = self.feature_extractor.extract_all_features([[ax_win], [ay_win], [az_win]], self.SAMPLE_RATE)
            FeatureExtractor-->>MainWindowGUI: Devuelve vector de características
            
            alt Características extraídas y modelo disponible
                MainWindowGUI->>ActivityClassifier: predicted_activity = self.classifier.predict(current_features)
                ActivityClassifier->>ActivityClassifier: self.scaler.transform(current_features)
                ActivityClassifier->>ActivityClassifier: self.model.predict(scaled_features)
                ActivityClassifier-->>MainWindowGUI: Devuelve etiqueta de actividad predicha (ej. "Caminar")
                MainWindowGUI->>MainWindowGUI: Actualiza etiqueta de "Actividad Predicha" en la GUI
            else Modelo no disponible o error
                MainWindowGUI->>MainWindowGUI: Muestra "N/A" o "Error" en etiqueta de actividad.
            end
        end
        MainWindowGUI->>MainWindowGUI: Actualiza gráficos (datos crudos, FFT)
    end

    ESP32-->>BLEDevice: Envía datos (ax, ay, az)
    BLEDevice-->>MainWindowGUI: data_received_signal(ax, ay, az)
    MainWindowGUI->>MainWindowGUI: handle_new_data(ax, ay, az)
    MainWindowGUI->>MainWindowGUI: Añade datos a ax_buffer, ay_buffer, az_buffer (usados por el loop de actualización)
```

---


## 3. Contribuyendo al Proyecto 

Si quieres que otros contribuyan:

*   **Forking (Bifurcación):** Los colaboradores pueden "bifurcar" tu repositorio en GitHub, creando su propia copia.
*   **Branches (Ramas):** Crear ramas para nuevas características o correcciones (`git checkout -b nombre-rama`).
*   **Pull Requests (Solicitudes de Extracción):** Una vez que los cambios están listos en su rama, pueden enviar un "Pull Request" al repositorio original para que revises e integres sus cambios.

---

## 4. Conclusión

Este tutorial te ha proporcionado las bases para gestionar el código del proyecto con Git y para entender la arquitectura y el funcionamiento de la interfaz `esp32_mpu6050_monitor.py`, con un enfoque en la comunicación BLE gestionada por la clase `BLEDevice`. ¡Esperamos que sea de gran utilidad para tus experimentos de visualización de datos de sensores!

No dudes en expandir este tutorial con más detalles específicos de tu implementación o problemas comunes que encuentres.
