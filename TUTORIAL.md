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

### 2.2. Arquitectura de `esp32_mpu6050_monitor.py`

La aplicación se compone de dos partes principales que trabajan juntas:
1.  **`MainWindow` (Clase de PyQt5):** Gestiona la interfaz gráfica, incluyendo botones, etiquetas de estado y los gráficos. Inicia las operaciones BLE y actualiza la GUI basándose en las señales recibidas desde `BLEDevice`.
2.  **`BLEDevice` (Clase de QObject):** Maneja toda la lógica de comunicación BLE de forma asíncrona. Se ejecuta en un bucle de eventos `asyncio` separado para no bloquear la GUI. Emite señales PyQt para notificar a `MainWindow` sobre eventos BLE (conexión, desconexión, datos recibidos, etc.).

*   **Diagrama Conceptual 1: Arquitectura de `esp32_mpu6050_monitor.py` (draw.io)**
    *   **Caja Izquierda: "ESP32 con MPU6050"**
        *   Sub-componentes: Sensor MPU6050, Microcontrolador ESP32, Módulo BLE.
        *   Funciones: Leer sensor, Empaquetar datos, Transmitir vía BLE.
    *   **Flecha de Conexión: "Bluetooth Low Energy (BLE)"**
    *   **Caja Derecha: "PC con `esp32_mpu6050_monitor.py`"**
        *   **Sub-Caja 1: `MainWindow` (GUI - PyQt5)**
            *   Elementos: Botones (Conectar, Iniciar/Detener Stream), Gráficos (pyqtgraph), Etiqueta de Estado.
            *   Responsabilidades: Interacción con el usuario, Iniciar/Detener operaciones BLE, Actualizar gráficos y estado.
        *   **Sub-Caja 2: `BLEDevice` (Lógica BLE - Bleak, asyncio)**
            *   Responsabilidades: Escanear dispositivos, Conectar/Desconectar, Manejar notificaciones de datos (`_notification_handler`), Enviar comandos (start/stop stream).
            *   Comunicación con `MainWindow`: A través de Señales y Slots de PyQt (`connected_signal`, `data_received_signal`, etc.).
        *   **Sub-Caja 3: Bucle de Eventos `asyncio`**
            *   Entorno donde se ejecutan las tareas asíncronas de `BLEDevice`.
            *   `MainWindow` gestiona la ejecución de este bucle de forma no bloqueante.

### 2.3. La Clase `BLEDevice` a Detalle

La clase `BLEDevice` es el corazón de la comunicación BLE. Aquí un desglose de sus métodos y flujo:

*   **Inicialización (`__init__`)**
    *   Recibe el bucle de eventos `asyncio` que será utilizado para las operaciones BLE.
    *   Inicializa variables de estado como `client` (el cliente Bleak), `is_connected`, `is_streaming`, y `device_address`.
    *   Define señales PyQt (`connected_signal`, `disconnected_signal`, `status_update_signal`, `data_received_signal`, `streaming_started_signal`, `streaming_stopped_signal`) para comunicarse con la `MainWindow`.

*   **Escaneo (`_scan_for_device`) - Asíncrono**
    *   Utiliza `BleakScanner.discover()` para encontrar dispositivos BLE cercanos.
    *   Busca un dispositivo cuyo nombre coincida con `ESP32_DEVICE_NAME`.
    *   Si lo encuentra, guarda su dirección MAC y la devuelve.
    *   Emite `status_update_signal` para informar sobre el progreso.

*   **Conexión (`connect_async`) - Asíncrono**
    1.  Si no hay una dirección de dispositivo guardada, llama a `_scan_for_device()`.
    2.  Crea una instancia de `BleakClient` con la dirección del dispositivo y el bucle `asyncio`.
    3.  Intenta conectar con `await self.client.connect()`.
    4.  Si la conexión es exitosa:
        *   Actualiza `is_connected` a `True`.
        *   Emite `connected_signal`.
        *   Automáticamente inicia la escucha de notificaciones en la característica `TX_CHAR_UUID` (donde el ESP32 envía datos), asignando `_notification_handler` como el callback.
    5.  Maneja excepciones y actualiza el estado/señales en caso de fallo.

*   **Manejador de Notificaciones (`_notification_handler`) - Callback Síncrono**
    *   Este método es llamado por la librería `Bleak` cada vez que se recibe una notificación (datos) del ESP32 en la característica `TX_CHAR_UUID`.
    *   Recibe `sender` (identificador de la característica) y `data` (los bytes recibidos).
    *   Verifica que la longitud de `data` sea la esperada (12 bytes para 3 floats).
    *   Desempaqueta los bytes usando `struct.unpack('<fff', data)` para obtener los tres valores flotantes (ax, ay, az). El formato `'<fff'` indica little-endian.
    *   Emite `data_received_signal` con los valores ax, ay, az. Esta señal será capturada por `MainWindow` para actualizar los gráficos.

*   **Inicio de Streaming (`start_streaming_async`) - Asíncrono**
    *   Verifica que el cliente esté conectado y que el streaming no esté ya activo.
    *   Escribe el comando `b'start'` en la característica `RX_CHAR_UUID` del ESP32. Esto le indica al ESP32 que comience a enviar datos.
    *   Actualiza `is_streaming` a `True` y emite `streaming_started_signal`.

*   **Detención de Streaming (`stop_streaming_async`) - Asíncrono**
    *   Verifica que el cliente esté conectado y que el streaming esté activo.
    *   Escribe el comando `b'stop'` en la característica `RX_CHAR_UUID` del ESP32.
    *   Actualiza `is_streaming` a `False` y emite `streaming_stopped_signal`.

*   **Desconexión (`disconnect_async`) - Asíncrono**
    1.  Si el streaming está activo, primero llama a `stop_streaming_async()`.
    2.  Detiene las notificaciones en `TX_CHAR_UUID` con `await self.client.stop_notify()`.
    3.  Desconecta el cliente con `await self.client.disconnect()`.
    4.  Actualiza los estados `is_connected`, `is_streaming` a `False`.
    5.  Emite `disconnected_signal`.
    6.  Limpia la instancia de `self.client`.

*   **Métodos Públicos Síncronos (`connect`, `disconnect`, `start_streaming`, `stop_streaming`)**
    *   Estos son los métodos que la `MainWindow` (que corre en el hilo principal de Qt) llama para iniciar las operaciones BLE.
    *   Internamente, estos métodos usan `self.loop.create_task()` para programar la ejecución de sus contrapartes asíncronas (`connect_async`, `disconnect_async`, etc.) en el bucle de eventos `asyncio`. Esto es crucial para no bloquear la GUI.

*   **Diagrama Conceptual 2: Flujo de Conexión y Recepción de Datos BLE (draw.io)**
    *   **Columna 1: `MainWindow` (Hilo GUI Qt)**
        *   Usuario hace clic en "Conectar".
        *   `MainWindow.connect_ble_device()` es llamado.
        *   Llama a `BLEDevice.connect()`.
    *   **Columna 2: `BLEDevice` (Métodos Síncronos de Interfaz)**
        *   `BLEDevice.connect()`:
            *   `loop.create_task(self.connect_async())`.
    *   **Columna 3: `BLEDevice` (Tareas Asíncronas - Bucle `asyncio`)**
        *   `connect_async()`:
            *   `_scan_for_device()` (si es necesario).
            *   `BleakClient.connect()`.
            *   `BleakClient.start_notify(TX_CHAR_UUID, _notification_handler)`.
            *   Emite `connected_signal` (hacia `MainWindow`).
        *   Cuando el ESP32 envía datos:
            *   `_notification_handler(sender, data)` es llamado por Bleak.
            *   Desempaqueta datos.
            *   Emite `data_received_signal(ax, ay, az)` (hacia `MainWindow`).
    *   **Columna 4: `MainWindow` (Slots/Manejadores de Señales - Hilo GUI Qt)**
        *   `MainWindow.on_ble_connected()`: Actualiza estado de la GUI.
        *   `MainWindow.handle_new_data(ax, ay, az)`:
            *   Añade datos a buffers.
            *   Dispara actualización de gráficos (ej. con `QTimer`).

### 2.4. Flujo de Ejecución de la Aplicación

1.  El usuario ejecuta `esp32_mpu6050_monitor.py`.
2.  Se crea una instancia de `MainWindow`.
3.  `MainWindow` crea una instancia de `BLEDevice`, pasándole el bucle de eventos `asyncio`.
4.  `MainWindow` inicia el bucle `asyncio` de forma no bloqueante (usando un `QTimer` para llamar periódicamente a `loop.run_until_complete(asyncio.sleep(0))` o similar, como se implementó en `run_async_tasks`).
5.  El usuario hace clic en el botón "Conectar" en la GUI.
6.  `MainWindow` llama a `ble_device.connect()`.
7.  `BLEDevice.connect()` programa `connect_async()` en el bucle `asyncio`.
8.  `connect_async()` escanea, se conecta e inicia notificaciones. Las señales actualizan la GUI.
9.  Cuando el ESP32 envía datos, `_notification_handler` los procesa y emite `data_received_signal`.
10. `MainWindow` recibe esta señal, actualiza sus buffers de datos y redibuja los gráficos.
11. El usuario puede hacer clic en "Iniciar/Detener Streaming" para enviar comandos al ESP32 a través de `BLEDevice.start_streaming()` o `BLEDevice.stop_streaming()`.
12. Al cerrar la aplicación o hacer clic en "Desconectar", se llama a `ble_device.disconnect()` para cerrar la conexión BLE limpiamente.

---

## 3. Contribuyendo al Proyecto (Opcional, si es un equipo)

Si quieres que otros contribuyan:

*   **Forking (Bifurcación):** Los colaboradores pueden "bifurcar" tu repositorio en GitHub, creando su propia copia.
*   **Branches (Ramas):** Crear ramas para nuevas características o correcciones (`git checkout -b nombre-rama`).
*   **Pull Requests (Solicitudes de Extracción):** Una vez que los cambios están listos en su rama, pueden enviar un "Pull Request" al repositorio original para que revises e integres sus cambios.

---

## 4. Conclusión

Este tutorial te ha proporcionado las bases para gestionar el código del proyecto con Git y para entender la arquitectura y el funcionamiento de la interfaz `esp32_mpu6050_monitor.py`, con un enfoque en la comunicación BLE gestionada por la clase `BLEDevice`. ¡Esperamos que sea de gran utilidad para tus experimentos de visualización de datos de sensores!

No dudes en expandir este tutorial con más detalles específicos de tu implementación o problemas comunes que encuentres.
