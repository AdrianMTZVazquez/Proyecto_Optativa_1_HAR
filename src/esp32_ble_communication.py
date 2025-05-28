import sys
import asyncio
import platform
from bleak import BleakScanner, BleakClient
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, 
                           QLabel, QComboBox, QWidget, QMessageBox, QHBoxLayout, QTextEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject, pyqtSlot
import numpy as np
import pyqtgraph as pg
from struct import unpack
import struct
import logging

# --- BEGIN ESP32 Configuration ---
# !!POR FAVOR, ACTUALIZA ESTOS VALORES CON LOS DE TU ESP32!!
ESP32_SERVICE_UUID = "0000xxxx-0000-1000-8000-00805f9b34fb"  # Placeholder
ESP32_RX_CHAR_UUID = "0000yyyy-0000-1000-8000-00805f9b34fb"  # Característica para ENVIAR datos al ESP32 (ESP32 recibe)
ESP32_TX_CHAR_UUID = "0000zzzz-0000-1000-8000-00805f9b34fb"  # Característica para RECIBIR datos del ESP32 (ESP32 envía)
# --- END ESP32 Configuration ---

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    stream=sys.stdout
)
logging.debug("Root logger configured for DEBUG level output to stdout.")

class ESP32BLEDevice(QObject):
    """Clase para manejar la comunicación BLE con un ESP32."""
    data_received = pyqtSignal(list)  # Adaptar según el tipo de datos que envíe el ESP32
    status_update = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)
    
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.client = None
        # Eventos para sincronización si el ESP32 requiere un handshake específico
        # self.handshake_step1_done_event = asyncio.Event()
        # self.handshake_step2_done_event = asyncio.Event()
        self.is_handler_active = False
    
    async def connect_to_device(self, address):
        """Conecta al dispositivo BLE ESP32."""
        logging.info(f"Intentando conectar a {address} (ESP32)")
        try:
            self.client = BleakClient(address, loop=self.loop)
            await self.client.connect()
            logging.info(f"Conectado a {self.client.address}. Obteniendo servicios...")
            self.status_update.emit(f"Conectado a {self.client.address}. Obteniendo servicios...")
            
            services = await self.client.get_services()
            logging.info(f"Servicios disponibles en {self.client.address}:")
            esp32_service_found = False
            for service in services:
                logging.info(f"  Servicio UUID descubierto: {service.uuid.lower()}")
                if service.uuid.lower() == ESP32_SERVICE_UUID.lower():
                    esp32_service_found = True
                    logging.info(f"    Características para el servicio {service.uuid}:")
                    for char in service.characteristics:
                        logging.info(f"      Característica: {char.uuid}, Propiedades: {char.properties}")
                        if char.uuid.lower() == ESP32_RX_CHAR_UUID.lower():
                            logging.info(f"        Encontrada RX Característica (para enviar a ESP32): {char.uuid}")
                        if char.uuid.lower() == ESP32_TX_CHAR_UUID.lower():
                            logging.info(f"        Encontrada TX Característica (para recibir de ESP32): {char.uuid}")
            
            if not esp32_service_found:
                logging.error(f"Servicio ESP32 ({ESP32_SERVICE_UUID}) no encontrado.")
                self.status_update.emit("Error: Servicio ESP32 no encontrado.")
                await self.disconnect_device()
                return False

            logging.info(f"Servicio ESP32 ({ESP32_SERVICE_UUID}) encontrado.")
            self.connection_changed.emit(True)
            return True
        except Exception as e:
            logging.error(f"Error inesperado al conectar al ESP32: {e}")
            self.status_update.emit(f"Error inesperado al conectar: {e}")
        self.connection_changed.emit(False)
        return False

    async def disconnect_device(self):
        """Desconecta el dispositivo BLE ESP32."""
        if self.client and self.client.is_connected:
            logging.info(f"Desconectando de {self.client.address}...")
            try:
                # Si se suscribió a notificaciones, detenerlas primero
                if self.is_handler_active and ESP32_TX_CHAR_UUID:
                    try:
                        await self.client.stop_notify(ESP32_TX_CHAR_UUID)
                        logging.info("Notificaciones detenidas.")
                    except Exception as e:
                        logging.warning(f"Error al detener notificaciones: {e}")
                await self.client.disconnect()
                logging.info("Desconectado.")
            except asyncio.TimeoutError:
                logging.warning("Timeout durante la desconexión.")
            except Exception as e:
                logging.error(f"Error durante la desconexión: {e}")
            finally:
                self.client = None
                self.is_handler_active = False
                self.connection_changed.emit(False)
                self.status_update.emit("Desconectado")
        else:
            logging.info("El cliente no está conectado o no existe.")

    # --- INICIO LÓGICA DE HANDSHAKE ESPEĆIFICA DEL ESP32 (SI ES NECESARIA) ---
    # async def perform_esp32_handshake(self):
    #     """Realiza el handshake específico del ESP32 después de la conexión."""
    #     try:
    #         logging.info("Iniciando handshake con ESP32...")
    #         # Ejemplo: Enviar un comando inicial al ESP32
    #         # initial_command = b'\x01' # Comando de ejemplo
    #         # await self.client.write_gatt_char(ESP32_RX_CHAR_UUID, initial_command)
    #         # logging.info(f"Comando inicial enviado al ESP32.")
    #         
    #         # Ejemplo: Esperar una respuesta específica o confirmación
    #         # await asyncio.wait_for(self.handshake_step1_done_event.wait(), timeout=5.0)
    #         # logging.info("Respuesta de handshake recibida del ESP32.")
    #         
    #         # Aquí iría la lógica para configurar el ESP32 para enviar datos
    #         # Por ejemplo, escribir en otra característica para habilitar notificaciones de sensores
    #         # enable_sensor_command = b'\x02\x01' # Comando de ejemplo para habilitar sensor
    #         # await self.client.write_gatt_char(ESP32_RX_CHAR_UUID, enable_sensor_command)
    #         # logging.info("Comando para habilitar sensor enviado al ESP32.")

    #         # await asyncio.wait_for(self.handshake_step2_done_event.wait(), timeout=5.0)
    #         # logging.info("ESP32 configurado para enviar datos.")
    #         self.status_update.emit("Handshake con ESP32 completado (si aplica).")
    #         return True
    #     except asyncio.TimeoutError:
    #         logging.error("Timeout durante el handshake con ESP32.")
    #         self.status_update.emit("Error: Timeout en handshake con ESP32.")
    #         return False
    #     except Exception as e:
    #         logging.error(f"Error en el handshake con ESP32: {e}")
    #         self.status_update.emit(f"Error en handshake con ESP32: {e}")
    #         return False
    # --- FIN LÓGICA DE HANDSHAKE ESPEĆIFICA DEL ESP32 ---

    def _notification_handler(self, sender, data):
        """Maneja las notificaciones de datos del ESP32."""
        logging.debug(f"Notificación recibida de {sender}: {data.hex()}")
        # Aquí debes parsear los 'data' según el formato que envíe tu ESP32
        # Ejemplo: si el ESP32 envía 3 valores flotantes de 4 bytes cada uno
        # try:
        #     if len(data) == 12: # 3 floats * 4 bytes/float
        #         values = list(unpack('<fff', data)) # '<fff' = 3 little-endian floats
        #         logging.info(f"Datos parseados del ESP32: {values}")
        #         self.data_received.emit(values) # Emitir como lista
        #     else:
        #         logging.warning(f"Datos recibidos con longitud inesperada: {len(data)} bytes")
        # except Exception as e:
        #     logging.error(f"Error al parsear datos del ESP32: {e} - Datos: {data.hex()}")
        
        # Si el ESP32 envía respuestas a comandos de handshake por aquí:
        # if data == b'expected_handshake_response_1':
        #     self.handshake_step1_done_event.set()
        # elif data == b'expected_handshake_response_2':
        #     self.handshake_step2_done_event.set()
        pass # Implementar el parseo de datos del ESP32

    async def connect_and_subscribe(self, address):
        """Conecta, (opcionalmente realiza handshake) y se suscribe a notificaciones."""
        self.status_update.emit(f"Conectando a ESP32 en {address}...")
        if not await self.connect_to_device(address):
            self.status_update.emit("Fallo al conectar con ESP32.")
            return False
        
        # self.handshake_step1_done_event.clear()
        # self.handshake_step2_done_event.clear()

        # Opcional: Realizar handshake si es necesario para tu ESP32
        # if not await self.perform_esp32_handshake():
        #     self.status_update.emit("Fallo en el handshake con ESP32.")
        #     await self.disconnect_device()
        #     return False
        
        self.status_update.emit("Suscribiéndose a notificaciones del ESP32...")
        try:
            await self.client.start_notify(ESP32_TX_CHAR_UUID, self._notification_handler)
            self.is_handler_active = True
            logging.info(f"Suscrito a notificaciones de {ESP32_TX_CHAR_UUID}")
            self.status_update.emit("Suscrito a notificaciones del ESP32.")
            return True
        except Exception as e:
            logging.error(f"Error al suscribirse a notificaciones del ESP32: {e}")
            self.status_update.emit(f"Error al suscribirse: {e}")
            await self.disconnect_device()
            return False

class BLEScannerThread(QThread):
    """Hilo para escanear dispositivos BLE."""
    devices_found = pyqtSignal(list) # Lista de tuplas (nombre, dirección)
    scan_finished = pyqtSignal()

    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.running = True

    async def _scan(self):
        logging.info("Iniciando escaneo BLE...")
        try:
            devices = await BleakScanner.discover(timeout=5.0, loop=self.loop)
            found_devices = []
            for device in devices:
                if device.name:
                    logging.info(f"Dispositivo encontrado: {device.name} ({device.address})")
                    found_devices.append((device.name, device.address))
            self.devices_found.emit(found_devices)
        except Exception as e:
            logging.error(f"Error durante el escaneo: {e}")
            self.devices_found.emit([]) # Emitir lista vacía en caso de error
        finally:
            logging.info("Escaneo BLE finalizado.")
            self.scan_finished.emit()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._scan())
    
    def stop(self):
        self.running = False

class ESP32App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ble_device = ESP32BLEDevice(asyncio.new_event_loop())
        self.scan_thread = None
        self.connection_thread = None
        self.setWindowTitle("ESP32 BLE Controller")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Controles
        self.controls_layout = QHBoxLayout()
        self.scan_button = QPushButton("Escanear Dispositivos")
        self.scan_button.clicked.connect(self.start_scan)
        self.controls_layout.addWidget(self.scan_button)

        self.device_combo = QComboBox()
        self.controls_layout.addWidget(self.device_combo)

        self.connect_button = QPushButton("Conectar")
        self.connect_button.clicked.connect(self.toggle_connection)
        self.connect_button.setEnabled(False)
        self.controls_layout.addWidget(self.connect_button)
        self.layout.addLayout(self.controls_layout)

        # Consola de Log
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.layout.addWidget(self.log_console)

        # Gráfico (opcional, adaptar según datos del ESP32)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("Datos del ESP32 (Ej: Sensor)")
        self.plot_widget.addLegend()
        # Ejemplo con 3 curvas, adaptar según los datos que envíe el ESP32
        self.curve_x = self.plot_widget.plot(pen='r', name='Dato X')
        self.curve_y = self.plot_widget.plot(pen='g', name='Dato Y')
        self.curve_z = self.plot_widget.plot(pen='b', name='Dato Z')
        self.data_buffer_size = 100 # Mostrar los últimos 100 puntos
        self.data_x = np.zeros(self.data_buffer_size)
        self.data_y = np.zeros(self.data_buffer_size)
        self.data_z = np.zeros(self.data_buffer_size)

        # Conectar señales
        self.ble_device.status_update.connect(self.update_status)
        self.ble_device.data_received.connect(self.on_data_received)
        self.ble_device.connection_changed.connect(self.on_connection_changed)

        # Timer para actualizar UI (si es necesario para el gráfico)
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_plot_ui)
        self.ui_update_timer.start(50) # Actualizar cada 50ms

        self.log("Aplicación ESP32 BLE Controller iniciada.")
        self.update_status("Listo para escanear.")

    def log(self, message):
        self.log_console.append(message)
        logging.info(message) # También al logger general

    def update_status(self, message):
        self.log(f"Estado: {message}")

    def start_scan(self):
        self.log("Iniciando escaneo...")
        self.scan_button.setEnabled(False)
        self.device_combo.clear()
        self.connect_button.setEnabled(False)
        # Asegurarse de que el loop de eventos para el scanner esté configurado correctamente
        scan_loop = asyncio.new_event_loop()
        self.scan_thread = BLEScannerThread(scan_loop)
        self.scan_thread.devices_found.connect(self.on_devices_found)
        self.scan_thread.scan_finished.connect(self.on_scan_finished)
        self.scan_thread.start()

    def on_scan_finished(self):
        self.log("Escaneo finalizado.")
        self.scan_button.setEnabled(True)
        if self.device_combo.count() > 0:
            self.connect_button.setEnabled(True)
        else:
            self.log("No se encontraron dispositivos.")

    def on_devices_found(self, devices):
        self.log(f"Dispositivos encontrados: {len(devices)}")
        for name, address in devices:
            self.device_combo.addItem(f"{name} ({address})", address)
        if devices:
            self.connect_button.setEnabled(True)

    def toggle_connection(self):
        if self.ble_device.client and self.ble_device.client.is_connected:
            self.disconnect_device()
        else:
            self.connect_device()

    def disconnect_device(self):
        self.log("Intentando desconectar...")
        # La desconexión ahora es manejada dentro de ESP32BLEDevice
        # y es asíncrona, por lo que se llama desde un hilo o tarea asyncio
        if self.ble_device.client:
            # Crear un nuevo loop para la tarea de desconexión si no estamos en un contexto asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.ble_device.disconnect_device())
            except Exception as e:
                self.log(f"Error al ejecutar desconexión: {e}")
            finally:
                loop.close()
        else:
            self.log("No hay cliente para desconectar.")
        self.update_status("Desconectado")
        self.connect_button.setText("Conectar")

    def connect_device(self):
        if self.device_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Advertencia", "Selecciona un dispositivo primero.")
            return

        address = self.device_combo.currentData()
        self.log(f"Intentando conectar a {self.device_combo.currentText()}...")
        self.connect_button.setEnabled(False)

        # Usar un hilo para la conexión para no bloquear la GUI
        self.connection_thread = ConnectionActionThread(self.ble_device, address, action_type="connect")
        self.connection_thread.action_result.connect(self.on_connection_action_result)
        self.connection_thread.start()

    def on_connection_action_result(self, success):
        self.connect_button.setEnabled(True)
        if success:
            self.log("Conexión y suscripción exitosas (o handshake si aplica).")
            self.connect_button.setText("Desconectar")
        else:
            self.log("Fallo en la conexión/suscripción (o handshake si aplica).")
            self.connect_button.setText("Conectar")
            # Asegurarse de que el estado de conexión se actualice si falla
            self.ble_device.connection_changed.emit(False)

    def on_connection_changed(self, connected):
        if connected:
            self.update_status("Conectado al ESP32")
            self.connect_button.setText("Desconectar")
            self.connect_button.setEnabled(True)
        else:
            self.update_status("Desconectado del ESP32")
            self.connect_button.setText("Conectar")
            self.connect_button.setEnabled(True) # Habilitar para intentar reconectar
            # Limpiar datos del gráfico al desconectar
            self.data_x.fill(0)
            self.data_y.fill(0)
            self.data_z.fill(0)
            self.update_plot_ui()

    def on_data_received(self, data_list):
        """Procesa los datos recibidos del ESP32."""
        # Asumimos que data_list es una lista de números (ej. [x, y, z] o similar)
        # Adaptar según la cantidad y tipo de datos que envíe tu ESP32
        try:
            if len(data_list) >= 1:
                self.data_x = np.roll(self.data_x, -1)
                self.data_x[-1] = data_list[0]
            if len(data_list) >= 2:
                self.data_y = np.roll(self.data_y, -1)
                self.data_y[-1] = data_list[1]
            if len(data_list) >= 3:
                self.data_z = np.roll(self.data_z, -1)
                self.data_z[-1] = data_list[2]
            # Añadir más si el ESP32 envía más de 3 valores

        except Exception as e:
            self.log(f"Error procesando datos del ESP32: {e}")
    
    def update_plot_ui(self):
        """Actualiza el gráfico en la interfaz de usuario."""
        self.curve_x.setData(self.data_x)
        self.curve_y.setData(self.data_y)
        self.curve_z.setData(self.data_z)
    
    def closeEvent(self, event):
        self.log("Cerrando aplicación...")
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.stop()
            self.scan_thread.quit()
            self.scan_thread.wait()
            
        if self.connection_thread and self.connection_thread.isRunning():
            # No hay un método stop directo, pero la finalización del hilo debería ser manejada
            self.connection_thread.quit()
            self.connection_thread.wait()
            
        # Desconectar el dispositivo BLE de forma segura
        if self.ble_device.client and self.ble_device.client.is_connected:
            self.log("Asegurando desconexión del dispositivo BLE...")
            # Crear un nuevo loop para la tarea de desconexión
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.ble_device.disconnect_device())
                self.log("Dispositivo desconectado al cerrar.")
            except Exception as e:
                self.log(f"Error al desconectar al cerrar: {e}")
            finally:
                loop.close()
        
        self.ui_update_timer.stop()
        event.accept()

class ConnectionActionThread(QThread):
    """Hilo para manejar acciones de conexión/suscripción BLE de forma asíncrona."""
    action_result = pyqtSignal(bool)  # True si éxito, False si fallo
    
    def __init__(self, ble_device, address_or_config, action_type="connect"):
        super().__init__()
        self.ble_device = ble_device
        self.address_or_config = address_or_config # Puede ser address para conectar
        self.action_type = action_type
    
    def run(self):
        # Cada hilo necesita su propio loop de eventos asyncio si no se pasa uno existente
        if platform.system() == "Windows" and sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        success = False
        try:
            if self.action_type == "connect":
                address = self.address_or_config
                success = loop.run_until_complete(self.ble_device.connect_and_subscribe(address))
            # Se podrían añadir otros tipos de acciones aquí si es necesario
            # elif self.action_type == "specific_command":
            #     command_data = self.address_or_config
            #     success = loop.run_until_complete(self.ble_device.send_command_to_esp32(command_data))
            
            self.action_result.emit(success)
        except Exception as e:
            logging.error(f"Error en ConnectionActionThread ({self.action_type}): {e}")
            self.action_result.emit(False)
        finally:
            loop.close()

def main():
    if platform.system() == "Windows" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ESP32App()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
