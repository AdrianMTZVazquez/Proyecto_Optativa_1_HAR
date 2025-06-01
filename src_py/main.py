"""
Guía para la Conexión BLE en ESP32 con MicroPython v1.25.0

Introducción:
Este script utiliza MicroPython en el ESP32 para implementar un periférico BLE basado en el módulo ubluetooth (importado como bluetooth).

Principales Consideraciones:
- Activar BLE con ble.active(True).
- Registrar servicios GATT usando ble.gatts_register_services(), definiendo características (por ejemplo, para transmisión y recepción) y usando flags apropiados.
- Advertir el dispositivo con un payload que incluya flags, un nombre (en este caso, 'ESP32') y el UUID del servicio.
- Convertir el UUID de una cadena (SERVICE_UUID_STR) a bytes para incluirlo en el payload utilizando una función auxiliar (uuid_str_to_bytes).
- Manejar eventos BLE asíncronos a través del callback irq (por ejemplo, _IRQ_CENTRAL_CONNECT y _IRQ_GATTS_WRITE) para detectar la conexión y controlar el comportamiento del dispositivo.
- Se optimiza el tamaño del payload para cumplir con el límite de 31 bytes de BLE.

Esta implementación sigue las recomendaciones y prácticas descritas en la Guía de Conexión BLE para MicroPython v1.25.0, que aprovecha mejoras en ROMFS, optimización de memoria y versiones actualizadas de ESP-IDF.
"""

# main.py para ESP32 con MPU6050 y BLE (con indicador LED)
import bluetooth
import struct
import time
from machine import Pin, I2C
from micropython import const

# --- Configuración LED ---
# La mayoría de las placas ESP32 tienen un LED incorporado en GPIO2
LED_PIN_NUM = 2
led = Pin(LED_PIN_NUM, Pin.OUT)
led.off()  # Asegurarse de que inicie apagado
last_led_toggle_time = time.ticks_ms()
BLINK_INTERVAL_MS = 500 # Intervalo de parpadeo (500ms ON, 500ms OFF = 1Hz)

# --- Configuración MPU6050 ---
MPU6050_ADDR = 0x69
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_PWR_MGMT_1 = 0x6B

# --- Configuración BLE ---
BLE_NAME = "ESP32"
SERVICE_UUID_STR = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
RX_CHAR_UUID_STR = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
TX_CHAR_UUID_STR = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

SERVICE_UUID = bluetooth.UUID(SERVICE_UUID_STR)
RX_CHAR_UUID = bluetooth.UUID(RX_CHAR_UUID_STR)
TX_CHAR_UUID = bluetooth.UUID(TX_CHAR_UUID_STR)

# --- Inicialización I2C ---
i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=100000)  # I2C0, pines SCL y SDA

def mpu6050_init():
    try:
        devices = i2c.scan()
        if MPU6050_ADDR not in devices:
            print(f"Error: MPU6050 no encontrado. Dispositivos: {devices}")
            return False
        i2c.writeto_mem(MPU6050_ADDR, MPU6050_PWR_MGMT_1, bytearray([0]))
        time.sleep(0.1)  # Delay para permitir estabilización del sensor
        print("MPU6050 inicializado.")
        return True
    except OSError as e:
        print(f"Error OSError inicializando MPU6050: {e}")
        return False

def read_accel():
    try:
        data = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_ACCEL_XOUT_H, 6)
        ax_raw = struct.unpack('>h', data[0:2])[0]
        ay_raw = struct.unpack('>h', data[2:4])[0]
        az_raw = struct.unpack('>h', data[4:6])[0]
        accel_x = ax_raw / 16384.0
        accel_y = ay_raw / 16384.0
        accel_z = az_raw / 16384.0
        return (accel_x, accel_y, accel_z)
    except Exception as e:
        print("Error leyendo MPU6050:", e)
        return (0.0, 0.0, 0.0)

def read_gyro():
    try:
        # Dirección de inicio para los datos del giroscopio (Gyro XOUT_H)
        data = i2c.readfrom_mem(MPU6050_ADDR, 0x43, 6)
        gx_raw = struct.unpack('>h', data[0:2])[0]
        gy_raw = struct.unpack('>h', data[2:4])[0]
        gz_raw = struct.unpack('>h', data[4:6])[0]
        
        # Convertir a grados/segundo (sensibilidad por defecto: ±250 °/s)
        # 131 LSB/(°/s) según la hoja de datos del MPU6050
        gyro_x = gx_raw / 131.0
        gyro_y = gy_raw / 131.0
        gyro_z = gz_raw / 131.0
        return (gyro_x, gyro_y, gyro_z)
    except Exception as e:
        print("Error leyendo giroscopio:", e)
        return (0.0, 0.0, 0.0)

# --- Inicialización BLE ---
ble = bluetooth.BLE()
if not ble.active():
    ble.active(True)
print("BLE activado.")

_RX_CHAR = (RX_CHAR_UUID, bluetooth.FLAG_WRITE | bluetooth.FLAG_WRITE_NO_RESPONSE,)
_TX_CHAR = (TX_CHAR_UUID, bluetooth.FLAG_READ | bluetooth.FLAG_NOTIFY,)
_SERVICE = (SERVICE_UUID, (_RX_CHAR, _TX_CHAR),)

services_handles = ble.gatts_register_services((_SERVICE,))
rx_handle = services_handles[0][0]
tx_handle = services_handles[0][1]
print(f"Servicio BLE registrado. RX_h: {rx_handle}, TX_h: {tx_handle}")

is_streaming = False
client_conn_handle = None

def ble_irq(event, data):
    global is_streaming, client_conn_handle, led
    if event == const(1):  # _IRQ_CENTRAL_CONNECT
        conn_handle, _, _ = data
        client_conn_handle = conn_handle
        print(f"Dispositivo conectado, handle: {client_conn_handle}")
        led.on()  # LED encendido fijo al conectar
    elif event == const(2):  # _IRQ_CENTRAL_DISCONNECT
        conn_handle, _, _ = data
        if conn_handle == client_conn_handle:
            print(f"Dispositivo desconectado, handle: {conn_handle}")
            client_conn_handle = None
            is_streaming = False
            # El LED volverá a parpadear en el bucle principal
            ble.gap_advertise(100, adv_payload, connectable=True)
            print("Publicidad BLE reiniciada tras desconexión.")
    elif event == const(3):  # _IRQ_GATTS_WRITE
        conn_handle, attr_handle = data
        if conn_handle == client_conn_handle and attr_handle == rx_handle:
            received_data = ble.gatts_read(rx_handle)
            print(f"Datos RX ({rx_handle}): {received_data}")
            if received_data == b'start':
                is_streaming = True
                print("Comando 'start' recibido.")
            elif received_data == b'stop':
                is_streaming = False
                print("Comando 'stop' recibido.")

ble.irq(ble_irq)

adv_payload = bytearray()
def _append_adv_field(adv_type, value):
    global adv_payload
    adv_payload += struct.pack('BB', len(value) + 1, adv_type) + value

_append_adv_field(const(0x01), struct.pack('B', 0x06))
_append_adv_field(const(0x09), BLE_NAME.encode())

def uuid_str_to_bytes(uuid_str):
    return bytes.fromhex(uuid_str.replace('-', ''))

service_uuid_bytes_for_adv = bytearray(uuid_str_to_bytes(SERVICE_UUID_STR))
service_uuid_bytes_for_adv = bytearray(reversed(service_uuid_bytes_for_adv))
_append_adv_field(const(0x07), service_uuid_bytes_for_adv)

ble.gap_advertise(100, adv_payload, connectable=True)
print(f"Publicitando como: {BLE_NAME}")

mpu_initialized = mpu6050_init()
print("Iniciando bucle principal...")

while True:
    # --- Manejo del LED ---
    if client_conn_handle is None:  # No conectado o desconectado, parpadear LED
        if time.ticks_diff(time.ticks_ms(), last_led_toggle_time) > BLINK_INTERVAL_MS:
            led.toggle()  # Alternar estado del LED
            last_led_toggle_time = time.ticks_ms()
    else:  # Conectado, asegurar que el LED quede encendido fijo
        if not led.value():
            led.on()
    
    # --- Streaming de datos ---
    if client_conn_handle is not None and is_streaming and mpu_initialized:
        try:
            accel = read_accel()
            gyro = read_gyro()
            print(f"Accel: {accel}, Gyro: {gyro}")
            time.sleep(0.05)  # Añadir delay para reducir la carga en el bus I2C
            data_packet = struct.pack('<fff', *accel)
            ble.gatts_notify(client_conn_handle, tx_handle, data_packet)
        except Exception as e:
            print(f"Error en bucle de streaming: {e}")
    time.sleep_ms(50)