import asyncio
import sys
import struct
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QTextEdit,
                           QTabWidget, QGridLayout, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, 
                           QGroupBox, QProgressBar, QCheckBox, QRadioButton, QButtonGroup, QMessageBox, QSizePolicy)
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QTextCursor, QFont
from bleak import BleakClient, BleakScanner
import csv
from datetime import datetime
import time # For timestamps in CSV

# BLE Configuration (must match ESP32 firmware)
ESP32_DEVICE_NAME = "ESP32"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Client writes to this (ESP32's RX)
TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Client receives notifications from this (ESP32's TX)

# Imports for HAR features
from scipy.fft import rfft, rfftfreq
import pickle
import os # For model saving/loading paths
from collections import deque # Will be needed later for data buffers

# Attempt to import scikit-learn for ML classification
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn no está disponible. La clasificación será simulada.")

# Activities to classify
ACTIVITIES = ["Squat", "Sit-Up", "Push-Up", "Jump"]

class FeatureExtractor:
    """Clase para extraer características de señales de aceleración para HAR."""
    
    @staticmethod
    def extract_time_domain_features(window):
        """Extrae características del dominio del tiempo."""
        features = []
        
        # Calcula características para cada canal
        for channel_list_data in window: # Iterate through [ax_list, ay_list, az_list]
            # Ensure data is a NumPy array of floats for calculations
            channel_np_data = np.array(channel_list_data, dtype=float)
            
            if channel_np_data.size == 0:
                # Handle empty channel data, perhaps append NaNs or zeros for all features
                # For now, let's append NaNs for each expected feature to maintain feature vector length
                # Number of features per channel: mean, std, var, max, min, median, q1, q3, sum_abs, rms, zero_crossings (11 features)
                features.extend([np.nan] * 11) 
                continue

            # Características estadísticas básicas
            features.append(np.mean(channel_np_data))       # Media
            features.append(np.std(channel_np_data))        # Desviación estándar
            features.append(np.var(channel_np_data))        # Varianza
            features.append(np.max(channel_np_data))        # Máximo
            features.append(np.min(channel_np_data))        # Mínimo
            features.append(np.median(channel_np_data))     # Mediana
            features.append(np.percentile(channel_np_data, 25))  # Primer cuartil
            features.append(np.percentile(channel_np_data, 75))  # Tercer cuartil
            
            # Características adicionales
            features.append(np.sum(np.abs(channel_np_data))) # Suma absoluta
            features.append(np.sqrt(np.mean(channel_np_data**2)))  # RMS
            
            # Cruces por cero (útil para análisis de movimiento)
            # Ensure there are at least two points for diff, and handle signbit for non-negative arrays if necessary
            if channel_np_data.size > 1:
                zero_crossings = np.sum(np.diff(np.signbit(channel_np_data - np.mean(channel_np_data)))) # Centered zero crossings
            else:
                zero_crossings = 0 # Or np.nan if preferred for single point data
            features.append(zero_crossings)
        
        return features
    
    @staticmethod
    def extract_frequency_domain_features(window, sample_rate):
        """Extrae características del dominio de la frecuencia."""
        features = []
        
        for channel_data in window:
            # Aplicar ventana Hanning para reducir fugas espectrales
            windowed_data = channel_data * np.hanning(len(channel_data))
            
            # Calcular FFT
            fft_values = np.abs(rfft(windowed_data))
            freqs = rfftfreq(len(windowed_data), 1.0/sample_rate)
            
            # Evitar división por cero
            if len(fft_values) == 0:
                features.extend([0, 0, 0, 0, 0]) # Should match number of features below
                continue
                
            # Características espectrales
            features.append(np.max(fft_values))  # Amplitud máxima
            dominant_freq_idx = np.argmax(fft_values)
            features.append(freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0)  # Frecuencia dominante
            
            # Energía espectral
            energy = np.sum(fft_values**2)
            features.append(energy)
            
            # Centroide espectral (frecuencia promedio ponderada por amplitud)
            if np.sum(fft_values) > 0:
                spectral_centroid = np.sum(freqs * fft_values) / np.sum(fft_values)
                features.append(spectral_centroid)
            else:
                features.append(0)
            
            # Bandas de frecuencia - dividir en 4 bandas y sumar la energía en cada una
            # Ensure there are enough frequency points for bands
            num_freq_points = len(freqs)
            num_bands = 4
            if num_freq_points >= num_bands:
                band_size = num_freq_points // num_bands
                for i in range(num_bands):
                    start = i * band_size
                    # Ensure 'end' does not exceed array bounds for the last band
                    end = (i + 1) * band_size if i < num_bands - 1 else num_freq_points
                    band_energy = np.sum(fft_values[start:end]**2)
                    features.append(band_energy)
            else:
                features.extend([0] * num_bands) # If not enough points, add zeros for band energies
        
        return features
    
    @staticmethod
    def extract_all_features(windows, sample_rate):
        """Extrae todas las características disponibles de una ventana de datos."""
        all_features_list = [] # Renamed to avoid conflict with 'features' variable name
        
        for window in windows:
            # Características del dominio del tiempo
            time_features = FeatureExtractor.extract_time_domain_features(window)
            
            # Características del dominio de la frecuencia
            freq_features = FeatureExtractor.extract_frequency_domain_features(window, sample_rate)
            
            # Combinar todas las características
            combined_features = time_features + freq_features # Renamed to avoid conflict
            all_features_list.append(combined_features)
        
        return all_features_list
    
    @staticmethod
    def segment_signal(data, window_size, overlap=0.5):
        """Segmenta la señal en ventanas con superposición.
           'data' is expected to be a list of lists/arrays, e.g., [ax_data, ay_data, az_data]
           where ax_data, ay_data, az_data are 1D arrays of samples.
        """
        windows = []
        # Ensure data is not empty and has channels
        if not data or not data[0]:
            return windows

        num_channels = len(data)
        min_len = len(data[0]) # Assuming all channels have same length, take first
        for i in range(1, num_channels):
            if len(data[i]) < min_len:
                min_len = len(data[i]) # Find the minimum length among channels

        step = int(window_size * (1 - overlap))
        
        # Asegurar que hay suficientes datos en el canal más corto
        if min_len < window_size:
            # print(f"Not enough data for segmentation. Min_len: {min_len}, window_size: {window_size}")
            return windows
        
        # Crear ventanas
        for i in range(0, min_len - window_size + 1, step):
            window = [channel_data[i:i+window_size] for channel_data in data]
            windows.append(window)
        
        return windows

class ActivityClassifier:
    """Clase para clasificar actividades humanas basándose en características extraídas."""
    
    def __init__(self, model_type='svm'):
        self.model = None
        self.scaler = None
        self.model_type = model_type
        self.is_trained = False
        
        # Comprobar si scikit-learn está disponible
        if not SKLEARN_AVAILABLE and model_type != 'dummy':
            print("scikit-learn no está disponible, usando clasificador simulado.")
            self.model_type = 'dummy'
    
    def train(self, X, y):
        """Entrena el clasificador con datos y etiquetas."""
        if not SKLEARN_AVAILABLE and self.model_type != 'dummy':
            print("Error: scikit-learn no está disponible. No se puede entrenar el modelo real.")
            self.model_type = 'dummy' # Fallback to dummy if SKLEARN_AVAILABLE changed after init

        if self.model_type == 'dummy':
            # Simulación de entrenamiento
            self.is_trained = True
            unique_labels = np.unique(y)
            num_classes = len(unique_labels) if len(unique_labels) > 0 else len(ACTIVITIES)
            accuracy = 0.85  # Valor simulado
            # Create a more realistic dummy confusion matrix
            conf_matrix = np.zeros((num_classes, num_classes))
            for i in range(num_classes):
                conf_matrix[i,i] = int(0.85 * (len(y)/num_classes)) # Correctly classified
                for j in range(num_classes):
                    if i != j:
                        conf_matrix[i,j] = int(0.15 * (len(y)/num_classes) / (num_classes -1 if num_classes > 1 else 1) ) # Misclassified
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'report': "Reporte simulado: Precisión aproximada de 85% (modelo dummy)"
            }
        
        # Implementación real con scikit-learn
        try:
            # Normalizar los datos
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y # Added stratify
            )
            
            # Crear el modelo según el tipo
            if self.model_type == 'svm':
                self.model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale') # Added some common params
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42) # Added random_state
            elif self.model_type == 'mlp':
                self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True) # Added random_state and early_stopping
            else:
                print(f"Tipo de modelo desconocido: {self.model_type}. Usando SVM por defecto.")
                self.model = SVC(kernel='rbf', probability=True)

            # Entrenar el modelo
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluar el modelo
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test)) # Ensure labels for conf_matrix
            report = classification_report(y_test, y_pred, labels=np.unique(y_test), zero_division=0) # Added zero_division
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'report': report
            }
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            self.is_trained = False
            return None
    
    def predict(self, X):
        """Predice la clase para nuevas características."""
        if not self.is_trained:
            # print("Advertencia: El modelo no está entrenado. Devolviendo predicción dummy.")
            # Fallback to dummy prediction if not trained, to avoid None returns that might crash UI
            activity_idx = np.random.choice(len(ACTIVITIES))
            activity = ACTIVITIES[activity_idx]
            probabilities = np.zeros(len(ACTIVITIES))
            probabilities[activity_idx] = 1.0 / len(ACTIVITIES) # Equal probability for dummy
            return activity, probabilities

        if self.model_type == 'dummy' or not SKLEARN_AVAILABLE:
            # Simulación de predicción
            activity_idx = np.random.choice(len(ACTIVITIES), p=[0.6, 0.1, 0.1, 0.1, 0.1]) # Adjusted probabilities
            activity = ACTIVITIES[activity_idx]
            probabilities = np.zeros(len(ACTIVITIES))
            probabilities[activity_idx] = 0.7 # Main probability
            remaining_prob_total = 0.3
            if len(ACTIVITIES) > 1:
                other_prob = remaining_prob_total / (len(ACTIVITIES) - 1)
                for i in range(len(ACTIVITIES)):
                    if i != activity_idx:
                        probabilities[i] = other_prob
            else: # Only one activity
                 probabilities[activity_idx] = 1.0
            return activity, probabilities
        
        try:
            # Normalizar los datos
            # X should be a 1D array of features for a single window
            if not isinstance(X, np.ndarray):
                X_np = np.array(X)
            else:
                X_np = X
            
            if X_np.ndim == 1:
                X_scaled = self.scaler.transform(X_np.reshape(1, -1))
            elif X_np.ndim == 2 and X_np.shape[0] == 1: # Already in correct shape for single sample
                X_scaled = self.scaler.transform(X_np)
            else:
                raise ValueError(f"Input X for prediction has unexpected shape: {X_np.shape}")

            # Realizar predicción
            prediction_idx = self.model.predict(X_scaled)[0]
            # Map index to activity string if model predicts indices
            # This depends on how labels were encoded during training (e.g., LabelEncoder)
            # For now, assuming model.predict returns the actual class label (e.g., string if trained with strings)
            # If it returns indices, a mapping back to ACTIVITIES would be needed.
            # For simplicity with SKLearn, it's often better to train with integer labels and map back.
            # However, the provided code seems to imply direct string labels or that ACTIVITIES matches encoded labels.
            # Let's assume prediction_idx is the string label or an index into ACTIVITIES if trained with 0..N-1
            if isinstance(prediction_idx, (int, np.integer)) and prediction_idx < len(ACTIVITIES):
                prediction_label = ACTIVITIES[prediction_idx]
            else:
                prediction_label = str(prediction_idx) # Or handle as error if it's not in ACTIVITIES

            probabilities = self.model.predict_proba(X_scaled)[0]
            
            return prediction_label, probabilities
            
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            # Fallback to dummy prediction on error
            activity_idx = np.random.choice(len(ACTIVITIES))
            activity = ACTIVITIES[activity_idx]
            probabilities = np.zeros(len(ACTIVITIES))
            probabilities[activity_idx] = 1.0 / len(ACTIVITIES)
            return activity, probabilities

    def save(self, filename="activity_model.pkl"):
        """Guarda el modelo entrenado en un archivo."""
        if not self.is_trained and self.model_type != 'dummy': # Allow saving dummy model state
            print("Advertencia: El modelo real no está entrenado. No se guardará.")
            return False
        
        try:
            model_data = {
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'activities_definition': ACTIVITIES # Save activities definition with model
            }
            
            if self.model_type != 'dummy' and SKLEARN_AVAILABLE and self.model and self.scaler:
                model_data['model'] = self.model
                model_data['scaler'] = self.scaler
            elif self.model_type != 'dummy':
                print("Advertencia: Modelo real no disponible o scikit-learn no cargado. Guardando estado limitado.")

            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Modelo guardado en {filename}")
            return True
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return False
    
    def load(self, filename="activity_model.pkl"):
        """Carga un modelo entrenado desde un archivo."""
        global ACTIVITIES # Allow updating ACTIVITIES if loaded from model
        try:
            if not os.path.exists(filename):
                print(f"Error: Archivo de modelo no encontrado en {filename}")
                self.is_trained = False
                return False

            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_type = model_data.get('model_type', 'dummy') # Default to dummy if not found
            self.is_trained = model_data.get('is_trained', False)
            loaded_activities = model_data.get('activities_definition')
            if loaded_activities:
                ACTIVITIES = loaded_activities # Update global ACTIVITIES if present in model
                print(f"Definición de actividades cargada desde el modelo: {ACTIVITIES}")
            
            if self.model_type != 'dummy' and SKLEARN_AVAILABLE:
                if 'model' in model_data and 'scaler' in model_data:
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    print(f"Modelo '{self.model_type}' y scaler cargados desde {filename}.")
                else:
                    print(f"Advertencia: Modelo real '{self.model_type}' no encontrado en el archivo. El modelo no está completamente cargado.")
                    self.is_trained = False # Mark as not trained if essential parts are missing
            elif self.model_type != 'dummy' and not SKLEARN_AVAILABLE:
                print(f"Advertencia: scikit-learn no está disponible. No se puede cargar el modelo real '{self.model_type}'.")
                self.is_trained = False # Cannot use the model if sklearn is missing
            elif self.model_type == 'dummy':
                 print(f"Modelo dummy cargado desde {filename}. Estado entrenado: {self.is_trained}")
            
            return True # Return true if file was processed, even if model is dummy/partially loaded
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.is_trained = False
            return False


class BLEDevice(QObject):
    connected_signal = pyqtSignal()
    disconnected_signal = pyqtSignal()
    status_update_signal = pyqtSignal(str)
    data_received_signal = pyqtSignal(float, float, float, float, float, float)
    streaming_started_signal = pyqtSignal()
    streaming_stopped_signal = pyqtSignal()

    def __init__(self, loop):
        super().__init__()
        print("[DEBUG] BLEDevice.__init__ called")
        self.loop = loop
        self.client = None
        self.is_connected = False
        self.is_streaming = False
        self.device_address = None

    async def _scan_for_device(self):
        self.status_update_signal.emit(f"Scanning for {ESP32_DEVICE_NAME}...")
        devices = await BleakScanner.discover()
        for device in devices:
            if device.name == ESP32_DEVICE_NAME:
                self.status_update_signal.emit(f"Found {ESP32_DEVICE_NAME} at {device.address}")
                self.device_address = device.address
                return device.address
        self.status_update_signal.emit(f"{ESP32_DEVICE_NAME} not found.")
        return None

    async def connect_async(self):
        if self.is_connected:
            return
        
        if not self.device_address:
            address = await self._scan_for_device()
            if not address:
                return

        try:
            self.status_update_signal.emit(f"Connecting to {self.device_address}...")
            self.client = BleakClient(self.device_address, loop=self.loop)
            await self.client.connect()
            self.is_connected = self.client.is_connected
            if self.is_connected:
                self.status_update_signal.emit("Connected.")
                self.connected_signal.emit()
                # Automatically start notifications
                await self.client.start_notify(TX_CHAR_UUID, self._notification_handler)
                self.status_update_signal.emit("Notifications started.")
            else:
                self.status_update_signal.emit("Failed to connect.")
        except Exception as e:
            self.is_connected = False
            self.status_update_signal.emit(f"Connection error: {e}")
            if self.client:
                await self.client.disconnect() # Ensure client is disconnected on error
            self.client = None # Reset client

    async def disconnect_async(self):
        if self.client and self.is_connected:
            try:
                if self.is_streaming:
                    await self.stop_streaming_async() # Stop streaming before disconnecting notifications
                await self.client.stop_notify(TX_CHAR_UUID)
                await self.client.disconnect()
            except Exception as e:
                self.status_update_signal.emit(f"Error during disconnect: {e}")
            finally:
                self.is_connected = False
                self.is_streaming = False # Ensure streaming is marked as stopped
                self.client = None
                self.status_update_signal.emit("Disconnected.")
                self.disconnected_signal.emit()
        else:
             # If already disconnected or no client, ensure state is correct
            self.is_connected = False
            self.is_streaming = False
            self.status_update_signal.emit("Already disconnected or no client.")
            self.disconnected_signal.emit()


    def _notification_handler(self, sender, data):
        if len(data) == 24:  # 6 floats * 4 bytes each
            ax, ay, az, gx, gy, gz = struct.unpack('<ffffff', data)  # Especificar little-endian
            self.data_received_signal.emit(ax, ay, az, gx, gy, gz)

    async def start_streaming_async(self):
        if self.client and self.is_connected and not self.is_streaming:
            try:
                await self.client.write_gatt_char(RX_CHAR_UUID, b'start', response=False)
                self.is_streaming = True
                self.status_update_signal.emit("Streaming started.")
                self.streaming_started_signal.emit()
            except Exception as e:
                self.status_update_signal.emit(f"Error starting stream: {e}")
        elif not (self.client and self.is_connected):
            self.status_update_signal.emit("Cannot start streaming: Not connected.")

    async def stop_streaming_async(self):
        if self.client and self.is_connected and self.is_streaming:
            try:
                await self.client.write_gatt_char(RX_CHAR_UUID, b'stop', response=False)
                self.is_streaming = False
                self.status_update_signal.emit("Streaming stopped.")
                self.streaming_stopped_signal.emit()
            except Exception as e:
                self.status_update_signal.emit(f"Error stopping stream: {e}")

    # Public methods to be called from GUI thread
    def connect(self):
        print("[DEBUG] BLEDevice.connect called")
        if self.loop:
            self.loop.create_task(self.connect_async())
        else:
            print("[ERROR] BLEDevice.loop is not set in connect!")

    def disconnect(self):
        print("[DEBUG] BLEDevice.disconnect called")
        if self.loop:
            self.loop.create_task(self.disconnect_async())
        else:
            print("[ERROR] BLEDevice.loop is not set in disconnect!")

    def start_streaming(self):
        print("[DEBUG] BLEDevice.start_streaming called")
        if self.loop:
            self.loop.create_task(self.start_streaming_async())
        else:
            print("[ERROR] BLEDevice.loop is not set in start_streaming!")

    def stop_streaming(self):
        print("[DEBUG] BLEDevice.stop_streaming called")
        if self.loop:
            self.loop.create_task(self.stop_streaming_async())
        else:
            print("[ERROR] BLEDevice.loop is not set in stop_streaming!")

class MainWindow(QMainWindow):
    # Constants for data handling and plotting
    SAMPLE_RATE = 50  # Hz (Placeholder - adjust based on ESP32's actual data rate)
    PLOT_TIME_WINDOW_S = 4  # Seconds of data to display in time plot
    PLOT_MAX_POINTS = int(PLOT_TIME_WINDOW_S * SAMPLE_RATE)
    FFT_WINDOW_SIZE_SAMPLES = 256  # Samples for FFT calculation (power of 2 is efficient)
    # Buffer a bit more than FFT window to ensure enough data is available
    DATA_BUFFER_MAXLEN = FFT_WINDOW_SIZE_SAMPLES * 2 

    # Overlap for segmenting signal for feature extraction
    FEATURE_WINDOW_SIZE_S = 2.56 # Seconds, common in HAR
    FEATURE_WINDOW_SAMPLES = int(FEATURE_WINDOW_SIZE_S * SAMPLE_RATE)
    FEATURE_OVERLAP = 0.5

    def __init__(self, loop): # Accept asyncio loop as argument
        super().__init__()
        self.setWindowTitle("ESP32 MPU6050 HAR Monitor")
        self.setGeometry(100, 100, 900, 750) # Adjusted size

        self.loop = loop # Store the asyncio loop

        # Initialize HAR components
        self.feature_extractor = FeatureExtractor()
        self.classifier = ActivityClassifier(model_type='svm') # Default model type
        try:
            if os.path.exists("activity_model.pkl"):
                self.classifier.load("activity_model.pkl")
                self.log_status(f"Modelo '{self.classifier.model_type}' cargado desde activity_model.pkl")
            else:
                self.log_status("No se encontró activity_model.pkl. Usando modelo nuevo/dummy.")
        except Exception as e:
            self.log_status(f"Error al cargar el modelo: {e}")

        # Data buffers using deque for efficient appends and pops from left
        self.ax_buffer = deque(maxlen=self.DATA_BUFFER_MAXLEN)
        self.ay_buffer = deque(maxlen=self.DATA_BUFFER_MAXLEN)
        self.az_buffer = deque(maxlen=self.DATA_BUFFER_MAXLEN)
        self.gx_buffer = deque(maxlen=self.DATA_BUFFER_MAXLEN)
        self.gy_buffer = deque(maxlen=self.DATA_BUFFER_MAXLEN)
        self.gz_buffer = deque(maxlen=self.DATA_BUFFER_MAXLEN)
        
        # Data for plotting (numpy arrays derived from deques)
        self.time_plot_data_x = np.linspace(-self.PLOT_TIME_WINDOW_S, 0, self.PLOT_MAX_POINTS)
        self.ax_plot_data = np.zeros(self.PLOT_MAX_POINTS)
        self.ay_plot_data = np.zeros(self.PLOT_MAX_POINTS)
        self.az_plot_data = np.zeros(self.PLOT_MAX_POINTS)
        self.gx_plot_data = np.zeros(self.PLOT_MAX_POINTS)
        self.gy_plot_data = np.zeros(self.PLOT_MAX_POINTS)
        self.gz_plot_data = np.zeros(self.PLOT_MAX_POINTS)

        # FFT plot data
        self.fft_freq_data = np.array([]) 
        self.ax_fft_mag_data = np.array([])
        self.ay_fft_mag_data = np.array([])
        self.az_fft_mag_data = np.array([])

        # Data collection for training
        self.is_collecting_training_data = False
        self.current_training_activity_label = ACTIVITIES[0]
        self.training_data_buffer = [] # List to store [ax, ay, az] segments
        self.training_labels_buffer = []

        # Buffers for collecting a single training segment with overlap
        self.ax_train_segment_buffer = deque(maxlen=self.FEATURE_WINDOW_SAMPLES + int(self.FEATURE_WINDOW_SAMPLES * self.FEATURE_OVERLAP))
        self.ay_train_segment_buffer = deque(maxlen=self.FEATURE_WINDOW_SAMPLES + int(self.FEATURE_WINDOW_SAMPLES * self.FEATURE_OVERLAP))
        self.az_train_segment_buffer = deque(maxlen=self.FEATURE_WINDOW_SAMPLES + int(self.FEATURE_WINDOW_SAMPLES * self.FEATURE_OVERLAP))
        self.collected_segments_count = 0
        self.target_segments_to_collect = 10 # Default value
        self.training_log_messages = [] # For messages in training tab

        # CSV Logging attributes
        self.is_csv_logging_active = False
        self.current_csv_file_path = None
        self.csv_file_object = None
        self.csv_writer = None

        # Setup Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._setup_monitoring_tab()
        self._setup_classification_tab() # Initialize classification UI elements first
        self._setup_training_tab()     # Training tab setup can now safely access them

        # BLE Device Handler
        self.ble_device = BLEDevice(self.loop)
        self.ble_device.connected_signal.connect(self.handle_connected)
        self.ble_device.disconnected_signal.connect(self.handle_disconnected)
        self.ble_device.status_update_signal.connect(self.log_status)
        self.ble_device.data_received_signal.connect(self.handle_new_data)
        self.ble_device.streaming_started_signal.connect(self.handle_streaming_started)
        self.ble_device.streaming_stopped_signal.connect(self.handle_streaming_stopped)

        # Timers
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots_and_classification) # Renamed
        self.plot_timer.start(50)  # Update plot every 50ms (20 Hz UI update)

        self.async_timer = QTimer(self)
        self.async_timer.timeout.connect(self.run_async_tasks)
        self.async_timer.start(10) 

        self.log_status("Aplicación iniciada. Conéctese al ESP32.")

    def _setup_monitoring_tab(self):
        self.monitoring_tab = QWidget()
        layout = QVBoxLayout(self.monitoring_tab)

        # Controls
        controls_group = QGroupBox("Controles BLE")
        controls_layout = QHBoxLayout()
        self.btn_connect = QPushButton("Escanear y Conectar")
        self.btn_connect.clicked.connect(self.toggle_connection)
        controls_layout.addWidget(self.btn_connect)

        self.btn_stream = QPushButton("Iniciar Streaming")
        self.btn_stream.clicked.connect(self.toggle_streaming)
        self.btn_stream.setEnabled(False)
        controls_layout.addWidget(self.btn_stream)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Plots Group
        plots_group = QGroupBox("Visualización de Datos")
        plots_layout = QGridLayout() # Use QGridLayout for two plots side-by-side or stacked

        # Time Domain Plot
        self.plot_widget_time = pg.PlotWidget(title="Acelerómetro (Tiempo Real)")
        self.plot_widget_time.setBackground('w')
        self.plot_widget_time.addLegend()
        self.plot_widget_time.setYRange(-10, 10, padding=0.1) # MPU6050 typically +/- 2g to 16g
        self.plot_widget_time.setXRange(-self.PLOT_TIME_WINDOW_S, 0, padding=0.01)
        self.plot_widget_time.showGrid(x=True, y=True)
        self.curve_ax_time = self.plot_widget_time.plot(self.time_plot_data_x, self.ax_plot_data, pen='r', name='Accel X')
        self.curve_ay_time = self.plot_widget_time.plot(self.time_plot_data_x, self.ay_plot_data, pen='g', name='Accel Y')
        self.curve_az_time = self.plot_widget_time.plot(self.time_plot_data_x, self.az_plot_data, pen='b', name='Accel Z')
        self.curve_gx_time = self.plot_widget_time.plot(self.time_plot_data_x, self.gx_plot_data, pen='r', name='Gyro X')
        self.curve_gy_time = self.plot_widget_time.plot(self.time_plot_data_x, self.gy_plot_data, pen='g', name='Gyro Y')
        self.curve_gz_time = self.plot_widget_time.plot(self.time_plot_data_x, self.gz_plot_data, pen='b', name='Gyro Z')
        plots_layout.addWidget(self.plot_widget_time, 0, 0) # Row 0, Col 0

        # FFT Plot
        self.plot_widget_fft = pg.PlotWidget(title="FFT Acelerómetro")
        self.plot_widget_fft.setBackground('w')
        self.plot_widget_fft.addLegend(offset=(10,10))
        self.plot_widget_fft.setLabel('bottom', 'Frecuencia', units='Hz')
        self.plot_widget_fft.setLabel('left', 'Magnitud', units='g')
        self.plot_widget_fft.setYRange(0, 0.5, padding=0.1) # Adjust based on expected magnitudes
        self.plot_widget_fft.setXRange(0, self.SAMPLE_RATE / 2, padding=0.01) # Nyquist frequency
        self.plot_widget_fft.showGrid(x=True, y=True)
        self.curve_ax_fft = self.plot_widget_fft.plot(self.fft_freq_data, self.ax_fft_mag_data, pen='r', name='FFT X')
        self.curve_ay_fft = self.plot_widget_fft.plot(self.fft_freq_data, self.ay_fft_mag_data, pen='g', name='FFT Y')
        self.curve_az_fft = self.plot_widget_fft.plot(self.fft_freq_data, self.az_fft_mag_data, pen='b', name='FFT Z')
        plots_layout.addWidget(self.plot_widget_fft, 0, 1) # Row 0, Col 1
        
        plots_group.setLayout(plots_layout)
        layout.addWidget(plots_group)

        # Status Log
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFixedHeight(100)
        layout.addWidget(self.status_log)

        # CSV Logging controls
        csv_logging_group = QGroupBox("Registro CSV de Datos Crudos")
        csv_logging_layout = QHBoxLayout()
        self.btn_toggle_csv_log = QPushButton("Iniciar Registro CSV")
        self.btn_toggle_csv_log.setCheckable(True)
        self.btn_toggle_csv_log.clicked.connect(self._toggle_csv_logging)
        self.csv_log_status_label = QLabel("Registro CSV: Inactivo")
        self.csv_log_status_label.setWordWrap(True)
        csv_logging_layout.addWidget(self.btn_toggle_csv_log)
        csv_logging_layout.addWidget(self.csv_log_status_label, 1) # Give label more space
        csv_logging_group.setLayout(csv_logging_layout)
        layout.addWidget(csv_logging_group) # Add to the main monitoring tab layout

        self.tabs.addTab(self.monitoring_tab, "Monitorización")

    def _toggle_csv_logging(self):
        if self.btn_toggle_csv_log.isChecked(): # Start logging
            log_dir = "csv_logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_csv_file_path = os.path.join(log_dir, f"esp32_data_{timestamp_str}.csv")
            try:
                self.csv_file_object = open(self.current_csv_file_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file_object)
                self.csv_writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'activity_label']) # Header
                self.is_csv_logging_active = True
                self.btn_toggle_csv_log.setText("Detener Registro CSV")
                self.csv_log_status_label.setText(f"Registrando en: {os.path.basename(self.current_csv_file_path)}")
                self.log_status(f"Iniciado registro CSV en {self.current_csv_file_path}")
            except IOError as e:
                self.log_status(f"Error al abrir archivo CSV: {e}")
                self.csv_log_status_label.setText("Error al iniciar registro CSV")
                self.btn_toggle_csv_log.setChecked(False)
                self.is_csv_logging_active = False
        else: # Stop logging
            if self.csv_file_object:
                self.csv_file_object.close()
                self.log_status(f"Registro CSV detenido. Datos guardados en {self.current_csv_file_path}")
            self.is_csv_logging_active = False
            self.csv_file_object = None
            self.csv_writer = None
            self.btn_toggle_csv_log.setText("Iniciar Registro CSV")
            self.csv_log_status_label.setText("Registro CSV: Inactivo")

    def _setup_training_tab(self):
        self.training_tab = QWidget()
        main_layout = QVBoxLayout(self.training_tab)

        # Data Collection Group
        collection_group = QGroupBox("Recolección de Datos de Entrenamiento")
        collection_layout = QGridLayout()

        collection_layout.addWidget(QLabel("Actividad:"), 0, 0)
        self.activity_combo_train = QComboBox()
        self.activity_combo_train.addItems(ACTIVITIES)
        self.activity_combo_train.currentTextChanged.connect(self._on_activity_selected_train)
        self._on_activity_selected_train(self.activity_combo_train.currentText()) # Initialize
        collection_layout.addWidget(self.activity_combo_train, 0, 1)

        collection_layout.addWidget(QLabel("Segmentos a recolectar:"), 1, 0)
        self.segments_spinbox = QSpinBox()
        self.segments_spinbox.setRange(1, 500)
        self.segments_spinbox.setValue(self.target_segments_to_collect) # Default 10 segments
        self.segments_spinbox.valueChanged.connect(lambda val: setattr(self, 'target_segments_to_collect', val))
        collection_layout.addWidget(self.segments_spinbox, 1, 1)

        self.btn_collect_data = QPushButton("Iniciar Recolección")
        self.btn_collect_data.clicked.connect(self._toggle_data_collection)
        collection_layout.addWidget(self.btn_collect_data, 2, 0, 1, 2)

        self.collection_progress = QProgressBar()
        self.collection_progress.setRange(0, self.target_segments_to_collect)
        self.collection_progress.setValue(0)
        collection_layout.addWidget(self.collection_progress, 3, 0, 1, 2)
        collection_group.setLayout(collection_layout)
        main_layout.addWidget(collection_group)

        # Model Training Group
        training_group = QGroupBox("Entrenamiento y Gestión del Modelo")
        training_layout = QGridLayout()

        training_layout.addWidget(QLabel("Tipo de Modelo:"), 0, 0)
        self.model_type_combo_train = QComboBox()
        self.model_type_combo_train.addItems(['svm', 'random_forest', 'mlp', 'dummy'])
        self.model_type_combo_train.currentTextChanged.connect(self._on_model_type_selected_train)
        # Initialize model type related status in _on_model_type_selected_train
        # self._on_model_type_selected_train(self.model_type_combo_train.currentText()) # Called in __init__ effectively
        training_layout.addWidget(self.model_type_combo_train, 0, 1)

        self.btn_train_model = QPushButton("Entrenar Modelo")
        self.btn_train_model.clicked.connect(self._train_activity_model)
        training_layout.addWidget(self.btn_train_model, 1, 0, 1, 2)

        self.btn_save_model = QPushButton("Guardar Modelo")
        self.btn_save_model.clicked.connect(self._save_model)
        training_layout.addWidget(self.btn_save_model, 2, 0)

        self.btn_load_model = QPushButton("Cargar Modelo")
        self.btn_load_model.clicked.connect(self._load_model)
        training_layout.addWidget(self.btn_load_model, 2, 1)
        training_group.setLayout(training_layout)
        main_layout.addWidget(training_group)

        # Training Log
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setFixedHeight(150)
        main_layout.addWidget(self.training_log)

        main_layout.addStretch()
        self.tabs.addTab(self.training_tab, "Entrenamiento")
        self._log_training_status("Pestaña de entrenamiento inicializada.")
        # Initialize model type status after UI is ready
        self._on_model_type_selected_train(self.model_type_combo_train.currentText())

    def _log_training_status(self, message):
        self.training_log_messages.append(message)
        if len(self.training_log_messages) > 100: # Keep log size manageable
            self.training_log_messages.pop(0)
        if hasattr(self, 'training_log'): # Ensure UI element exists
            self.training_log.setPlainText("\n".join(self.training_log_messages))
            self.training_log.moveCursor(QTextCursor.End)
        print(f"[TRAIN_LOG] {message}")

    def _on_activity_selected_train(self, activity_name):
        self.current_training_activity_label = activity_name
        self._log_training_status(f"Actividad seleccionada para recolección: {activity_name}")

    def _on_model_type_selected_train(self, model_type):
        if not hasattr(self, 'classifier') or self.classifier.model_type != model_type:
            self.classifier = ActivityClassifier(model_type=model_type)
            self._log_training_status(f"Tipo de modelo cambiado a: {model_type}. Modelo reiniciado.")
            self.log_status(f"Clasificador reiniciado a tipo: {model_type}")
        # Update classification tab status if model was trained, and if classification_status_label exists
        if hasattr(self, 'classification_status_label'):
            if not self.classifier.is_trained:
                 self.classification_status_label.setText(f"Actividad Predicha: Modelo {model_type} no entrenado")
                 if hasattr(self, 'probabilities_text'): self.probabilities_text.setText("")
            else:
                # If model is trained, but type changed, it's effectively a new untrained model of this type
                self.classification_status_label.setText(f"Actividad Predicha: Modelo {model_type} listo (pero datos anteriores eran para otro tipo)")

    def _toggle_data_collection(self):
        if not self.ble_device.is_streaming:
            QMessageBox.warning(self, "Advertencia", "Debe iniciar el streaming de datos desde el ESP32 antes de recolectar.")
            self._log_training_status("Advertencia: Streaming no iniciado.")
            return

        if self.is_collecting_training_data:
            self.is_collecting_training_data = False
            self.btn_collect_data.setText("Iniciar Recolección")
            self._log_training_status("Recolección de datos detenida.")
            self.activity_combo_train.setEnabled(True)
            self.segments_spinbox.setEnabled(True)
        else:
            self.is_collecting_training_data = True
            self.btn_collect_data.setText("Detener Recolección")
            self.target_segments_to_collect = self.segments_spinbox.value()
            self.collection_progress.setRange(0, self.target_segments_to_collect)
            self.collection_progress.setValue(0)
            self.collected_segments_count = 0
            # Clear previous segment buffers for new collection session
            self.ax_train_segment_buffer.clear()
            self.ay_train_segment_buffer.clear()
            self.az_train_segment_buffer.clear()
            self._log_training_status(f"Iniciando recolección para '{self.current_training_activity_label}', {self.target_segments_to_collect} segmentos.")
            self.activity_combo_train.setEnabled(False)
            self.segments_spinbox.setEnabled(False)

    def _process_training_data_collection(self):
        # This method is called periodically by the plot_timer (via update_plots_and_classification)
        # if is_collecting_training_data is true
        if not self.is_collecting_training_data or self.collected_segments_count >= self.target_segments_to_collect:
            return

        # Check if enough data accumulated in the training segment buffer for one feature window
        if len(self.ax_train_segment_buffer) >= self.FEATURE_WINDOW_SAMPLES:
            # Extract one segment of FEATURE_WINDOW_SAMPLES length
            current_ax_segment = list(self.ax_train_segment_buffer)[:self.FEATURE_WINDOW_SAMPLES]
            current_ay_segment = list(self.ay_train_segment_buffer)[:self.FEATURE_WINDOW_SAMPLES]
            current_az_segment = list(self.az_train_segment_buffer)[:self.FEATURE_WINDOW_SAMPLES]

            segment_data_for_features = [current_ax_segment, current_ay_segment, current_az_segment]
            
            self.training_data_buffer.append(segment_data_for_features)
            self.training_labels_buffer.append(self.current_training_activity_label)
            self.collected_segments_count += 1
            self.collection_progress.setValue(self.collected_segments_count)
            self._log_training_status(f"Segmento {self.collected_segments_count}/{self.target_segments_to_collect} para '{self.current_training_activity_label}' recolectado.")

            # Remove data based on overlap to prepare for the next segment
            samples_to_remove = int(self.FEATURE_WINDOW_SAMPLES * (1 - self.FEATURE_OVERLAP))
            for _ in range(samples_to_remove):
                if self.ax_train_segment_buffer: self.ax_train_segment_buffer.popleft()
                if self.ay_train_segment_buffer: self.ay_train_segment_buffer.popleft()
                if self.az_train_segment_buffer: self.az_train_segment_buffer.popleft()

            if self.collected_segments_count >= self.target_segments_to_collect:
                self.is_collecting_training_data = False # Stop collection automatically
                self.btn_collect_data.setText("Iniciar Recolección")
                self._log_training_status(f"Recolección completada para '{self.current_training_activity_label}'.")
                self.activity_combo_train.setEnabled(True)
                self.segments_spinbox.setEnabled(True)
                QMessageBox.information(self, "Recolección Completada", f"Se han recolectado {self.target_segments_to_collect} segmentos para '{self.current_training_activity_label}'.")

    def _train_activity_model(self):
        if not self.training_data_buffer or not self.training_labels_buffer:
            self._log_training_status("No hay datos recolectados para entrenar.")
            QMessageBox.warning(self, "Entrenamiento", "No hay datos de entrenamiento disponibles.")
            return

        self._log_training_status(f"Iniciando entrenamiento del modelo '{self.classifier.model_type}'...")
        self._log_training_status(f"Total segmentos para entrenamiento: {len(self.training_data_buffer)}")
        
        all_features_X = []
        for segment_data in self.training_data_buffer:
            time_features = self.feature_extractor.extract_time_domain_features(segment_data)
            freq_features = self.feature_extractor.extract_frequency_domain_features(segment_data, self.SAMPLE_RATE)
            all_features_X.append(time_features + freq_features)
        
        X = np.array(all_features_X)
        y = np.array(self.training_labels_buffer)

        if X.shape[0] == 0:
            self._log_training_status("Error: No se pudieron extraer features de los datos recolectados.")
            return

        self._log_training_status(f"Features extraídas: {X.shape[0]} muestras, {X.shape[1]} features por muestra.")
        
        try:
            self.classifier.train(X, y)
            self._log_training_status(f"Modelo '{self.classifier.model_type}' entrenado exitosamente.")
            self.log_status(f"Modelo '{self.classifier.model_type}' entrenado.")
            QMessageBox.information(self, "Entrenamiento", f"Modelo '{self.classifier.model_type}' entrenado exitosamente.")
            if hasattr(self, 'classification_status_label'): self.classification_status_label.setText("Actividad Predicha: Esperando datos...") 
        except Exception as e:
            self._log_training_status(f"Error durante el entrenamiento: {e}")
            self.log_status(f"Error entrenando modelo: {e}")
            QMessageBox.critical(self, "Error de Entrenamiento", f"Ocurrió un error: {e}")

    def _save_model(self):
        if not self.classifier.is_trained and self.classifier.model_type != 'dummy':
            QMessageBox.warning(self, "Guardar Modelo", "El modelo no ha sido entrenado todavía.")
            self._log_training_status("Intento de guardar modelo no entrenado.")
            return
        
        filePath, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo", "activity_model.pkl", "Pickle Files (*.pkl)")
        if filePath:
            try:
                self.classifier.save(filePath)
                self._log_training_status(f"Modelo guardado en: {filePath}")
                self.log_status(f"Modelo guardado en: {filePath}")
            except Exception as e:
                self._log_training_status(f"Error al guardar modelo: {e}")
                self.log_status(f"Error al guardar modelo: {e}")
                QMessageBox.critical(self, "Error", f"No se pudo guardar el modelo: {e}")

    def _load_model(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Cargar Modelo", "", "Pickle Files (*.pkl)")
        if filePath:
            try:
                self.classifier.load(filePath)
                self._log_training_status(f"Modelo '{self.classifier.model_type}' cargado desde: {filePath}")
                self.log_status(f"Modelo '{self.classifier.model_type}' cargado desde: {filePath}")
                idx = self.model_type_combo_train.findText(self.classifier.model_type, Qt.MatchFixedString)
                if idx >= 0:
                    self.model_type_combo_train.setCurrentIndex(idx)
                else: # Model type from file not in combo box, add it or warn
                    self._log_training_status(f"Advertencia: Tipo de modelo '{self.classifier.model_type}' del archivo no está en la lista. Usando como está.")
                
                if hasattr(self, 'classification_status_label'):
                    if self.classifier.is_trained:
                        self.classification_status_label.setText("Actividad Predicha: Esperando datos...")
                    else:
                        self.classification_status_label.setText(f"Actividad Predicha: Modelo {self.classifier.model_type} cargado pero no entrenado (?)")
                if hasattr(self, 'probabilities_text'): self.probabilities_text.setText("")

            except Exception as e:
                self._log_training_status(f"Error al cargar modelo: {e}")
                self.log_status(f"Error al cargar modelo: {e}")
                QMessageBox.critical(self, "Error", f"No se pudo cargar el modelo: {e}")

    def _setup_classification_tab(self):
        self.classification_tab = QWidget()
        layout = QVBoxLayout(self.classification_tab)
        
        self.classification_status_label = QLabel("Actividad Predicha: Esperando datos...")
        self.classification_status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        self.classification_status_label.setFont(font)
        layout.addWidget(self.classification_status_label)

        self.probabilities_text = QTextEdit()
        self.probabilities_text.setReadOnly(True)
        self.probabilities_text.setFixedHeight(150)
        layout.addWidget(self.probabilities_text)

        layout.addStretch() # Push content to the top
        self.tabs.addTab(self.classification_tab, "Clasificación en Tiempo Real")

    def log_status(self, message):
        # Ensure status_log is initialized before logging
        if hasattr(self, 'status_log') and self.status_log:
            self.status_log.append(message)
            self.status_log.moveCursor(QTextCursor.End)
        else:
            # This case handles logs attempted before status_log is created (e.g. during __init__)
            print(f"[LOG_STATUS_EARLY]: {message}")

    def toggle_connection(self):
        if not self.ble_device.is_connected:
            self.ble_device.connect() # connect is a wrapper for async version
        else:
            self.ble_device.disconnect()

    def toggle_streaming(self):
        if not self.ble_device.is_streaming:
            self.ble_device.start_streaming()
        else:
            self.ble_device.stop_streaming()

    @pyqtSlot()
    def handle_connected(self):
        self.btn_connect.setText("Desconectar")
        self.btn_stream.setEnabled(True)
        self.btn_stream.setText("Iniciar Streaming")
        self.log_status("Conectado al dispositivo BLE.")

    @pyqtSlot()
    def handle_disconnected(self):
        self.btn_connect.setText("Escanear y Conectar")
        self.btn_stream.setEnabled(False)
        self.btn_stream.setText("Iniciar Streaming")
        self.log_status("Desconectado del dispositivo BLE.")
        # Clear buffers and plots on disconnect if desired
        self.ax_buffer.clear()
        self.ay_buffer.clear()
        self.az_buffer.clear()
        self.gx_buffer.clear()
        self.gy_buffer.clear()
        self.gz_buffer.clear()
        self.ax_plot_data.fill(0)
        self.ay_plot_data.fill(0)
        self.az_plot_data.fill(0)
        self.ax_fft_mag_data = np.array([])
        self.ay_fft_mag_data = np.array([])
        self.az_fft_mag_data = np.array([])
        self.update_time_plot()
        self.update_fft_plot()

    @pyqtSlot()
    def handle_streaming_started(self):
        self.btn_stream.setText("Detener Streaming")
        self.log_status("Streaming de datos iniciado.")

    @pyqtSlot()
    def handle_streaming_stopped(self):
        self.btn_stream.setText("Iniciar Streaming")
        self.log_status("Streaming de datos detenido.")

    def update_time_plot(self):
        self._update_single_plot(self.ax_buffer, self.ax_plot_data)
        self._update_single_plot(self.ay_buffer, self.ay_plot_data)
        self._update_single_plot(self.az_buffer, self.az_plot_data)
        self._update_single_plot(self.gx_buffer, self.gx_plot_data)
        self._update_single_plot(self.gy_buffer, self.gy_plot_data)
        self._update_single_plot(self.gz_buffer, self.gz_plot_data)

        self._update_curves()

    def _update_single_plot(self, buffer, plot_data):
        length = len(buffer)
        if length > 0:
            start = max(0, length - self.PLOT_MAX_POINTS)
            plot_data[-length + start:] = list(buffer)[start:]
            if length < self.PLOT_MAX_POINTS:
                plot_data[:self.PLOT_MAX_POINTS - length] = 0

    def _update_curves(self):
        self.curve_ax_time.setData(self.time_plot_data_x, self.ax_plot_data)
        self.curve_ay_time.setData(self.time_plot_data_x, self.ay_plot_data)
        self.curve_az_time.setData(self.time_plot_data_x, self.az_plot_data)
        self.curve_gx_time.setData(self.time_plot_data_x, self.gx_plot_data)
        self.curve_gy_time.setData(self.time_plot_data_x, self.gy_plot_data)
        self.curve_gz_time.setData(self.time_plot_data_x, self.gz_plot_data)

    def update_fft_plot(self):
        if len(self.ax_buffer) >= self.FFT_WINDOW_SIZE_SAMPLES:
            # Use the most recent FFT_WINDOW_SIZE_SAMPLES for FFT
            ax_fft_input = np.array(list(self.ax_buffer)[-self.FFT_WINDOW_SIZE_SAMPLES:])
            ay_fft_input = np.array(list(self.ay_buffer)[-self.FFT_WINDOW_SIZE_SAMPLES:])
            az_fft_input = np.array(list(self.az_buffer)[-self.FFT_WINDOW_SIZE_SAMPLES:])

            # Apply windowing function (e.g., Hanning) before FFT
            hanning_window = np.hanning(self.FFT_WINDOW_SIZE_SAMPLES)
            ax_fft_input = ax_fft_input * hanning_window
            ay_fft_input = ay_fft_input * hanning_window
            az_fft_input = az_fft_input * hanning_window

            # Perform FFT
            self.ax_fft_mag_data = np.abs(np.fft.rfft(ax_fft_input)) / self.FFT_WINDOW_SIZE_SAMPLES * 2 # Normalize
            self.ay_fft_mag_data = np.abs(np.fft.rfft(ay_fft_input)) / self.FFT_WINDOW_SIZE_SAMPLES * 2
            self.az_fft_mag_data = np.abs(np.fft.rfft(az_fft_input)) / self.FFT_WINDOW_SIZE_SAMPLES * 2
            self.fft_freq_data = np.fft.rfftfreq(self.FFT_WINDOW_SIZE_SAMPLES, d=1./self.SAMPLE_RATE)
            
            self.curve_ax_fft.setData(self.fft_freq_data, self.ax_fft_mag_data)
            self.curve_ay_fft.setData(self.fft_freq_data, self.ay_fft_mag_data)
            self.curve_az_fft.setData(self.fft_freq_data, self.az_fft_mag_data)
        else:
            # Not enough data, clear FFT plot or show placeholder
            self.curve_ax_fft.setData([], [])
            self.curve_ay_fft.setData([], [])
            self.curve_az_fft.setData([], [])

    def perform_real_time_classification(self):
        if not self.classifier.is_trained:
            # self.classification_status_label.setText("Actividad Predicha: Modelo no entrenado")
            return # Don't attempt if model isn't trained

        if len(self.ax_buffer) >= self.FEATURE_WINDOW_SAMPLES:
            # Prepare a single window of data for feature extraction
            current_ax_window = list(self.ax_buffer)[-self.FEATURE_WINDOW_SAMPLES:]
            current_ay_window = list(self.ay_buffer)[-self.FEATURE_WINDOW_SAMPLES:]
            current_az_window = list(self.az_buffer)[-self.FEATURE_WINDOW_SAMPLES:]
            window_data = [current_ax_window, current_ay_window, current_az_window]

            time_features = self.feature_extractor.extract_time_domain_features(window_data)
            freq_features = self.feature_extractor.extract_frequency_domain_features(window_data, self.SAMPLE_RATE)
            features = np.array(time_features + freq_features).reshape(1, -1)

            predicted_activity, probabilities = self.classifier.predict(features)
            self.classification_status_label.setText(f"Actividad Predicha: {predicted_activity}")
            
            # Display probabilities
            prob_text = "Probabilidades:\n"
            if probabilities is not None and isinstance(probabilities, dict):
                for activity, prob in probabilities.items():
                    prob_text += f"  {activity}: {prob*100:.2f}%\n"
            elif probabilities is not None: # For dummy classifier that might return a string or simple list
                prob_text += str(probabilities)
            else:
                prob_text += "No disponibles."
            self.probabilities_text.setText(prob_text)
        # else:
            # self.classification_status_label.setText("Actividad Predicha: Recolectando datos...")

    def update_plots_and_classification(self):
        # This is the main callback for the plot_timer
        self.update_time_plot()
        self.update_fft_plot()
        self.perform_real_time_classification()
        
        # If collecting training data, process any accumulated data for segments
        if self.is_collecting_training_data:
            self._process_training_data_collection()

    def run_async_tasks(self):
        # This method is called by a QTimer to give asyncio loop time to run.
        # It processes all pending asyncio tasks.
        self.loop.call_soon(self.loop.stop) # Stop the loop if it's running
        self.loop.run_forever() # Run until stop() is called

    def closeEvent(self, event):
        self.log_status("Cerrando aplicación...")
        if self.ble_device.is_connected:
            self.log_status("Desconectando del dispositivo BLE...")
            # Synchronously disconnect if possible, or manage via async tasks with timeout
            # For simplicity, we'll rely on the BLEDevice's disconnect logic
            # but ensure it's triggered.
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.ble_device.disconnect_async(), self.loop)
            else:
                # If loop isn't running, try to run disconnect briefly
                try:
                    self.loop.run_until_complete(asyncio.wait_for(self.ble_device.disconnect_async(), timeout=2.0))
                except asyncio.TimeoutError:
                    self.log_status("Timeout al desconectar BLE durante el cierre.")
                except Exception as e:
                    self.log_status(f"Error desconectando BLE: {e}")
        
        # Stop timers
        self.plot_timer.stop()
        self.async_timer.stop()

        # Close CSV file if open
        if self.is_csv_logging_active and self.csv_file_object:
            self.csv_file_object.close()
            self.log_status(f"Registro CSV cerrado. Datos guardados en {self.current_csv_file_path}")

        self.log_status("Limpieza finalizada. Saliendo.")
        QApplication.instance().quit()
        event.accept()


    @pyqtSlot(float, float, float, float, float, float)
    def handle_new_data(self, ax, ay, az, gx, gy, gz):
        # print(f"[DATA] ax={ax:.2f}, ay={ay:.2f}, az={az:.2f}")
        self.ax_buffer.append(ax)
        self.ay_buffer.append(ay)
        self.az_buffer.append(az)
        self.gx_buffer.append(gx)
        self.gy_buffer.append(gy)
        self.gz_buffer.append(gz)

        # If collecting training data, add to specific training buffers
        if self.is_collecting_training_data:
            self.ax_train_segment_buffer.append(ax)
            self.ay_train_segment_buffer.append(ay)
            self.az_train_segment_buffer.append(az)
            # The logic to process these buffers into segments is in _process_training_data_collection,
            # which is called by the plot_timer via update_plots_and_classification.

        # If CSV logging is active, write data
        if self.is_csv_logging_active and self.csv_writer:
            current_timestamp = time.time() # Using time.time() for simplicity
            current_activity_label_for_log = self.activity_combo_train.currentText() # Get from training tab
            try:
                self.csv_writer.writerow([current_timestamp, ax, ay, az, gx, gy, gz, current_activity_label_for_log])
            except Exception as e:
                # self.log_status(f"Error escribiendo en CSV: {e}") # This can be very verbose
                # Consider a flag to stop logging or a counter for errors
                pass # For now, silently ignore write errors to prevent spamming logs

    def update_time_plot(self):
        # Copy last PLOT_MAX_POINTS from deques for plotting
        # If deque has fewer than PLOT_MAX_POINTS, pad with zeros at the beginning
        len_ax = len(self.ax_buffer)
        
        if len_ax == 0:
            self.ax_plot_data.fill(0)
            self.ay_plot_data.fill(0)
            self.az_plot_data.fill(0)
            self.gx_plot_data.fill(0)
            self.gy_plot_data.fill(0)
            self.gz_plot_data.fill(0)
        else:
            ax_src = np.array(self.ax_buffer)
            ay_src = np.array(self.ay_buffer)
            az_src = np.array(self.az_buffer)
            gx_src = np.array(self.gx_buffer)
            gy_src = np.array(self.gy_buffer)
            gz_src = np.array(self.gz_buffer)

            points_to_copy = min(len_ax, self.PLOT_MAX_POINTS)
            
            self.ax_plot_data.fill(0)
            self.ay_plot_data.fill(0)
            self.az_plot_data.fill(0)
            self.gx_plot_data.fill(0)
            self.gy_plot_data.fill(0)
            self.gz_plot_data.fill(0)

            self.ax_plot_data[-points_to_copy:] = ax_src[-points_to_copy:]
            self.ay_plot_data[-points_to_copy:] = ay_src[-points_to_copy:]
            self.az_plot_data[-points_to_copy:] = az_src[-points_to_copy:]
            self.gx_plot_data[-points_to_copy:] = gx_src[-points_to_copy:]
            self.gy_plot_data[-points_to_copy:] = gy_src[-points_to_copy:]
            self.gz_plot_data[-points_to_copy:] = gz_src[-points_to_copy:]

        self.curve_ax_time.setData(self.time_plot_data_x, self.ax_plot_data)
        self.curve_ay_time.setData(self.time_plot_data_x, self.ay_plot_data)
        self.curve_az_time.setData(self.time_plot_data_x, self.az_plot_data)
        self.curve_gx_time.setData(self.time_plot_data_x, self.gx_plot_data)
        self.curve_gy_time.setData(self.time_plot_data_x, self.gy_plot_data)
        self.curve_gz_time.setData(self.time_plot_data_x, self.gz_plot_data)

    def update_fft_plot(self):
        if len(self.ax_buffer) >= self.FFT_WINDOW_SIZE_SAMPLES:
            # Take the most recent FFT_WINDOW_SIZE_SAMPLES from the deques
            ax_segment = np.array(list(self.ax_buffer)[-self.FFT_WINDOW_SIZE_SAMPLES:])
            ay_segment = np.array(list(self.ay_buffer)[-self.FFT_WINDOW_SIZE_SAMPLES:])
            az_segment = np.array(list(self.az_buffer)[-self.FFT_WINDOW_SIZE_SAMPLES:])

            # Apply Hanning window
            hanning_window = np.hanning(self.FFT_WINDOW_SIZE_SAMPLES)
            ax_segment_w = ax_segment * hanning_window
            ay_segment_w = ay_segment * hanning_window
            az_segment_w = az_segment * hanning_window

            # Calculate FFT
            self.ax_fft_mag_data = np.abs(rfft(ax_segment_w)) / self.FFT_WINDOW_SIZE_SAMPLES * 2
            self.ay_fft_mag_data = np.abs(rfft(ay_segment_w)) / self.FFT_WINDOW_SIZE_SAMPLES * 2
            self.az_fft_mag_data = np.abs(rfft(az_segment_w)) / self.FFT_WINDOW_SIZE_SAMPLES * 2
            
            # Correct DC component (first element)
            if len(self.ax_fft_mag_data) > 0: self.ax_fft_mag_data[0] /= 2
            if len(self.ay_fft_mag_data) > 0: self.ay_fft_mag_data[0] /= 2
            if len(self.az_fft_mag_data) > 0: self.az_fft_mag_data[0] /= 2

            if len(self.fft_freq_data) != len(self.ax_fft_mag_data):
                 self.fft_freq_data = rfftfreq(self.FFT_WINDOW_SIZE_SAMPLES, 1.0/self.SAMPLE_RATE)

            self.curve_ax_fft.setData(self.fft_freq_data, self.ax_fft_mag_data)
            self.curve_ay_fft.setData(self.fft_freq_data, self.ay_fft_mag_data)
            self.curve_az_fft.setData(self.fft_freq_data, self.az_fft_mag_data)
        else:
            # Not enough data, clear FFT plot or show placeholder
            self.curve_ax_fft.setData([], [])
            self.curve_ay_fft.setData([], [])
            self.curve_az_fft.setData([], [])

    def perform_real_time_classification(self):
        if not self.classifier.is_trained:
            self.classification_status_label.setText("Actividad Predicha: Modelo no entrenado")
            self.probabilities_text.setText("")
            return

        if len(self.ax_buffer) >= self.FEATURE_WINDOW_SAMPLES:
            # Prepare data window for feature extraction
            # Data must be [channel_ax, channel_ay, channel_az]
            # Taking the most recent FEATURE_WINDOW_SAMPLES
            current_ax_window = list(self.ax_buffer)[-self.FEATURE_WINDOW_SAMPLES:]
            current_ay_window = list(self.ay_buffer)[-self.FEATURE_WINDOW_SAMPLES:]
            current_az_window = list(self.az_buffer)[-self.FEATURE_WINDOW_SAMPLES:]
            
            data_window_for_features = [current_ax_window, current_ay_window, current_az_window]

            # Extract features (extract_all_features expects a list of windows)
            # Here we process one window at a time for real-time classification
            time_features = self.feature_extractor.extract_time_domain_features(data_window_for_features)
            freq_features = self.feature_extractor.extract_frequency_domain_features(data_window_for_features, self.SAMPLE_RATE)
            current_features = np.array(time_features + freq_features)

            if current_features.size > 0:
                predicted_activity, probabilities = self.classifier.predict(current_features)
                if predicted_activity is not None and probabilities is not None:
                    self.classification_status_label.setText(f"Actividad Predicha: {predicted_activity}")
                    prob_text = "Probabilidades:\n"
                    for i, activity_name in enumerate(ACTIVITIES):
                        if i < len(probabilities):
                            prob_text += f"  {activity_name}: {probabilities[i]*100:.2f}%\n"
                        else:
                            prob_text += f"  {activity_name}: N/A\n"
                    self.probabilities_text.setText(prob_text)
                else:
                    self.classification_status_label.setText("Actividad Predicha: Error en predicción")
                    self.probabilities_text.setText("")
            else:
                self.classification_status_label.setText("Actividad Predicha: No hay suficientes features")
                self.probabilities_text.setText("")
        else:
            self.classification_status_label.setText("Actividad Predicha: Recolectando datos...")
            self.probabilities_text.setText("")

    def run_async_tasks(self):
        if self.ble_device and self.ble_device.loop and not self.ble_device.loop.is_closed():
            try:
                # Process pending asyncio tasks without blocking Qt event loop
                self.ble_device.loop.call_soon(self.ble_device.loop.stop) # Schedule stop if nothing else runs
                self.ble_device.loop.run_forever() # Will run until stop() is called
            except RuntimeError as e:
                # This can happen if the loop is closed or already running in another thread, though less likely with call_soon/run_forever pattern
                # print(f"[DEBUG] RuntimeError in run_async_tasks: {e}")
                pass 
        # If loop is closed, try to get/set a new one for future tasks (might be needed on re-connect)
        elif self.ble_device and self.ble_device.loop and self.ble_device.loop.is_closed():
            # print("[DEBUG] Asyncio loop was closed. Attempting to set a new one.")
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                self.ble_device.loop = new_loop
            except Exception as e:
                print(f"[ERROR] Could not set new asyncio loop: {e}")

    def closeEvent(self, event):
        self.log_status("Cerrando aplicación...")
        self.plot_timer.stop()
        self.async_timer.stop()

        if self.ble_device and self.ble_device.is_connected:
            # Create a task for disconnection. This is tricky with event loops.
            # Best effort to disconnect without blocking GUI too much.
            if self.ble_device.loop and not self.ble_device.loop.is_closed():
                # Ensure the loop is running to process the disconnect task
                # self.ble_device.loop.run_until_complete(self.ble_device.disconnect_async())
                # Or, if disconnect_async is truly fire-and-forget and handles its own errors:
                asyncio.run_coroutine_threadsafe(self.ble_device.disconnect_async(), self.ble_device.loop)
                # Give it a brief moment. This is not ideal but common for GUI cleanup.
                # time.sleep(0.5) # Blocking, avoid if possible.
            else:
                print("[WARN] BLE device connected but asyncio loop is not available for clean disconnect.")
        
        # Attempt to stop and close the loop if it's managed by this MainWindow context
        if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
            # print("[DEBUG] Closing asyncio event loop from closeEvent.")
            # self.loop.call_soon_threadsafe(self.loop.stop) # Stop the loop
            # This can be complex if tasks are still pending. A more robust shutdown is needed for production.
            pass # For now, let run_async_tasks handle its lifecycle or rely on process exit

        super().closeEvent(event)
        self.plot_timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    print("[DEBUG] Script started.")
    try:
        app = QApplication(sys.argv)
        print("[DEBUG] QApplication instantiated.")

        # Ensure an event loop is running for asyncio.create_task
        try:
            loop = asyncio.get_event_loop()
            print("[DEBUG] Asyncio event loop obtained.")
        except RuntimeError: # This happens if there's no current event loop.
            print("[DEBUG] RuntimeError: No current asyncio event loop. Creating new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop) # Set the new loop as the current one.
            print("[DEBUG] New asyncio event loop created and set.")

        print("[DEBUG] Instantiating MainWindow...")
        window = MainWindow(loop) # Pass the loop to MainWindow
        print("[DEBUG] MainWindow instantiated.")
        
        print("[DEBUG] Showing MainWindow...")
        window.show()
        print("[DEBUG] MainWindow show() called.")
        
        print("[DEBUG] Starting QApplication event loop (app.exec_())...")
        exit_code = app.exec_()
        print(f"[DEBUG] QApplication event loop finished with exit code: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        print(f"[ERROR] An error occurred in __main__: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
