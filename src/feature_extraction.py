# feature_extraction.py
import numpy as np
import pywt
from scipy.signal.windows import hann  # <-- Cambio aquí
from scipy.fft import rfft, rfftfreq

def extract_force_features(bw_signal, t):
    """
    Extrae características del ciclo de histeresis a partir de la señal generada por el modelo Bouc-Wen.
    Se segmenta la señal mediante cruces por cero y se calcula la amplitud, energía y duración promedio.
    Retorna un vector: [amp, energy, duration]
    """
    cruces = np.where(np.diff(np.signbit(bw_signal)))[0]
    ciclos = []
    for i in range(0, len(cruces)-1, 2):
        inicio = cruces[i]
        fin = cruces[i+1]
        ciclo = bw_signal[inicio:fin+1]
        if len(ciclo) == 0:
            continue
        amp = (np.max(ciclo) - np.min(ciclo)) / 2.0
        energy = np.sum(ciclo**2)
        duration = t[fin] - t[inicio] if fin < len(t) else 0
        ciclos.append([amp, energy, duration])
    if len(ciclos) == 0:
        return np.array([0, 0, 0])
    return np.mean(np.array(ciclos), axis=0)

def extract_vibration_features(vib_signal, fs):
    """
    Extrae características de la señal de vibración.
    Usa:
      - La energía de los coeficientes de detalle de niveles 2 y 3 (DWT con la wavelet 'db5')
      - La frecuencia dominante mediante FFT.
    Retorna un vector: [energy_lvl2, energy_lvl3, dominant_freq]
    """
    # Realizar DWT
    coeffs = pywt.wavedec(vib_signal, 'db5', level=3)
    # Los coeficientes de detalle del nivel 2 y 3
    detail_lvl2 = coeffs[-2]
    detail_lvl3 = coeffs[-1]
    energy_lvl2 = np.sum(np.square(detail_lvl2))
    energy_lvl3 = np.sum(np.square(detail_lvl3))
    
    # FFT para obtener frecuencia dominante
    N = len(vib_signal)
    if N < 2:
        dominant_freq = 0.0
    else:
        window = hann(N)
        sig_win = vib_signal * window
        fft_vals = np.abs(rfft(sig_win))
        freqs = rfftfreq(N, 1.0/fs)
        idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq = freqs[idx]
        
    return np.array([energy_lvl2, energy_lvl3, dominant_freq])

def fuse_features(force_features, vib_features):
    """
    Fusiona las características extraídas de fuerza y vibración en un solo vector.
    """
    return np.concatenate((force_features, vib_features))
