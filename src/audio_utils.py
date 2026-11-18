"""Preprocesamiento de audio y extracción de características.

Se usan las mismas funciones en entrenamiento y en la aplicación, para que
el audio en vivo pase exactamente por el mismo proceso que el dataset.
"""
import numpy as np
import librosa

from src import config as C


# --- Limpieza ------------------------------------------------------------
def cargar_audio(ruta_o_archivo):
    """Carga cualquier audio (wav, mp3, bytes de Streamlit) a 16 kHz mono."""
    y, _ = librosa.load(ruta_o_archivo, sr=C.SR, mono=True)
    return y


def limpiar(y):
    """Recorta silencios del inicio/fin y normaliza el volumen a [-1, 1]."""
    y, _ = librosa.effects.trim(y, top_db=C.TOP_DB)
    pico = np.max(np.abs(y)) if y.size else 0
    return y / pico if pico > 0 else y


def segmentar(y):
    """Divide el audio en ventanas de DURACION_SEG con SALTO_SEG de salto.

    Si el audio es más corto que una ventana se rellena con ceros.
    """
    n, salto = C.MUESTRAS_SEG, int(C.SR * C.SALTO_SEG)
    if len(y) <= n:
        return [np.pad(y, (0, n - len(y)))]
    return [y[i:i + n] for i in range(0, len(y) - n + 1, salto)]


# --- Aumento de datos (solo para entrenamiento) ------------------------------
def aumentar(y, rng):
    """Genera variantes del audio para combatir el sobreajuste."""
    variantes = []
    ruido = rng.normal(0, 0.005, len(y))                      # ruido de aula
    variantes.append(np.clip(y + ruido, -1, 1))
    pasos = rng.choice([-2, -1, 1, 2])                        # cambio de tono
    variantes.append(librosa.effects.pitch_shift(y, sr=C.SR, n_steps=float(pasos)))
    ganancia = rng.uniform(0.6, 0.9)                          # hablar más bajo
    variantes.append(y * ganancia)
    return variantes


# --- Características ---------------------------------------------------------
def espectrograma_mel(seg):
    """Espectrograma log-Mel estandarizado: 'imagen' de entrada a la CNN."""
    mel = librosa.feature.melspectrogram(
        y=seg, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP, n_mels=C.N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)).astype(np.float32)


def mfcc_secuencia(seg):
    """Matriz MFCC (frames x N_MFCC), útil para una LSTM/GRU."""
    m = librosa.feature.mfcc(y=seg, sr=C.SR, n_mfcc=C.N_MFCC,
                             n_fft=C.N_FFT, hop_length=C.HOP)
    return m.T.astype(np.float32)


def vector_mfcc(seg):
    """Vector fijo de 2*N_MFCC valores (media y desviación de cada MFCC).

    Es la representación que usan K-Means, jerárquico y DBSCAN.
    """
    m = librosa.feature.mfcc(y=seg, sr=C.SR, n_mfcc=C.N_MFCC,
                             n_fft=C.N_FFT, hop_length=C.HOP)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def audio_a_segmentos(ruta_o_archivo):
    """Pipeline completo para un audio nuevo: cargar -> limpiar -> segmentar."""
    return segmentar(limpiar(cargar_audio(ruta_o_archivo)))
