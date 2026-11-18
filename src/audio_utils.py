"""Preprocesamiento de audio y extracción de características.

Se usan las mismas funciones en el clustering y en la aplicación, para que
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


# --- Características ---------------------------------------------------------
def vector_mfcc(seg):
    """Vector fijo de 2*N_MFCC valores (media y desviación de cada MFCC).

    Es la representación que usan K-Means, jerárquico y DBSCAN.
    """
    m = librosa.feature.mfcc(y=seg, sr=C.SR, n_mfcc=C.N_MFCC,
                             n_fft=C.N_FFT, hop_length=C.HOP)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def evaluar_calidad(y):
    """Revisa una grabación cruda (16 kHz). Devuelve (ok, mensaje)."""
    if y.size == 0:
        return False, "La grabación está vacía."
    if np.mean(np.abs(y) > 0.99) > 0.01:
        return False, "El audio está saturado (demasiado fuerte o muy cerca del micrófono)."
    if np.sqrt(np.mean(y ** 2)) < 0.005:
        return False, "Se escucha muy bajo o en silencio. Habla más cerca del micrófono."
    voz, _ = librosa.effects.trim(y, top_db=C.TOP_DB)
    if len(voz) / C.SR < 0.8:
        return False, "Hay muy poca voz (menos de 1 segundo). Repite la frase completa."
    return True, f"Bien: {len(voz) / C.SR:.1f} s de voz."


def audio_a_segmentos(ruta_o_archivo):
    """Pipeline completo para un audio nuevo: cargar -> limpiar -> segmentar."""
    return segmentar(limpiar(cargar_audio(ruta_o_archivo)))
