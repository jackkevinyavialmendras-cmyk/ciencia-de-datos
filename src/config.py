"""Configuración central del proyecto: rutas y parámetros de audio."""
from pathlib import Path

# --- Rutas -------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"              # data/raw/<estudiante>/*.wav
DATA_PROCESSED = RAIZ / "data" / "processed"  # dataset.npz generado
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"                    # gráficas y métricas para el informe
DB_ASISTENCIA = RAIZ / "data" / "asistencia.db"

# --- Audio -------------------------------------------------------------
SR = 16000            # frecuencia de muestreo (Hz)
DURACION_SEG = 2.0    # duración de cada segmento que entra a los modelos (s)
SALTO_SEG = 1.0       # salto entre segmentos (s) -> 50 % de solapamiento
TOP_DB = 25           # umbral para recortar silencios (dB)

# --- Características ---------------------------------------------------
N_FFT = 1024
HOP = 512
N_MELS = 128          # alto del espectrograma Mel (entrada de la CNN)
N_MFCC = 40           # coeficientes MFCC (entrada del clustering)

# --- Entrenamiento -----------------------------------------------------
SEMILLA = 42
PROP_TEST = 0.15
PROP_VAL = 0.15
UMBRAL_CONFIANZA_CNN = 0.70   # por debajo -> "no reconocido"
PERCENTIL_DESCONOCIDO = 95    # distancia al centroide para rechazar en K-Means

MUESTRAS_SEG = int(SR * DURACION_SEG)
FRAMES_MEL = 1 + MUESTRAS_SEG // HOP   # 63 columnas para 2 s
