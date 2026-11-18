"""Configuración central del proyecto (enfoque: aprendizaje no supervisado)."""
from pathlib import Path

# --- Rutas -------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"              # data/raw/<estudiante>/*.wav
DATA_PROCESSED = RAIZ / "data" / "processed"  # dataset.npz generado
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"                    # gráficas y métricas para el informe
DB_ASISTENCIA = RAIZ / "data" / "asistencia.db"
DATA_DESCONOCIDOS = RAIZ / "data" / "desconocidos"  # voces NO registradas (prueba de rechazo)

# --- Dataset -----------------------------------------------------------
MIN_AUDIOS = 5            # mínimo por persona para entrar al clustering
AUDIOS_RECOMENDADOS = 20  # recomendado por persona (sesión 1)
FRASES = [
    "Di tu nombre completo",
    "Presente, licenciada",
    "Cuenta del uno al diez",
    "Hoy es un buen día para aprender",
    "Mi código de estudiante es... (dilo)",
    "El aprendizaje automático reconoce mi voz",
    "Di los días de la semana",
    "Lee en voz alta: tres, siete, cinco, nueve",
    "Buenos días a todos",
    "Estoy en la clase de Machine Learning",
    "Presente, y mi nombre es... (dilo)",
    "Di los meses del año que recuerdes",
]

# --- Audio -------------------------------------------------------------
SR = 16000            # frecuencia de muestreo (Hz)
DURACION_SEG = 2.0    # duración de cada segmento (s)
SALTO_SEG = 1.0       # salto entre segmentos (s) -> 50 % de solapamiento
TOP_DB = 25           # umbral para recortar silencios (dB)

# --- Características (MFCC) ---------------------------------------------
N_FFT = 1024
HOP = 512
N_MFCC = 40           # coeficientes MFCC -> vector de 80 valores (media + desviación)

# --- Clustering ----------------------------------------------------------
SEMILLA = 42
PROP_TEST = 0.20              # audios reservados para probar la asignación a clusters
PERCENTIL_DESCONOCIDO = 99    # distancia al centroide para rechazar en K-Means...
FACTOR_UMBRAL = 1.25          # ...con un margen (las voces de otro día varían más)

MUESTRAS_SEG = int(SR * DURACION_SEG)
