"""Identificación de un estudiante a partir de un audio usando K-Means.

La voz nueva se divide en segmentos; cada segmento se asigna al centroide más
cercano y se vota. Si la voz queda lejos de todos los centroides, se rechaza
como desconocida.
"""
import joblib
import numpy as np

from src import config as C
from src import audio_utils as A

DESCONOCIDO = "No reconocido"
RUTA_MODELO = C.MODELS / "clustering.joblib"


class Predictor:
    def __init__(self):
        self.m = joblib.load(RUTA_MODELO)

    @staticmethod
    def disponible():
        return RUTA_MODELO.exists()

    def identificar(self, audio):
        m = self.m
        segmentos = A.audio_a_segmentos(audio)
        Z = m["scaler"].transform(np.stack([A.vector_mfcc(s) for s in segmentos]))
        clusters = m["kmeans"].predict(Z)
        dist = float(np.median(m["kmeans"].transform(Z).min(axis=1)))
        votos = np.bincount([m["mapa"][int(c)] for c in clusters if m["mapa"][int(c)] >= 0],
                            minlength=len(m["clases"]))
        i = int(votos.argmax())
        confianza = float(votos[i] / max(votos.sum(), 1))
        nombre = DESCONOCIDO if (dist > m["umbral"] or votos.sum() == 0) else m["clases"][i]
        return {"estudiante": nombre, "confianza": confianza,
                "detalle": f"Cluster más cercano a distancia {dist:.2f} "
                           f"(umbral de rechazo {m['umbral']:.2f}) · "
                           f"{votos[i]} de {len(segmentos)} segmentos votaron por "
                           f"{m['clases'][i]}"}
