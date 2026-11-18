"""Identificación de un estudiante a partir de un audio, con cualquiera de los dos modelos."""
import json

import joblib
import numpy as np

from src import config as C
from src import audio_utils as A

DESCONOCIDO = "No reconocido"


class Predictor:
    def __init__(self):
        self._cnn = None
        self._clu = None
        self.clases = json.loads((C.MODELS / "clases.json").read_text(encoding="utf-8")) \
            if (C.MODELS / "clases.json").exists() else []

    # --- disponibilidad -------------------------------------------------------
    @staticmethod
    def disponibles():
        m = []
        if (C.MODELS / "cnn_voz.keras").exists():
            m.append("CNN")
        if (C.MODELS / "lstm_voz.keras").exists():
            m.append("LSTM")
        if (C.MODELS / "clustering.joblib").exists():
            m.append("K-Means")
        return m

    # --- deep learning --------------------------------------------------------
    def _red(self, tipo):
        import tensorflow as tf   # import tardío: TensorFlow tarda en cargar
        if self._cnn is None or self._cnn[0] != tipo:
            self._cnn = (tipo, tf.keras.models.load_model(C.MODELS / f"{tipo}_voz.keras"))
        return self._cnn[1]

    def predecir_red(self, segmentos, tipo="cnn"):
        X = np.stack([A.espectrograma_mel(s) for s in segmentos])
        X = X[..., np.newaxis] if tipo == "cnn" else np.transpose(X, (0, 2, 1))
        prob = self._red(tipo).predict(X, verbose=0).mean(axis=0)   # promedio de segmentos
        i = int(prob.argmax())
        nombre = self.clases[i] if prob[i] >= C.UMBRAL_CONFIANZA_CNN else DESCONOCIDO
        ranking = sorted(zip(self.clases, prob.tolist()), key=lambda t: -t[1])[:3]
        return nombre, float(prob[i]), ranking

    # --- no supervisado -------------------------------------------------------
    def predecir_kmeans(self, segmentos):
        if self._clu is None:
            self._clu = joblib.load(C.MODELS / "clustering.joblib")
        m = self._clu
        Z = m["scaler"].transform(np.stack([A.vector_mfcc(s) for s in segmentos]))
        clusters = m["kmeans"].predict(Z)
        dist = m["kmeans"].transform(Z).min(axis=1)
        votos = np.bincount([m["mapa"][int(c)] for c in clusters if m["mapa"][int(c)] >= 0],
                            minlength=len(m["clases"]))
        i = int(votos.argmax())
        confianza = float(votos[i] / max(votos.sum(), 1))
        # Si la voz está lejos de todos los centroides -> no pertenece a ningún estudiante
        if np.median(dist) > m["umbral"] or votos.sum() == 0:
            return DESCONOCIDO, confianza, float(np.median(dist))
        return m["clases"][i], confianza, float(np.median(dist))

    # --- interfaz común -------------------------------------------------------
    def identificar(self, audio, modelo="CNN"):
        segmentos = A.audio_a_segmentos(audio)
        if modelo == "K-Means":
            nombre, conf, dist = self.predecir_kmeans(segmentos)
            return {"estudiante": nombre, "confianza": conf,
                    "detalle": f"distancia al centroide: {dist:.2f} (umbral {self._clu['umbral']:.2f})"}
        nombre, conf, ranking = self.predecir_red(segmentos, modelo.lower())
        return {"estudiante": nombre, "confianza": conf,
                "detalle": " | ".join(f"{n}: {p:.0%}" for n, p in ranking)}
