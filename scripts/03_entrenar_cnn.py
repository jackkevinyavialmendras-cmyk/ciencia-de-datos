"""Modelo 1 — Aprendizaje profundo (puntos V.b y V.c del informe).

CNN sobre espectrogramas log-Mel:
  Conv+ReLU -> MaxPool -> Conv+ReLU -> MaxPool -> Conv+ReLU -> MaxPool
  -> Flatten -> Dense+Dropout -> Softmax (un estudiante por clase)

Con --modelo lstm se entrena en cambio una LSTM sobre la secuencia de frames
(para la comparación CNN vs. RNN que se vio en clase).

Problemas comunes atendidos (diapositivas de la docente):
  - Sobreajuste        -> aumento de datos, Dropout, EarlyStopping
  - Desequilibrio      -> class_weight
  - Desvanecimiento    -> ReLU / LSTM

Uso:
    python scripts/03_entrenar_cnn.py
    python scripts/03_entrenar_cnn.py --modelo lstm
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks

from src import config as C


def construir_cnn(n_clases):
    return models.Sequential([
        layers.Input(shape=(C.N_MELS, C.FRAMES_MEL, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_clases, activation="softmax"),
    ], name="cnn_voz")


def construir_lstm(n_clases):
    return models.Sequential([
        layers.Input(shape=(C.FRAMES_MEL, C.N_MELS)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_clases, activation="softmax"),
    ], name="lstm_voz")


def adaptar(X, tipo):
    # CNN: (N, 128, 63, 1) como imagen | LSTM: (N, 63, 128) como secuencia temporal
    return X[..., np.newaxis] if tipo == "cnn" else np.transpose(X, (0, 2, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=["cnn", "lstm"], default="cnn")
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--lote", type=int, default=32)
    a = p.parse_args()

    tf.keras.utils.set_random_seed(C.SEMILLA)
    d = np.load(C.DATA_PROCESSED / "dataset.npz", allow_pickle=True)
    clases = list(d["clases"])
    X, y, split = adaptar(d["X_mel"], a.modelo), d["y"], d["split"]
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"Clases: {len(clases)} | train {tr.sum()} | val {va.sum()} | test {te.sum()}")

    modelo = (construir_cnn if a.modelo == "cnn" else construir_lstm)(len(clases))
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.summary()

    pesos = compute_class_weight("balanced", classes=np.unique(y[tr]), y=y[tr])
    hist = modelo.fit(
        X[tr], y[tr], validation_data=(X[va], y[va]),
        epochs=a.epocas, batch_size=a.lote,
        class_weight=dict(zip(np.unique(y[tr]), pesos)),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ],
        verbose=2,
    )

    # --- Validación sobre test ---------------------------------------------
    pred = modelo.predict(X[te], verbose=0).argmax(axis=1)
    acc = accuracy_score(y[te], pred)
    f1 = f1_score(y[te], pred, average="macro")
    print(f"\nTEST -> accuracy {acc:.4f} | F1-macro {f1:.4f}\n")
    reporte = classification_report(y[te], pred, labels=range(len(clases)),
                                    target_names=clases, zero_division=0)
    print(reporte)

    # --- Guardar modelo, gráficas y métricas --------------------------------
    C.MODELS.mkdir(exist_ok=True)
    C.REPORTS.mkdir(exist_ok=True)
    modelo.save(C.MODELS / f"{a.modelo}_voz.keras")
    (C.MODELS / "clases.json").write_text(json.dumps(clases, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, m in enumerate(["loss", "accuracy"]):
        ax[i].plot(hist.history[m], label="entrenamiento")
        ax[i].plot(hist.history[f"val_{m}"], label="validación")
        ax[i].set_title(f"{a.modelo.upper()} — {m}")
        ax[i].set_xlabel("época")
        ax[i].legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS / f"{a.modelo}_curvas.png", dpi=150)

    cm = confusion_matrix(y[te], pred, labels=range(len(clases)))
    fig, ax = plt.subplots(figsize=(max(6, len(clases) * 0.5), max(5, len(clases) * 0.45)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases,
                yticklabels=clases, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {a.modelo.upper()}")
    fig.tight_layout()
    fig.savefig(C.REPORTS / f"{a.modelo}_matriz_confusion.png", dpi=150)

    (C.REPORTS / f"{a.modelo}_reporte.txt").write_text(reporte, encoding="utf-8")
    (C.REPORTS / f"metricas_{a.modelo}.json").write_text(json.dumps({
        "modelo": a.modelo.upper(),
        "accuracy_test": round(acc, 4),
        "f1_macro_test": round(f1, 4),
        "epocas_entrenadas": len(hist.history["loss"]),
        "parametros": int(modelo.count_params()),
        "requiere_etiquetas": True,
    }, indent=2), encoding="utf-8")
    print(f"Modelo y reportes guardados en models/ y reports/")


if __name__ == "__main__":
    main()
