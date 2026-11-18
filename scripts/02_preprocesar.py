"""Preprocesamiento del dataset (punto V.a del informe).

Lee data/raw/<estudiante>/*.wav y genera data/processed/dataset.npz con:
  - X_mel : espectrogramas log-Mel (N, 128, 63)  -> CNN
  - X_vec : vectores MFCC media+desv (N, 80)    -> clustering
  - y     : etiqueta numérica de cada segmento
  - split : 'train' / 'val' / 'test'
  - clases: nombres de los estudiantes

La división se hace POR ARCHIVO (no por segmento) para que trozos del mismo
audio no queden a la vez en entrenamiento y prueba. Si existen grabaciones de
la sesión 2 (archivos *_s2_*), se usan como conjunto de prueba.
El aumento de datos se aplica solo a entrenamiento.

Uso:
    python scripts/02_preprocesar.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from src import config as C
from src import audio_utils as A

EXTENSIONES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MIN_ARCHIVOS = 5


def listar_archivos():
    archivos, etiquetas = [], []
    for carpeta in sorted(p for p in C.DATA_RAW.iterdir() if p.is_dir()):
        wavs = sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXTENSIONES)
        if len(wavs) < MIN_ARCHIVOS:
            print(f"  ! '{carpeta.name}' omitido: solo {len(wavs)} audios (mínimo {MIN_ARCHIVOS})")
            continue
        archivos += wavs
        etiquetas += [carpeta.name] * len(wavs)
    return archivos, etiquetas


def dividir(archivos, y):
    idx = np.arange(len(archivos))
    sesion2 = np.array(["_s2_" in f.name for f in archivos])
    clases_con_s2 = {y[i] for i in idx[sesion2]}
    if sesion2.any() and clases_con_s2 == set(y):
        print("  División: sesión 1 -> train/val, sesión 2 -> test")
        test = idx[sesion2]
        resto = idx[~sesion2]
    else:
        resto, test = train_test_split(idx, test_size=C.PROP_TEST, stratify=y,
                                       random_state=C.SEMILLA)
    val_rel = C.PROP_VAL / (1 - C.PROP_TEST)
    train, val = train_test_split(resto, test_size=val_rel, stratify=y[resto],
                                  random_state=C.SEMILLA)
    split = np.empty(len(archivos), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return split


def main():
    if not C.DATA_RAW.exists() or not any(C.DATA_RAW.iterdir()):
        sys.exit("No hay audios en data/raw/. Graba primero con scripts/01_grabar_dataset.py")

    print("1) Leyendo archivos...")
    archivos, nombres = listar_archivos()
    clases = sorted(set(nombres))
    y_arch = np.array([clases.index(n) for n in nombres])
    print(f"   {len(archivos)} audios de {len(clases)} estudiantes")

    print("2) Dividiendo train / val / test por archivo...")
    split_arch = dividir(archivos, y_arch)

    print("3) Limpiando, segmentando, aumentando y extrayendo características...")
    rng = np.random.default_rng(C.SEMILLA)
    X_mel, X_vec, y, split, origen = [], [], [], [], []
    for k, (ruta, etiqueta, s) in enumerate(zip(archivos, y_arch, split_arch)):
        y_audio = A.limpiar(A.cargar_audio(ruta))
        versiones = [y_audio] + (A.aumentar(y_audio, rng) if s == "train" else [])
        for v, audio in enumerate(versiones):
            for seg in A.segmentar(audio):
                X_mel.append(A.espectrograma_mel(seg))
                X_vec.append(A.vector_mfcc(seg))
                y.append(etiqueta)
                split.append(s)
                origen.append(v > 0)   # True = segmento aumentado
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(archivos)}")

    X_mel, X_vec = np.stack(X_mel), np.stack(X_vec)
    y, split, origen = np.array(y), np.array(split), np.array(origen)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    salida = C.DATA_PROCESSED / "dataset.npz"
    np.savez_compressed(salida, X_mel=X_mel, X_vec=X_vec, y=y, split=split,
                        aumentado=origen, clases=np.array(clases))

    print("\nResumen:")
    for s in ("train", "val", "test"):
        print(f"   {s:5s}: {np.sum(split == s):5d} segmentos")
    conteo = Counter(y[split == "train"])
    print(f"   segmentos de train por estudiante: min={min(conteo.values())}, "
          f"max={max(conteo.values())}  (si difieren mucho hay desequilibrio)")
    print(f"   X_mel {X_mel.shape} | X_vec {X_vec.shape}")
    print(f"\nGuardado en {salida.relative_to(C.RAIZ)}")


if __name__ == "__main__":
    main()
