"""Preprocesamiento del dataset (punto V.a del informe).

Lee data/raw/<estudiante>/*.wav y genera data/processed/dataset.npz con:
  - X_vec : vectores MFCC, media + desviación (N, 80) -> entrada del clustering
  - y     : estudiante de cada segmento (solo para EVALUAR, nunca para agrupar)
  - split : 'train' (se agrupa) / 'test' (se usa para probar la asignación)
  - clases: nombres de los estudiantes

Pasos: cargar a 16 kHz mono -> recortar silencios -> normalizar volumen ->
segmentar en ventanas de 2 s -> extraer MFCC.

La división se hace POR ARCHIVO (no por segmento) para que trozos del mismo
audio no queden a la vez en ambos conjuntos. Si todos los estudiantes tienen
grabaciones de la sesión 2 (archivos *_s2_*), esas se usan como prueba.

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

EXTENSIONES = {".wav", ".mp3", ".flac", ".ogg"}


def listar_archivos():
    archivos, etiquetas = [], []
    for carpeta in sorted(p for p in C.DATA_RAW.iterdir() if p.is_dir()):
        wavs = sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXTENSIONES)
        if len(wavs) < C.MIN_AUDIOS:
            print(f"  ! '{carpeta.name}' omitido: solo {len(wavs)} audios (mínimo {C.MIN_AUDIOS})")
            continue
        archivos += wavs
        etiquetas += [carpeta.name] * len(wavs)
    return archivos, etiquetas


def dividir(archivos, y):
    idx = np.arange(len(archivos))
    sesion2 = np.array(["_s2_" in f.name for f in archivos])
    split = np.full(len(archivos), "train", dtype=object)
    if sesion2.any() and {y[i] for i in idx[sesion2]} == set(y):
        print("   sesión 1 -> agrupar (train) | sesión 2 -> probar (test)")
        split[sesion2] = "test"
    else:
        print(f"   división aleatoria: {1 - C.PROP_TEST:.0%} agrupar (train) | "
              f"{C.PROP_TEST:.0%} probar (test)")
        _, test = train_test_split(idx, test_size=C.PROP_TEST, stratify=y,
                                   random_state=C.SEMILLA)
        split[test] = "test"
    return split


def main():
    if not C.DATA_RAW.exists() or not any(p.is_dir() for p in C.DATA_RAW.iterdir()):
        sys.exit("No hay audios en data/raw/. Graba primero las voces.")

    print("1) Leyendo archivos...")
    archivos, nombres = listar_archivos()
    clases = sorted(set(nombres))
    if len(clases) < 2:
        sys.exit(f"Se necesitan al menos 2 estudiantes con {C.MIN_AUDIOS}+ audios.")
    y_arch = np.array([clases.index(n) for n in nombres])
    print(f"   {len(archivos)} audios de {len(clases)} estudiantes")

    print("2) Dividiendo por archivo...")
    split_arch = dividir(archivos, y_arch)

    print("3) Limpiando, segmentando y extrayendo MFCC...")
    X_vec, y, split = [], [], []
    for k, (ruta, etiqueta, s) in enumerate(zip(archivos, y_arch, split_arch)):
        try:
            audio = A.limpiar(A.cargar_audio(ruta))
        except Exception as e:
            print(f"  ! no se pudo leer {ruta.name}: {e}")
            continue
        for seg in A.segmentar(audio):
            X_vec.append(A.vector_mfcc(seg))
            y.append(etiqueta)
            split.append(s)
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(archivos)}")

    X_vec, y, split = np.stack(X_vec), np.array(y), np.array(split)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    salida = C.DATA_PROCESSED / "dataset.npz"
    np.savez_compressed(salida, X_vec=X_vec, y=y, split=split, clases=np.array(clases))

    print("\nResumen:")
    for s in ("train", "test"):
        print(f"   {s:5s}: {np.sum(split == s):5d} segmentos")
    conteo = Counter(y[split == "train"])
    print(f"   segmentos por estudiante: min={min(conteo.values())}, "
          f"max={max(conteo.values())}  (si difieren mucho hay desequilibrio)")
    print(f"   X_vec {X_vec.shape}")
    print(f"\nGuardado en {salida.relative_to(C.RAIZ)}")


if __name__ == "__main__":
    main()
