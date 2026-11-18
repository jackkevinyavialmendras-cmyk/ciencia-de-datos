"""Graba el dataset propio desde el micrófono.

Uso:
    python scripts/01_grabar_dataset.py --nombre juan_perez --sesion 1 --cantidad 30

Guarda los audios en data/raw/<nombre>/<nombre>_s<sesion>_<nnn>.wav
Graben a cada persona en 2 sesiones (días distintos) para una validación honesta.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
import soundfile as sf

from src import config as C

FRASES = C.FRASES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nombre", required=True, help="identificador, ej. juan_perez")
    p.add_argument("--sesion", type=int, default=1)
    p.add_argument("--cantidad", type=int, default=30)
    p.add_argument("--segundos", type=float, default=3.0)
    a = p.parse_args()

    carpeta = C.DATA_RAW / a.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    inicio = len(list(carpeta.glob(f"{a.nombre}_s{a.sesion}_*.wav")))

    print(f"\nGrabando a '{a.nombre}' (sesión {a.sesion}), {a.cantidad} audios de {a.segundos}s.")
    print("Presiona Enter antes de cada grabación (Ctrl+C para salir).\n")
    for i in range(inicio, inicio + a.cantidad):
        frase = FRASES[i % len(FRASES)]
        input(f"[{i + 1 - inicio}/{a.cantidad}] Frase: «{frase}»  -> Enter para grabar")
        time.sleep(0.2)
        audio = sd.rec(int(a.segundos * C.SR), samplerate=C.SR, channels=1, dtype="float32")
        sd.wait()
        ruta = carpeta / f"{a.nombre}_s{a.sesion}_{i:03d}.wav"
        sf.write(ruta, audio, C.SR)
        print(f"   guardado: {ruta.relative_to(C.RAIZ)}")
    print("\nListo.")


if __name__ == "__main__":
    main()
