"""Funciones compartidas por las páginas de la interfaz."""
import json
import re
import sqlite3
import unicodedata
from datetime import datetime

import pandas as pd
import soundfile as sf
import streamlit as st

from src import config as C
from src.predictor import Predictor

EXT_AUDIO = {".wav", ".mp3", ".flac", ".ogg"}


# --- Nombres y archivos ------------------------------------------------------------
def normalizar_nombre(texto):
    """'José Núñez ' -> 'jose_nunez' (sin tildes, ñ ni espacios)."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
    return t


def bonito(nombre):
    return nombre.replace("_", " ").title()


def audios_de(carpeta, sesion=None):
    if not carpeta.exists():
        return []
    fs = [f for f in carpeta.iterdir() if f.suffix.lower() in EXT_AUDIO]
    if sesion is not None:
        fs = [f for f in fs if f"_s{sesion}_" in f.name]
    return sorted(fs)


def guardar_audio(y, carpeta, nombre, sesion):
    """Guarda el audio (16 kHz mono) con el siguiente número libre."""
    carpeta.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for f in carpeta.glob(f"{nombre}_s{sesion}_*.wav")
            if (m := re.search(r"_(\d+)\.wav$", f.name))]
    ruta = carpeta / f"{nombre}_s{sesion}_{(max(nums) + 1) if nums else 0:03d}.wav"
    sf.write(ruta, y, C.SR)
    return ruta


def inventario():
    """Tabla con cuántos audios tiene cada persona registrada."""
    filas = []
    if C.DATA_RAW.exists():
        for carpeta in sorted(p for p in C.DATA_RAW.iterdir() if p.is_dir()):
            todos = audios_de(carpeta)
            s1 = sum("_s1_" in f.name for f in todos)
            s2 = sum("_s2_" in f.name for f in todos)
            total = len(todos)
            if total < C.MIN_AUDIOS:
                estado = f"❌ faltan audios (mín. {C.MIN_AUDIOS})"
            elif total < C.AUDIOS_RECOMENDADOS:
                estado = "⚠️ pocos audios"
            elif s2 == 0:
                estado = "⚠️ falta sesión 2"
            else:
                estado = "✅ listo"
            filas.append({"estudiante": carpeta.name, "sesión 1": s1, "sesión 2": s2,
                          "otros": total - s1 - s2, "total": total, "estado": estado})
    return pd.DataFrame(filas, columns=["estudiante", "sesión 1", "sesión 2", "otros",
                                        "total", "estado"])


def ultimo_audio_mtime():
    fs = list(C.DATA_RAW.glob("*/*")) if C.DATA_RAW.exists() else []
    return max((f.stat().st_mtime for f in fs), default=0)


# --- Métricas ----------------------------------------------------------------------
def metricas():
    out = {}
    for f in sorted(C.REPORTS.glob("metricas_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["modelo"]] = d
    return out


# --- Predictor (se recarga tras el clustering) ------------------------------------------
@st.cache_resource
def predictor():
    return Predictor()


# --- Base de datos de asistencia ---------------------------------------------------
def db():
    C.DB_ASISTENCIA.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.DB_ASISTENCIA)
    con.execute("""CREATE TABLE IF NOT EXISTS asistencia (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        estudiante TEXT, fecha TEXT, hora TEXT, modelo TEXT, confianza REAL)""")
    return con


def registrar_asistencia(estudiante, modelo, confianza):
    hoy = datetime.now().strftime("%Y-%m-%d")
    con = db()
    try:
        ya = con.execute("SELECT hora FROM asistencia WHERE estudiante=? AND fecha=?",
                         (estudiante, hoy)).fetchone()
        if ya:
            return False, ya[0]
        hora = datetime.now().strftime("%H:%M:%S")
        con.execute("INSERT INTO asistencia (estudiante, fecha, hora, modelo, confianza) "
                    "VALUES (?,?,?,?,?)", (estudiante, hoy, hora, modelo, confianza))
        con.commit()
        return True, hora
    finally:
        con.close()


def leer_asistencia():
    con = db()
    try:
        return pd.read_sql("SELECT id, estudiante, fecha, hora, modelo, confianza "
                           "FROM asistencia ORDER BY fecha DESC, hora", con)
    finally:
        con.close()


def borrar_asistencia(id_registro):
    con = db()
    try:
        con.execute("DELETE FROM asistencia WHERE id=?", (int(id_registro),))
        con.commit()
    finally:
        con.close()
