"""Prototipo: control de asistencia por reconocimiento de voz.

Ejecutar desde la raíz del proyecto:
    streamlit run app/app.py
"""
import io
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf
import streamlit as st

from src import config as C
from src import audio_utils as A
from src.predictor import DESCONOCIDO, Predictor

st.set_page_config(page_title="Asistencia por voz", page_icon="🎙️", layout="centered")


# --- Base de datos de asistencia -------------------------------------------------
def db():
    C.DB_ASISTENCIA.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.DB_ASISTENCIA)
    con.execute("""CREATE TABLE IF NOT EXISTS asistencia (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        estudiante TEXT, fecha TEXT, hora TEXT, modelo TEXT, confianza REAL)""")
    return con


def registrar_asistencia(estudiante, modelo, confianza):
    hoy = datetime.now().strftime("%Y-%m-%d")
    with db() as con:
        ya = con.execute("SELECT hora FROM asistencia WHERE estudiante=? AND fecha=?",
                         (estudiante, hoy)).fetchone()
        if ya:
            return False, ya[0]
        hora = datetime.now().strftime("%H:%M:%S")
        con.execute("INSERT INTO asistencia (estudiante, fecha, hora, modelo, confianza) "
                    "VALUES (?,?,?,?,?)", (estudiante, hoy, hora, modelo, confianza))
        return True, hora


@st.cache_resource
def cargar_predictor():
    return Predictor()


st.title("🎙️ Asistencia por reconocimiento de voz")
tab_asis, tab_reg, tab_rep = st.tabs(["Tomar asistencia", "Registrar estudiante", "Reporte"])

# --- Tomar asistencia ------------------------------------------------------------
with tab_asis:
    modelos = Predictor.disponibles()
    if not modelos:
        st.warning("Aún no hay modelos entrenados. Ejecuta los scripts 02, 03 y 04 (ver README).")
    else:
        modelo = st.radio("Modelo", modelos, horizontal=True,
                          help="CNN/LSTM = aprendizaje profundo · K-Means = no supervisado")
        st.write("Di **«Presente»** seguido de tu nombre completo (unos 3 segundos).")
        audio = st.audio_input("Grabar voz", key="asistencia")
        if audio is not None:
            with st.spinner("Identificando..."):
                r = cargar_predictor().identificar(io.BytesIO(audio.getvalue()), modelo)
            if r["estudiante"] == DESCONOCIDO:
                st.error(f"❌ Voz no reconocida (confianza {r['confianza']:.0%}). Intenta de nuevo "
                         "o pide al docente registrarte manualmente.")
            else:
                nuevo, hora = registrar_asistencia(r["estudiante"], modelo, r["confianza"])
                nombre = r["estudiante"].replace("_", " ").title()
                if nuevo:
                    st.success(f"✅ **{nombre}** — PRESENTE a las {hora} "
                               f"(confianza {r['confianza']:.0%})")
                else:
                    st.info(f"ℹ️ {nombre} ya registró asistencia hoy a las {hora}.")
            st.caption(r["detalle"])

# --- Registrar estudiante -----------------------------------------------------------
with tab_reg:
    st.write("Graba al menos **20 muestras** por estudiante. Luego vuelve a ejecutar los "
             "scripts 02, 03 y 04 para reentrenar los modelos.")
    nombre = st.text_input("Identificador (ej. juan_perez)").strip().lower().replace(" ", "_")
    sesion = st.selectbox("Sesión", [1, 2], help="Graben la sesión 2 otro día: se usa como prueba")
    if nombre:
        carpeta = C.DATA_RAW / nombre
        existentes = sorted(carpeta.glob(f"{nombre}_s{sesion}_*.wav")) if carpeta.exists() else []
        st.write(f"Muestras guardadas en sesión {sesion}: **{len(existentes)}**")
        muestra = st.audio_input("Grabar muestra", key=f"reg_{nombre}_{sesion}_{len(existentes)}")
        if muestra is not None and st.button("💾 Guardar muestra"):
            carpeta.mkdir(parents=True, exist_ok=True)
            y = A.cargar_audio(io.BytesIO(muestra.getvalue()))
            sf.write(carpeta / f"{nombre}_s{sesion}_{len(existentes):03d}.wav", y, C.SR)
            st.rerun()

# --- Reporte -----------------------------------------------------------------------
with tab_rep:
    with db() as con:
        df = pd.read_sql("SELECT estudiante, fecha, hora, modelo, confianza FROM asistencia "
                         "ORDER BY fecha DESC, hora", con)
    if df.empty:
        st.info("Todavía no hay registros de asistencia.")
    else:
        fecha = st.selectbox("Fecha", sorted(df["fecha"].unique(), reverse=True))
        dia = df[df["fecha"] == fecha].copy()
        dia["confianza"] = (dia["confianza"] * 100).round(1).astype(str) + " %"
        st.metric("Presentes", len(dia))
        st.dataframe(dia, use_container_width=True, hide_index=True)
        excel = io.BytesIO()
        dia.to_excel(excel, index=False)
        st.download_button("⬇️ Descargar Excel", excel.getvalue(),
                           file_name=f"asistencia_{fecha}.xlsx")
