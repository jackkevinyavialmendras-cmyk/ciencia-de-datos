import io

import streamlit as st

import comun
from src import config as C
from src.predictor import DESCONOCIDO, Predictor

st.title("✅ Tomar asistencia")

if not Predictor.disponible():
    st.warning("Aún no se ejecutó el clustering.")
    st.page_link("paginas/entrenar.py", label="Ir a agrupar voces →", icon="🧠")
    st.stop()

modelo = "K-Means"
st.caption("Modelo: **K-Means** — la voz se asigna al grupo (cluster) más cercano; "
           "si está lejos de todos, se rechaza como desconocida.")


def mostrar(r, registrar=True):
    if r["estudiante"] == DESCONOCIDO:
        st.error(f"### ❌ Voz no reconocida\nConfianza {r['confianza']:.0%}. Intenta de nuevo "
                 "o pide al docente registrarte manualmente.")
    else:
        nombre = comun.bonito(r["estudiante"])
        if registrar:
            nuevo, hora = comun.registrar_asistencia(r["estudiante"], modelo, r["confianza"])
            if nuevo:
                st.success(f"### ✅ {nombre} — PRESENTE\nRegistrado a las {hora} · "
                           f"confianza {r['confianza']:.0%}")
                st.balloons()
            else:
                st.info(f"### ℹ️ {nombre}\nYa registró asistencia hoy a las {hora}.")
        else:
            st.success(f"### Identificado: {nombre}\nConfianza {r['confianza']:.0%}")
    st.caption(r["detalle"])


st.markdown("#### Di **«Presente»** seguido de tu nombre completo")
audio = st.audio_input("Grabar voz", key="asistencia", label_visibility="collapsed")
if audio is not None:
    with st.spinner("Identificando la voz..."):
        r = comun.predictor().identificar(io.BytesIO(audio.getvalue()))
    mostrar(r)

# --- Demostración con audios guardados (útil en la defensa) ---------------------------
with st.expander("🧪 Probar con un audio guardado (sin registrar asistencia)"):
    origen = st.radio("Origen", ["Voces desconocidas", "Dataset (sesión 2)"], horizontal=True)
    if origen == "Voces desconocidas":
        archivos = sorted(C.DATA_DESCONOCIDOS.glob("*/*.wav")) if C.DATA_DESCONOCIDOS.exists() else []
    else:
        archivos = sorted(C.DATA_RAW.glob("*/*_s2_*.wav")) if C.DATA_RAW.exists() else []
    if not archivos:
        st.caption("No hay audios de este tipo. Graba voces desconocidas en «Grabar voces».")
    else:
        f = st.selectbox("Audio", archivos, format_func=lambda p: f"{p.parent.name} / {p.name}")
        st.audio(str(f))
        if st.button("Identificar"):
            with st.spinner("Identificando..."):
                r = comun.predictor().identificar(str(f))
            st.write(f"Voz real: **{comun.bonito(f.parent.name)}**")
            mostrar(r, registrar=False)
