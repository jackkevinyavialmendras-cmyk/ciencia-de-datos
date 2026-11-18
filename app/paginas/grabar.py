import io

import streamlit as st

import comun
from src import audio_utils as A
from src import config as C

st.title("🎙️ Grabar voces")

ss = st.session_state
ss.setdefault("contador_grab", 0)
ss.setdefault("ultimo_audio", None)

modo = st.radio("¿Qué vas a grabar?",
                ["Estudiante (para el dataset)", "Voz desconocida (para probar el rechazo)"],
                horizontal=True)
es_desconocido = modo.startswith("Voz desconocida")

c1, c2, c3 = st.columns([2, 1, 1])
nombre = comun.normalizar_nombre(c1.text_input(
    "Nombre completo", placeholder="Ej. Juan Pérez",
    help="Se guarda sin tildes ni espacios: juan_perez"))
sesion = c2.selectbox("Sesión", [1, 2],
                      help="Graben la sesión 2 otro día: se usa para examinar al modelo")
meta = c3.number_input("Meta de audios", 5, 100, 25 if sesion == 1 else 10)

if not nombre:
    st.info("Escribe el nombre de la persona para empezar.")
    st.stop()

base = C.DATA_DESCONOCIDOS if es_desconocido else C.DATA_RAW
carpeta = base / nombre
existentes = comun.audios_de(carpeta, sesion)
n = len(existentes)

st.caption(f"Se guarda en `{carpeta.relative_to(C.RAIZ)}`")
st.progress(min(n / meta, 1.0), text=f"{comun.bonito(nombre)} · sesión {sesion}: {n} de {meta} audios")
if n >= meta:
    st.success("🎉 ¡Meta cumplida! Puedes seguir grabando más o pasar a la siguiente persona.")

# --- Grabación guiada ---------------------------------------------------------------
frase = C.FRASES[n % len(C.FRASES)]
st.markdown(f"### Di: «{frase}»")
st.caption("1) Presiona el micrófono · 2) Di la frase con voz normal · 3) Presiona de nuevo para "
           "terminar. Se guarda automáticamente si la calidad es buena.")

grabacion = st.audio_input("Grabar", key=f"grab_{nombre}_{sesion}_{ss.contador_grab}",
                           label_visibility="collapsed")
if grabacion is not None:
    y = A.cargar_audio(io.BytesIO(grabacion.getvalue()))
    ok, mensaje = A.evaluar_calidad(y)
    if ok:
        ruta = comun.guardar_audio(y, carpeta, nombre, sesion)
        ss.ultimo_audio = str(ruta)
        ss.contador_grab += 1
        st.toast(f"✅ Guardado: {ruta.name} — {mensaje}")
        st.rerun()
    else:
        st.error(f"❌ No se guardó. {mensaje} Vuelve a grabar.")

if ss.ultimo_audio:
    from pathlib import Path
    ultimo = Path(ss.ultimo_audio)
    if ultimo.exists():
        with st.container(border=True):
            st.write(f"Última grabación: `{ultimo.name}`")
            col_a, col_b = st.columns([3, 1])
            col_a.audio(str(ultimo))
            if col_b.button("🗑️ Borrar y repetir"):
                ultimo.unlink()
                ss.ultimo_audio = None
                st.rerun()

# --- Subir archivos grabados con el celular ------------------------------------------
with st.expander("📤 Subir audios ya grabados (por ejemplo, desde el celular)"):
    st.caption("Formatos: WAV, MP3, OGG, FLAC. Se convierten a WAV 16 kHz y se guardan "
               f"para {comun.bonito(nombre)}, sesión {sesion}.")
    subidos = st.file_uploader("Archivos", type=["wav", "mp3", "ogg", "flac"],
                               accept_multiple_files=True, key=f"subir_{ss.contador_grab}")
    if subidos and st.button(f"Guardar {len(subidos)} archivo(s)"):
        guardados, rechazados = 0, []
        for f in subidos:
            try:
                y = A.cargar_audio(io.BytesIO(f.getvalue()))
            except Exception:
                rechazados.append(f"{f.name}: no se pudo leer")
                continue
            ok, mensaje = A.evaluar_calidad(y)
            if ok:
                comun.guardar_audio(y, carpeta, nombre, sesion)
                guardados += 1
            else:
                rechazados.append(f"{f.name}: {mensaje}")
        st.success(f"Se guardaron {guardados} audios.")
        for r in rechazados:
            st.warning(r)
        ss.contador_grab += 1
