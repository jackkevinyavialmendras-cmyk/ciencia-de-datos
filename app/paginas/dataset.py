import shutil

import streamlit as st

import comun
from src import config as C

st.title("📁 Revisar dataset")

inv = comun.inventario()
if inv.empty:
    st.info("Todavía no hay voces grabadas.")
    st.page_link("paginas/grabar.py", label="Ir a grabar voces →", icon="🎙️")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", len(inv))
c2.metric("Audios en total", int(inv["total"].sum()))
c3.metric("Listos para agrupar", int((inv["total"] >= C.MIN_AUDIOS).sum()))
c4.metric("Con sesión 2", int((inv["sesión 2"] > 0).sum()))

st.dataframe(inv, hide_index=True, width="stretch")

# --- Avisos ----------------------------------------------------------------------
avisos = []
validos = inv[inv["total"] >= C.MIN_AUDIOS]
if len(validos) < 2:
    avisos.append(f"Se necesitan al menos 2 estudiantes con {C.MIN_AUDIOS}+ audios para el clustering.")
if len(validos) and validos["total"].max() > 2 * validos["total"].min():
    avisos.append("Hay desequilibrio: algunos estudiantes tienen más del doble de audios que otros.")
if len(validos) and 0 < (validos["sesión 2"] > 0).sum() < len(validos):
    avisos.append("No todos tienen sesión 2: la prueba se hará con una división aleatoria "
                  "(la sesión 2 solo se usa como prueba si **todos** la tienen).")
for a in avisos:
    st.warning(a)
if not avisos:
    st.success("El dataset está en buen estado para el clustering.")

# --- Escuchar / borrar --------------------------------------------------------------
st.subheader("Escuchar o borrar audios")
persona = st.selectbox("Estudiante", inv["estudiante"], format_func=comun.bonito)
archivos = comun.audios_de(C.DATA_RAW / persona)
if archivos:
    archivo = st.selectbox("Audio", archivos, format_func=lambda p: p.name)
    col_a, col_b = st.columns([3, 1])
    col_a.audio(str(archivo))
    if col_b.button("🗑️ Borrar este audio"):
        archivo.unlink()
        st.rerun()

with st.expander(f"⚠️ Borrar a {comun.bonito(persona)} por completo"):
    if st.checkbox(f"Sí, quiero borrar los {len(archivos)} audios de {comun.bonito(persona)}"):
        if st.button("Borrar definitivamente", type="primary"):
            shutil.rmtree(C.DATA_RAW / persona)
            st.rerun()

# --- Voces desconocidas -------------------------------------------------------------------
if C.DATA_DESCONOCIDOS.exists():
    desc = [p for p in C.DATA_DESCONOCIDOS.iterdir() if p.is_dir()]
    if desc:
        st.subheader("Voces desconocidas (no se usan en el clustering)")
        st.write(", ".join(f"{comun.bonito(p.name)} ({len(comun.audios_de(p))})" for p in desc))
