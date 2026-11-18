import os
import subprocess
import sys
import time

import streamlit as st

import comun
from src import config as C

st.title("🧠 Agrupar voces (clustering)")

inv = comun.inventario()
validos = int((inv["total"] >= C.MIN_AUDIOS).sum()) if not inv.empty else 0
if validos < 2:
    st.warning(f"Se necesitan al menos 2 estudiantes con {C.MIN_AUDIOS}+ audios. "
               f"Ahora hay {validos}.")
    st.page_link("paginas/grabar.py", label="Ir a grabar voces →", icon="🎙️")
    st.stop()

st.write(f"Dataset: **{validos} estudiantes** · **{int(inv['total'].sum())} audios**")

with st.expander("⚙️ Opciones"):
    forzar_k = st.checkbox("Elegir k manualmente",
                           help=f"Por defecto k = número de estudiantes ({validos})")
    k = st.number_input("k (número de grupos)", 2, 60, validos, disabled=not forzar_k)


def ejecutar(titulo, script, *args):
    """Ejecuta un script del proyecto mostrando su salida en vivo."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", str(C.RAIZ / "scripts" / script), *map(str, args)]
    inicio = time.time()
    with st.status(f"{titulo}...", expanded=True) as estado:
        caja = st.empty()
        lineas = []
        proc = subprocess.Popen(cmd, cwd=C.RAIZ, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        for linea in proc.stdout:
            lineas.append(linea.rstrip())
            caja.code("\n".join(lineas[-30:]), language=None)
        proc.wait()
        seg = time.time() - inicio
        if proc.returncode == 0:
            estado.update(label=f"✅ {titulo} ({seg:.0f} s)", state="complete", expanded=False)
        else:
            estado.update(label=f"❌ {titulo} falló", state="error", expanded=True)
    return proc.returncode == 0


PASOS = {   # clave: (botón, título completo, script)
    "pre": ("1 · Preprocesar", "1. Preprocesar dataset (limpieza y MFCC)", "02_preprocesar.py"),
    "clu": ("2 · Clustering", "2. Clustering: K-Means, jerárquico y DBSCAN", "03_clustering.py"),
}


def correr(claves):
    for clave in claves:
        _, titulo, script = PASOS[clave]
        args = ["--k", k] if (clave == "clu" and forzar_k) else []
        if not ejecutar(titulo, script, *args):
            st.error("Se detuvo el proceso. Revisa el mensaje de error de arriba.")
            return
    comun.predictor.clear()   # que la asistencia use el modelo recién entrenado
    st.success("🎉 Listo. Mira los resultados en la página «Resultados».")
    st.page_link("paginas/resultados.py", label="Ver resultados →", icon="📊")


st.subheader("Todo de una vez")
st.caption("Preprocesa los audios y ejecuta K-Means, clustering jerárquico y DBSCAN. "
           "Suele tardar menos de un minuto.")
if st.button("▶️ Ejecutar todo", type="primary", width="stretch"):
    correr(["pre", "clu"])

st.subheader("Paso por paso")
cols = st.columns(len(PASOS))
elegido = None
for col, (clave, (boton, titulo, _)) in zip(cols, PASOS.items()):
    if col.button(boton, help=titulo, width="stretch"):
        elegido = clave
if elegido:
    if elegido == "clu" and not (C.DATA_PROCESSED / "dataset.npz").exists():
        st.error("Primero ejecuta «1 · Preprocesar».")
    else:
        correr([elegido])
