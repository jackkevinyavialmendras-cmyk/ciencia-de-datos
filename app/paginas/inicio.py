import streamlit as st

import comun
from src import config as C
from src.predictor import Predictor

st.title("🎙️ Asistencia de estudiantes por reconocimiento de voz")
st.caption("Práctica del segundo parcial · Machine Learning · Área: Reconocimiento de voz · "
           "Enfoque: aprendizaje no supervisado")

st.markdown("""
Cada voz se convierte en una «huella» de 80 números (MFCC) y los algoritmos
**agrupan las voces por parecido sin saber de quién son**. Cada grupo debería
corresponder a un estudiante; al tomar asistencia, la voz nueva se asigna al grupo
más cercano o se rechaza si no se parece a ninguno.

| Algoritmo | Tipo (diapositivas) | Para qué |
|---|---|---|
| **K-Means** | Agrupamiento por particiones | Modelo principal: forma k grupos con centroides |
| **Jerárquico aglomerativo** | Clustering jerárquico | Dendrograma: qué voces se parecen; compara MIN, MAX, Promedio y Centroides |
| **DBSCAN** | Basado en densidad | Detecta ruido / voces que no encajan en ningún grupo |
""")

st.subheader("Estado del proyecto")
inv = comun.inventario()
listos = int((inv["total"] >= C.MIN_AUDIOS).sum()) if not inv.empty else 0
npz = C.DATA_PROCESSED / "dataset.npz"
hay_modelo = Predictor.disponible()
desactualizado = npz.exists() and comun.ultimo_audio_mtime() > npz.stat().st_mtime

pasos = [
    ("1. Grabar voces", listos >= 2,
     f"{listos} estudiantes con {C.MIN_AUDIOS}+ audios ({inv['total'].sum() if not inv.empty else 0} audios en total)"
     if listos else "Aún no hay voces grabadas", "paginas/grabar.py"),
    ("2. Preprocesar", npz.exists() and not desactualizado,
     "Hay audios nuevos: vuelve a preprocesar y agrupar" if desactualizado
     else ("Dataset preparado" if npz.exists() else "Pendiente"), "paginas/entrenar.py"),
    ("3. Agrupar voces (clustering)", hay_modelo,
     "K-Means, jerárquico y DBSCAN ejecutados" if hay_modelo else "Pendiente", "paginas/entrenar.py"),
    ("4. Tomar asistencia", hay_modelo,
     "Listo para usar" if hay_modelo else "Necesita el clustering", "paginas/asistencia.py"),
]
for titulo, ok, detalle, pagina in pasos:
    c1, c2, c3 = st.columns([0.05, 0.6, 0.35])
    c1.markdown("✅" if ok else "⬜")
    c2.markdown(f"**{titulo}** — {detalle}")
    c3.page_link(pagina, label="Ir →")
