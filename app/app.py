"""Interfaz web del proyecto: asistencia de estudiantes por reconocimiento de voz.

Ejecutar desde la raíz del proyecto (o con doble clic en iniciar.bat):
    streamlit run app/app.py
y abrir http://localhost:8501 en Chrome.
"""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))   # para importar src.*
sys.path.insert(0, str(APP))          # para importar comun

import streamlit as st

st.set_page_config(page_title="Asistencia por voz", page_icon="🎙️", layout="wide")

paginas = {
    "Proyecto": [
        st.Page("paginas/inicio.py", title="Inicio", icon="🏠", default=True),
    ],
    "1 · Dataset": [
        st.Page("paginas/grabar.py", title="Grabar voces", icon="🎙️"),
        st.Page("paginas/dataset.py", title="Revisar dataset", icon="📁"),
    ],
    "2 · Clustering": [
        st.Page("paginas/entrenar.py", title="Agrupar voces", icon="🧠"),
        st.Page("paginas/resultados.py", title="Resultados", icon="📊"),
    ],
    "3 · Prototipo": [
        st.Page("paginas/asistencia.py", title="Tomar asistencia", icon="✅"),
        st.Page("paginas/reporte.py", title="Reporte de asistencia", icon="📋"),
    ],
}

st.navigation(paginas).run()
