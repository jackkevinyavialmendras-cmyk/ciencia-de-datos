import pandas as pd
import streamlit as st

import comun
from src import config as C

st.title("📊 Resultados")
st.caption("Todas las gráficas se guardan también en la carpeta `reports/` para el informe.")

km = comun.metricas().get("K-MEANS")
if not km:
    st.info("Aún no hay resultados. Ejecuta el clustering primero.")
    st.page_link("paginas/entrenar.py", label="Ir a agrupar voces →", icon="🧠")
    st.stop()


def imagen(nombre, titulo, nota=None):
    ruta = C.REPORTS / nombre
    if ruta.exists():
        st.markdown(f"**{titulo}**")
        st.image(str(ruta), width="stretch")
        if nota:
            st.caption(nota)


# --- Resumen ------------------------------------------------------------------------
st.subheader("K-Means (modelo principal)")
c = st.columns(5)
c[0].metric("Accuracy en test", f"{km['accuracy_test']:.1%}",
            help="Voces de prueba asignadas al cluster del estudiante correcto")
c[1].metric("Pureza", km["pureza"], help="Qué tan «de una sola persona» es cada cluster (1 = perfecto)")
c[2].metric("Silhouette", km["silhouette"], help="Separación entre clusters: de -1 a 1, mayor es mejor")
c[3].metric("ARI", km["ARI"], help="Coincidencia con los grupos reales: 0 = al azar, 1 = perfecto")
c[4].metric("NMI", km["NMI"], help="Información mutua normalizada: 0 a 1")

csv = C.REPORTS / "comparacion_no_supervisado.csv"
if csv.exists():
    st.subheader("Comparación de algoritmos")
    st.caption("Mismos datos y mismo número de grupos. Las etiquetas reales solo se usan aquí "
               "para evaluar, nunca para agrupar.")
    st.dataframe(pd.read_csv(csv), hide_index=True, width="stretch")

# --- Detalle ------------------------------------------------------------------------
tab_km, tab_jer, tab_db = st.tabs(["K-Means", "Jerárquico", "DBSCAN"])
with tab_km:
    st.caption(f"k usado = {km['k']} · k sugerido por silhouette = {km['k_sugerido_silhouette']}")
    imagen("kmeans_codo_silhouette.png", "Elección de k: método del codo y silhouette",
           "El «codo» y el máximo de silhouette deberían estar cerca del número de estudiantes.")
    imagen("clusters_pca.png", "Grupos en 2D (PCA)",
           "Arriba a la izquierda, los grupos reales; los demás, lo que encontró cada algoritmo sin etiquetas.")
with tab_jer:
    st.caption(f"Mejor enlace según ARI: **{km.get('mejor_jerarquico', '—')}**")
    imagen("jerarquico_dendrograma.png", "Dendrogramas con las 4 distancias entre clusters vistas en clase",
           "La altura de la unión indica qué tan distintas son las voces. MIN tiende a formar "
           "cadenas; MAX y Promedio suelen dar grupos más compactos.")
with tab_db:
    st.caption(f"eps = {km['dbscan_eps']} · {km['dbscan_clusters']} grupos · "
               f"{km['dbscan_ruido']:.1%} de puntos marcados como ruido")
    imagen("dbscan_k_distancia.png", "Gráfico k-distancia para elegir eps",
           "eps se toma donde la curva empieza a subir rápido: los puntos por encima son ruido.")
