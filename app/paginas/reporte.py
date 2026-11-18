import io

import streamlit as st

import comun

st.title("📋 Reporte de asistencia")

df = comun.leer_asistencia()
if df.empty:
    st.info("Todavía no hay registros de asistencia.")
    st.stop()

fecha = st.selectbox("Fecha", sorted(df["fecha"].unique(), reverse=True))
dia = df[df["fecha"] == fecha].copy()
dia["estudiante"] = dia["estudiante"].map(comun.bonito)
dia["confianza"] = (dia["confianza"] * 100).round(1).astype(str) + " %"

registrados = len([p for p in comun.inventario()["estudiante"]])
c1, c2 = st.columns(2)
c1.metric("Presentes", len(dia))
c2.metric("Ausentes", max(registrados - len(dia), 0))

st.dataframe(dia.drop(columns="id"), hide_index=True, width="stretch")

excel = io.BytesIO()
dia.drop(columns="id").to_excel(excel, index=False)
st.download_button("⬇️ Descargar Excel", excel.getvalue(),
                   file_name=f"asistencia_{fecha}.xlsx", type="primary")

with st.expander("Corregir un registro equivocado"):
    fila = st.selectbox("Registro", dia.to_dict("records"),
                        format_func=lambda r: f"{r['hora']} · {r['estudiante']} · {r['modelo']}")
    if st.button("🗑️ Borrar registro"):
        comun.borrar_asistencia(fila["id"])
        st.rerun()
