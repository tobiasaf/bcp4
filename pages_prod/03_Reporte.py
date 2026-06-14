import streamlit as st
import pandas as pd
from ui.style_prod import inject_custom_css

inject_custom_css()

st.title("3. Reporte Final y Exportación")

st.markdown("Genera un informe con los resultados de los análisis y las simulaciones para compartir con los stakeholders (Bolsa de Cereales, Productores, Exportadores).")

# Esto es un mockup para el MVP
titulo = st.text_input("Título del Informe", value="Proyección de Escenarios Agrícolas y Logísticos - Campaña 2026/27")
autor = st.text_input("Autor / Equipo", value="Equipo Analistas BCPsim")

st.markdown("### Resumen Ejecutivo")
resumen = st.text_area("Contexto de la recomendación", value="Ante el inicio de una fase Niña fuerte, simulamos el impacto de la disminución de precipitaciones en el precio del trigo en el puerto de Bahía Blanca y sus consecuencias logísticas.")

st.markdown("### Conclusiones Principales")
st.markdown("""
- **Alerta Logística:** Se detectó que el aumento del precio concentra las ventas en un período más corto, aumentando la probabilidad de congestión en los accesos al puerto.
- **Riesgo Precio:** El modelo estima que los precios locales se despegarán (basis positivo) respecto a Chicago hacia el último trimestre debido a la merma de oferta regional.
""")

if st.button("Generar PDF", type="primary"):
  with st.spinner("Compilando reporte..."):
    # Mock de generación de PDF
    import time
    time.sleep(2)
    st.success(" Informe generado exitosamente.")
    st.download_button(
      label="Descargar Reporte PDF",
      data=b"Mock PDF Content",
      file_name="Reporte_BCPsim.pdf",
      mime="application/pdf"
    )
    
st.info(" En una versión futura, este módulo se integra con un LLM (como Gemini) para redactar párrafos explicativos a partir de los datos crudos del simulador.")
