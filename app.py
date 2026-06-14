import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import streamlit as st
from ui.style_prod import inject_custom_css
from db.repository import cargar_ultimo_modelo

# Configuramos la app para que use la carpeta pages_prod
st.set_page_config(
    page_title="BCP Simulador Estratégico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Establecer contexto de productor para gráficos y CSS
st.session_state.es_economista = False

inject_custom_css()

# Cargar automáticamente el último modelo desplegado desde la base de datos
estado, reglas = cargar_ultimo_modelo()

if estado is None or reglas is None:
    st.title("BCP Simulador Estratégico")
    st.error("No se encontró ningún modelo desplegado en la base de datos.")
    st.info("Por favor, contacte al departamento de Economía para que desplieguen un modelo desde BCP Studio.")
    st.stop()

# Guardar en memoria de sesión de solo lectura
if 'estado_base' not in st.session_state:
    st.session_state.estado_base = estado
if 'reglas_activas' not in st.session_state:
    st.session_state.reglas_activas = reglas

# Definir navegación programática
page1 = st.Page("pages_prod/01_Simulador.py", title="1. Simulador de Escenarios")
page2 = st.Page("pages_prod/02_Comparacion.py", title="2. Comparar Estrategias")
page3 = st.Page("pages_prod/03_Reporte.py", title="3. Reporte Ejecutivo")

pg = st.navigation([page1, page2, page3])
pg.run()
