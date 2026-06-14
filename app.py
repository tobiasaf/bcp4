import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import streamlit as st
import pandas as pd
import numpy as np
from db.repository import cargar_ultimo_modelo
from engine.simulation import MotorSimulacion
from engine.models import EventoProgramado

# Configuración de página limpia y centrada
st.set_page_config(
    page_title="BCPsim - Demo Motor",
    layout="centered"
)

# Estilización premium sencilla
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    h1 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #F8F9FA;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9EADB6;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #0056B3 !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
    }
    .stSlider > label {
        color: #F8F9FA !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("BCPsim: Simulador Estratégico (PoC)")
st.markdown('<p class="subtitle">Prueba de concepto del motor predictivo y propagación de shocks en el mercado de trigo y logística portuaria (Bahía Blanca)</p>', unsafe_allow_html=True)

# 1. Cargar el modelo persistido en SQLite
estado, reglas = cargar_ultimo_modelo()

if estado is None or reglas is None:
    st.error("No se encontró el modelo pre-entrenado en la base de datos local.")
    st.stop()

# 2. Configurar Escenarios en el centro de la pantalla
st.subheader("1. Selección del Shock / Crisis")
col1, col2 = st.columns(2)

with col1:
    escenario = st.selectbox(
        "Escenario a simular:",
        options=[
            "Ninguno (Mercado Neutral)",
            "Paro de Camioneros (Logística)",
            "Bajante del Paraná (Desvío River)",
            "Sequía Severa (La Niña)"
        ]
    )

with col2:
    dias_simular = st.slider("Días de Proyección:", min_value=30, max_value=180, value=120, step=10)

st.markdown("---")

# 3. Sliders de Variables de Entrada "Top" en la misma pantalla (organizados en 3 columnas)
st.subheader("2. Ajustar Variables Iniciales (Simulador)")
col_var1, col_var2, col_var3 = st.columns(3)

# Valores por defecto cargados desde la base de datos (o fallbacks seguros si no existen)
val_chicago = float(estado.get('precio_chicago_usd', 188.10))
val_fob = float(estado.get('precio_fob_usd', 209.50))
val_fas_usd = float(estado.get('precio_fas_usd', 175.09))
val_rinde = float(estado.get('rendimiento_estimado_tn_ha', 2.30))
val_lluvia = float(estado.get('lluvia_mm', 10.0))
val_camiones = int(estado.get('descargas_camiones', 800))
val_tc = float(estado.get('tipo_cambio', 900.0))
val_brecha = float(estado.get('brecha_cambiaria_pct', 35.0))
val_temp = float(estado.get('temp_media', 18.0))

with col_var1:
    st.markdown("**💰 Precios en Dólares**")
    user_chicago = st.slider("Trigo Chicago (USD/tn)", 100.0, 400.0, val_chicago, step=5.0)
    user_fob = st.slider("Trigo FOB Oficial (USD/tn)", 100.0, 400.0, val_fob, step=5.0)
    user_fas_usd = st.slider("Trigo FAS Local (USD/tn)", 50.0, 300.0, val_fas_usd, step=5.0)

with col_var2:
    st.markdown("**🌾 Agronomía y Clima**")
    user_rinde = st.slider("Rendimiento (tn/ha)", 0.5, 5.0, val_rinde, step=0.1)
    user_lluvia = st.slider("Lluvia Semanal (mm)", 0.0, 150.0, val_lluvia, step=1.0)
    user_temp = st.slider("Temp. Media (°C)", 5.0, 40.0, val_temp, step=1.0)

with col_var3:
    st.markdown("**🚛 Logística y Tipo de Cambio**")
    user_camiones = st.slider("Camiones Diarios", 100, 2000, val_camiones, step=50)
    user_tc = st.slider("Dólar Oficial (ARS)", 500.0, 1500.0, val_tc, step=10.0)
    user_brecha = st.slider("Brecha Cambiaria (%)", 0.0, 150.0, val_brecha, step=5.0)

# Inicializar eventos y exógenas
eventos_manuales = []
datos_futuros = {}

# Clonar el estado base e inyectar las variables que el usuario modificó mediante los sliders
estado_inicial = estado.copy()
estado_inicial['precio_chicago_usd'] = user_chicago
estado_inicial['precio_fob_usd'] = user_fob
estado_inicial['precio_fas_usd'] = user_fas_usd
estado_inicial['rendimiento_estimado_tn_ha'] = user_rinde
estado_inicial['lluvia_mm'] = user_lluvia
estado_inicial['temp_media'] = user_temp
estado_inicial['descargas_camiones'] = user_camiones
estado_inicial['tipo_cambio'] = user_tc
estado_inicial['brecha_cambiaria_pct'] = user_brecha

# Configurar parámetros específicos del shock elegido
if escenario == "Paro de Camioneros (Logística)":
    st.info("ℹ️ **Paro de Camioneros**: Reduce el ingreso diario de camiones al puerto en un 80% durante el conflicto.")
    duracion = st.slider("Duración del Conflicto (días):", 3, 15, 7)
    dia_inicio = 15
    impacto_camiones = -float(estado_inicial.get('descargas_camiones', 1000) * 0.8)
    eventos_manuales.append(EventoProgramado(dia_ejecucion=dia_inicio, variable='descargas_camiones', impacto=impacto_camiones, origen="Inicio de Paro"))
    eventos_manuales.append(EventoProgramado(dia_ejecucion=dia_inicio + duracion, variable='descargas_camiones', impacto=-impacto_camiones, origen="Fin de Paro"))

elif escenario == "Bajante del Río Paraná (Desvío River)":
    st.info("ℹ️ **Bajante del Río Paraná**: Aumenta las descargas de camiones en Bahía Blanca en un 25% y el precio de pizarra local en un 5% por la demanda física inmediata de completamiento.")
    datos_futuros['nivel_parana_m'] = [0.5] * dias_simular

elif escenario == "Sequía Severa (La Niña)":
    st.info("ℹ️ **Sequía Severa**: Reduce lluvias en un 40% e incrementa el precio de Chicago en un 15% como shock de oferta global.")
    datos_futuros['lluvia_mm'] = [max(0.0, float(estado_inicial.get('lluvia_mm', 10.0)) * 0.6)] * dias_simular
    precio_chicago = float(estado_inicial.get('precio_chicago_usd', 200.0))
    datos_futuros['precio_chicago_usd'] = [precio_chicago * 1.15] * dias_simular

st.markdown("---")

# 4. Botón para correr la simulación
if st.button("Ejecutar Simulación con tus Parámetros", type="primary", use_container_width=True):
    motor = MotorSimulacion(estado_inicial, reglas)
    
    with st.spinner("Propagando shock día a día..."):
        historial = motor.correr(dias=dias_simular, eventos_manuales=eventos_manuales, datos_futuros_conocidos=datos_futuros)
        
        # Procesar resultados
        df_sim = pd.DataFrame([s.valores for s in historial])
        df_sim['dia'] = df_sim.index
        
        # Graficar variables clave: Precio Pizarra, FAS y Descarga de Camiones
        st.subheader("3. Resultados Proyectados")
        
        col_res1, col_res2 = st.columns(2)
        
        # Gráfica de Precios en Dólares (las variables "más top")
        with col_res1:
            st.markdown("**📈 Evolución de Precios en Dólares (USD/tn)**")
            df_precios_usd = df_sim[['dia', 'precio_chicago_usd', 'precio_fob_usd', 'precio_fas_usd']].copy()
            df_precios_usd.columns = ['Día', 'Precio Chicago', 'Precio FOB Oficial', 'Precio FAS Local']
            st.line_chart(df_precios_usd.set_index('Día'))
            
        # Gráfica de Rendimiento Estimado (tn/ha) y Logística
        with col_res2:
            st.markdown("**🌾 Rendimiento Estimado y Lluvia acumulada**")
            df_agro = df_sim[['dia', 'rendimiento_estimado_tn_ha', 'lluvia_mm']].copy()
            df_agro.columns = ['Día', 'Rendimiento (tn/ha)', 'Lluvia (mm)']
            st.line_chart(df_agro.set_index('Día'))
            
        # Gráfica adicional de Camiones
        st.markdown("**🚛 Volumen Logístico en Puerto (Toneladas/Día)**")
        df_logistica = df_sim[['dia', 'descargas_camiones_tn']].copy()
        df_logistica.columns = ['Día', 'Descarga de Camiones (tn)']
        st.line_chart(df_logistica.set_index('Día'))
            
        # 5. Mostrar Log de Reglas del Motor
        st.markdown("---")
        st.subheader("4. Bitácora del Motor (Reglas Lógicas Gatilladas)")
        
        log_eventos = []
        for s in historial:
            if s.reglas_disparadas or s.eventos_ejecutados:
                for r in s.reglas_disparadas:
                    log_eventos.append(f"📅 **Día {s.dia}**: Se disparó la regla de comportamiento: `{r}`")
                for e in s.eventos_ejecutados:
                    log_eventos.append(f"📅 **Día {s.dia}**: Se aplicó el impacto del evento: `{e}`")
                    
        if log_eventos:
            for log in log_eventos[:20]:  # Mostrar los primeros 20 eventos
                st.write(log)
            if len(log_eventos) > 20:
                st.caption(f"...y otros {len(log_eventos) - 20} eventos lógicos menores.")
        else:
            st.info("No se gatillaron anomalías ni reglas de comportamiento fuera del estándar durante el período simulado (comportamiento inercial del mercado).")
