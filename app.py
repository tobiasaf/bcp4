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

st.title("BCPsim: Simulador Estratégico y Backtesting")
st.markdown('<p class="subtitle">Prueba de concepto del motor predictivo con contraste de escenarios reales vs. simulados para la Bolsa de Cereales y Productos de Bahía Blanca</p>', unsafe_allow_html=True)

# 1. Cargar el modelo persistido en SQLite
estado, reglas = cargar_ultimo_modelo()

if estado is None or reglas is None:
    st.error("No se encontró el modelo pre-entrenado en la base de datos local.")
    st.stop()

# Cargar datos históricos reales para comparación y condiciones iniciales
csv_path = "data/historico_trigo_real.csv"
df_real = None
campanas_disponibles = []

if os.path.exists(csv_path):
    df_real = pd.read_csv(csv_path)
    df_real['fecha'] = pd.to_datetime(df_real['fecha'])
    
    # Calcular campañas trigueras (Junio a Mayo)
    def obtener_campana(fecha):
        yr = fecha.year
        if fecha.month >= 6:
            return f"{yr}/{str(yr+1)[2:]}"
        else:
            return f"{yr-1}/{str(yr)[2:]}"
            
    df_real['campaña'] = df_real['fecha'].apply(obtener_campana)
    campanas_disponibles = sorted(df_real['campaña'].unique())
    # Filtrar campañas desde 2021/22 en adelante para tener datos limpios
    campanas_disponibles = [c for c in campanas_disponibles if c >= "2021/22"]

# 2. Configurar Campaña y Fecha de Partida
st.subheader("1. Selección del Punto de Partida de la Campaña")
col_c1, col_c2 = st.columns(2)

campana_seleccionada = None
fecha_inicio = None

with col_c1:
    options_camp = campanas_disponibles + ["Proyección Campaña 2026/27 (En Vivo)"]
    campana_seleccionada = st.selectbox(
        "Campaña Triguera a simular:",
        options=options_camp,
        index=options_camp.index("2024/25") if "2024/25" in options_camp else 0,
        help="Permite elegir una campaña histórica para contrastar predicción vs. realidad, o la campaña futura en vivo."
    )

with col_c2:
    if campana_seleccionada == "Proyección Campaña 2026/27 (En Vivo)":
        fecha_inicio = pd.to_datetime("2026-06-01").date()
        st.caption("📅 Proyectando desde el 01-Jun-2026 en adelante.")
    elif df_real is not None:
        df_camp = df_real[df_real['campaña'] == campana_seleccionada].sort_values('fecha')
        fechas_list = df_camp['fecha'].dt.date.tolist()
        
        fecha_inicio = st.selectbox(
            "Punto de partida (Semana de inicio):",
            options=fechas_list,
            index=0,
            format_func=lambda d: f"{d.strftime('%d-%b-%Y')} ({'Siembra' if d.month in [6,7,8] else 'Espigazón' if d.month in [9,10] else 'Cosecha' if d.month in [11,12,1] else 'Comercialización'})",
            help="Selecciona la fecha histórica exacta desde la cual comenzará la simulación."
        )

# Cargar variables iniciales en base a la fecha de partida elegida
if df_real is not None and campana_seleccionada != "Proyección Campaña 2026/27 (En Vivo)":
    row_real = df_real[df_real['fecha'] == pd.to_datetime(fecha_inicio)].iloc[0]
    val_fas_usd = float(row_real.get('precio_fas_usd', 175.09))
    val_precio_futuro = float(row_real.get('rofex_precio_usd', 215.0) if 'rofex_precio_usd' in row_real and not pd.isna(row_real['rofex_precio_usd']) else 215.0)
    val_lluvia = float(row_real.get('lluvia_mm', 10.0))
    val_temp = float(row_real.get('temp_media', 18.0))
    val_tc = float(row_real.get('tipo_cambio', 900.0))
    val_brecha = float(row_real.get('brecha_cambiaria_pct', 35.0))
    
    val_rinde_init = float(row_real.get('rendimiento_estimado_tn_ha', 2.80))
    val_camiones_init = float(row_real.get('descargas_camiones', 675.0))
else:
    val_fas_usd = float(estado.get('precio_fas_usd', 175.09))
    val_precio_futuro = 215.0
    val_lluvia = float(estado.get('lluvia_mm', 10.0))
    val_temp = float(estado.get('temp_media', 18.0))
    val_tc = float(estado.get('tipo_cambio', 900.0))
    val_brecha = float(estado.get('brecha_cambiaria_pct', 35.0))
    val_rinde_init = 2.80
    val_camiones_init = 675.0

# 3. Sliders de Variables de Entrada "Top" en la misma pantalla (organizados en 3 columnas)
st.markdown("---")
st.subheader("2. Ajustar Variables Iniciales (Pre-cargadas de la Fecha de Inicio)")
st.info("💡 **Simulación Endógena Activa:** Las variables de **Rendimiento (tn/ha)** y **Camiones Diarios** se inicializan con el valor real de la fecha de partida, pero luego el motor las calcula dinámicamente día a día en base a las lluvias y la temperatura.")

col_var1, col_var2, col_var3 = st.columns(3)

with col_var1:
    st.markdown("**💰 Precios en Dólares (USD/tn)**")
    user_precio_actual = st.slider("Precio FAS Actual", 100.0, 400.0, val_fas_usd, step=5.0, help="Precio físico spot del trigo en el mercado local")
    user_precio_futuro = st.slider("Precio Futuros fin de campaña", 100.0, 400.0, val_precio_futuro, step=5.0, help="Precio pactado para entrega al final de la cosecha (MatbaRofex)")

with col_var2:
    st.markdown("**🌾 Agronomía y Clima (Exógenas)**")
    user_lluvia = st.slider("Lluvia Semanal (mm)", 0.0, 150.0, val_lluvia, step=1.0)
    user_temp = st.slider("Temp. Media (°C)", 5.0, 40.0, val_temp, step=1.0)

with col_var3:
    st.markdown("**🚛 Macroeconómico y Cambiario**")
    user_tc = st.slider("Dólar Oficial (ARS)", 500.0, 1500.0, val_tc, step=10.0)
    user_brecha = st.slider("Brecha Cambiaria (%)", 0.0, 150.0, val_brecha, step=5.0)

# 4. Configurar Escenarios en el centro de la pantalla
st.markdown("---")
st.subheader("3. Selección del Shock / Crisis y Duración")
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

# Inicializar eventos y exógenas
eventos_manuales = []
datos_futuros = {}

# Clonar el estado base e inyectar las variables
estado_inicial = estado.copy()
if df_real is not None and campana_seleccionada != "Proyección Campaña 2026/27 (En Vivo)":
    # Copiar variables históricas de la fila real para coherencia de todo el vector de 54 variables
    row_real = df_real[df_real['fecha'] == pd.to_datetime(fecha_inicio)].iloc[0]
    for k, v in row_real.items():
        if k != 'fecha' and k != 'campaña' and not pd.isna(v):
            estado_inicial[k] = v

estado_inicial['precio_fas_usd'] = user_precio_actual
estado_inicial['precio_futuro_usd'] = user_precio_futuro
estado_inicial['precio_fob_usd'] = user_precio_actual + 34.4
estado_inicial['lluvia_mm'] = user_lluvia
estado_inicial['temp_media'] = user_temp
estado_inicial['tipo_cambio'] = user_tc
estado_inicial['brecha_cambiaria_pct'] = user_brecha
estado_inicial['rendimiento_estimado_tn_ha'] = val_rinde_init
estado_inicial['descargas_camiones'] = val_camiones_init

# Configurar parámetros específicos del shock elegido para variables exógenas
if escenario == "Paro de Camioneros (Logística)":
    st.info("ℹ️ **Paro de Camioneros**: Reduce el ingreso diario de camiones al puerto en un 80% durante el conflicto (Días 15 al 22 de la simulación).")

elif escenario == "Bajante del Río Paraná (Desvío River)":
    st.info("ℹ️ **Bajante del Río Paraná**: Desvía carga hacia Bahía Blanca aumentando los camiones en un 25% y presionando al alza el FAS local.")
    datos_futuros['nivel_parana_m'] = [0.5] * dias_simular

elif escenario == "Sequía Severa (La Niña)":
    st.info("ℹ️ **Sequía Severa**: Reduce lluvias en un 60% e impacta progresivamente en el rendimiento y sube los futuros.")
    datos_futuros['lluvia_mm'] = [max(0.0, user_lluvia * 0.4)] * dias_simular

st.markdown("---")

# 5. Botón para correr la simulación y graficar
if st.button("Ejecutar Simulación con tus Parámetros", type="primary", use_container_width=True):
    motor = MotorSimulacion(estado_inicial, reglas)
    
    with st.spinner("Propagando shock día a día..."):
        historial = motor.correr(dias=dias_simular, escenario=escenario, eventos_manuales=eventos_manuales, datos_futuros_conocidos=datos_futuros)
        
        # Procesar resultados
        df_sim = pd.DataFrame([s.valores for s in historial])
        df_sim['dia'] = df_sim.index
        
        # Asignar fechas al eje X de la simulación
        fechas_proyeccion = [pd.to_datetime(fecha_inicio) + pd.Timedelta(days=t) for t in range(dias_simular)]
        df_sim['fecha'] = fechas_proyeccion
        
        # Obtener el subconjunto real correspondiente si la campaña es histórica
        df_real_subset = pd.DataFrame()
        if df_real is not None and campana_seleccionada != "Proyección Campaña 2026/27 (En Vivo)":
            df_real_subset = df_real[
                (df_real['fecha'] >= pd.to_datetime(fecha_inicio)) & 
                (df_real['fecha'] <= pd.to_datetime(fecha_inicio) + pd.Timedelta(days=dias_simular))
            ].copy()
            
        st.subheader("4. Resultados Proyectados y Contraste Histórico")
        if not df_real_subset.empty:
            st.write("📈 **Líneas Sólidas:** Proyección Simulada | 🏁 **Líneas Punteadas con Marcadores:** Datos Reales Históricos")
        else:
            st.write("Proyecciones simuladas a futuro (Modo En Vivo).")
            
        col_res1, col_res2 = st.columns(2)
        
        # Gráfica de Precios en Dólares (con Plotly)
        with col_res1:
            import plotly.graph_objects as go
            fig_precios = go.Figure()
            
            # FAS Spot
            fig_precios.add_trace(go.Scatter(
                x=df_sim['fecha'], y=df_sim['precio_fas_usd'],
                mode='lines', name='FAS Spot (Simulado)',
                line=dict(color='#2962FF', width=2.5)
            ))
            if not df_real_subset.empty and 'precio_fas_usd' in df_real_subset.columns:
                fig_precios.add_trace(go.Scatter(
                    x=df_real_subset['fecha'], y=df_real_subset['precio_fas_usd'],
                    mode='lines+markers', name='FAS Spot (Real)',
                    line=dict(color='#2962FF', width=1.5, dash='dash')
                ))
                
            # Futuros
            fig_precios.add_trace(go.Scatter(
                x=df_sim['fecha'], y=df_sim['precio_futuro_usd'],
                mode='lines', name='Futuro (Simulado)',
                line=dict(color='#00E676', width=2.5)
            ))
            if not df_real_subset.empty and 'rofex_precio_usd' in df_real_subset.columns:
                fig_precios.add_trace(go.Scatter(
                    x=df_real_subset['fecha'], y=df_real_subset['rofex_precio_usd'],
                    mode='lines+markers', name='Futuro (Real)',
                    line=dict(color='#00E676', width=1.5, dash='dash')
                ))
                
            # Chicago Ref
            fig_precios.add_trace(go.Scatter(
                x=df_sim['fecha'], y=df_sim['precio_chicago_usd'],
                mode='lines', name='Chicago (Simulado)',
                line=dict(color='#FF3D00', width=2.0)
            ))
            if not df_real_subset.empty and 'precio_chicago_usd' in df_real_subset.columns:
                fig_precios.add_trace(go.Scatter(
                    x=df_real_subset['fecha'], y=df_real_subset['precio_chicago_usd'],
                    mode='lines+markers', name='Chicago (Real)',
                    line=dict(color='#FF3D00', width=1.2, dash='dash')
                ))
                
            fig_precios.update_layout(
                title="💰 Evolución de Precios (USD/tn)",
                xaxis_title="Fecha",
                yaxis_title="USD/tn",
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#2E2E2E" if st.session_state.get('es_economista', False) else "#E0E0E0"),
                yaxis=dict(showgrid=True, gridcolor="#2E2E2E" if st.session_state.get('es_economista', False) else "#E0E0E0"),
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_precios, use_container_width=True)
            
        # Gráfica de Rendimiento Estimado (tn/ha) y Lluvia (con Plotly)
        with col_res2:
            fig_agro = go.Figure()
            
            # Rendimiento
            fig_agro.add_trace(go.Scatter(
                x=df_sim['fecha'], y=df_sim['rendimiento_estimado_tn_ha'],
                mode='lines', name='Rendimiento (Simulado)',
                line=dict(color='#F9A826', width=2.5)
            ))
            if not df_real_subset.empty and 'rendimiento_estimado_tn_ha' in df_real_subset.columns:
                fig_agro.add_trace(go.Scatter(
                    x=df_real_subset['fecha'], y=df_real_subset['rendimiento_estimado_tn_ha'],
                    mode='lines+markers', name='Rendimiento (Real)',
                    line=dict(color='#F9A826', width=1.5, dash='dash')
                ))
                
            # Lluvia
            fig_agro.add_trace(go.Scatter(
                x=df_sim['fecha'], y=df_sim['lluvia_mm'],
                mode='lines', name='Lluvia (Simulada)',
                line=dict(color='#0288D1', width=2.0)
            ))
            if not df_real_subset.empty and 'lluvia_mm' in df_real_subset.columns:
                fig_agro.add_trace(go.Scatter(
                    x=df_real_subset['fecha'], y=df_real_subset['lluvia_mm'],
                    mode='lines+markers', name='Lluvia (Real)',
                    line=dict(color='#0288D1', width=1.2, dash='dash')
                ))
                
            fig_agro.update_layout(
                title="🌾 Rendimiento Estimado (tn/ha) y Lluvia Semanal (mm)",
                xaxis_title="Fecha",
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#2E2E2E" if st.session_state.get('es_economista', False) else "#E0E0E0"),
                yaxis=dict(showgrid=True, gridcolor="#2E2E2E" if st.session_state.get('es_economista', False) else "#E0E0E0"),
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_agro, use_container_width=True)
            
        # Gráfica de Camiones Diarios (con Plotly)
        fig_logistica = go.Figure()
        
        # Camiones
        fig_logistica.add_trace(go.Scatter(
            x=df_sim['fecha'], y=df_sim['descargas_camiones'],
            mode='lines', name='Camiones Diarios (Simulado)',
            line=dict(color='#9C27B0', width=2.5)
        ))
        if not df_real_subset.empty and 'descargas_camiones' in df_real_subset.columns:
            fig_logistica.add_trace(go.Scatter(
                x=df_real_subset['fecha'], y=df_real_subset['descargas_camiones'],
                mode='lines+markers', name='Camiones Diarios (Real)',
                line=dict(color='#9C27B0', width=1.5, dash='dash')
            ))
            
        fig_logistica.update_layout(
            title="🚛 Flujo Logístico Diario en Puerto (Camiones/Día)",
            xaxis_title="Fecha",
            yaxis_title="Camiones/Día",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#2E2E2E" if st.session_state.get('es_economista', False) else "#E0E0E0"),
            yaxis=dict(showgrid=True, gridcolor="#2E2E2E" if st.session_state.get('es_economista', False) else "#E0E0E0"),
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_logistica, use_container_width=True)
            
        # 6. Mostrar Log de Reglas del Motor
        st.markdown("---")
        st.subheader("5. Bitácora del Motor (Reglas Lógicas Gatilladas)")
        
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
