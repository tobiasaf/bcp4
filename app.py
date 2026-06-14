import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import streamlit as st
import pandas as pd
import numpy as np
from db.repository import cargar_ultimo_modelo
from ml.multi_predictor import entrenar_y_predecir_todo
from ui.charts import plot_backtest_single
from ui.style_eco import inject_custom_css
import ui.charts

# Configuración de página amplia estilo Bloomberg Terminal
st.set_page_config(
    page_title="BCPsim - Terminal de Simulación y Backtesting",
    layout="wide"
)

# Forzar modo economista (Bloomberg theme) por defecto
st.session_state['es_economista'] = True

# Inyectar CSS Bloomberg Premium
inject_custom_css()

st.title("BCPsim: Terminal de Simulación y Backtesting (ML Ensemble)")
st.markdown('<p class="subtitle" style="color: #9EADB6; font-size: 1.1rem; margin-bottom: 2rem;">Simulación recursiva out-of-sample con el Ensamble de Machine Learning de 7 modelos de la BCP</p>', unsafe_allow_html=True)

# 1. Cargar datos
file_path_real = "data/historico_trigo_real.csv"
if not os.path.exists(file_path_real):
    st.error("No se encontró el archivo de datos históricos reales en data/historico_trigo_real.csv.")
    st.stop()

df_raw = pd.read_csv(file_path_real)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

# Determinar campañas
def obtener_campana(fecha):
    yr = fecha.year
    if fecha.month >= 6:
        return f"{yr}/{str(yr+1)[2:]}"
    else:
        return f"{yr-1}/{str(yr)[2:]}"

df_raw['campaña'] = df_raw['fecha'].apply(obtener_campana)
campanas_disponibles = sorted(df_raw['campaña'].unique())
campanas_a_predecir = [c for c in campanas_disponibles if c >= "2021/22"]

# Asegurar que la campaña 2026/27 esté en las opciones
if '2026/27' not in campanas_a_predecir:
    campanas_a_predecir.append('2026/27')

# 2. Configurar Campaña y Corte en dos columnas
st.subheader("1. Configuración del Backtesting / Simulación")
col_c1, col_c2 = st.columns(2)

with col_c1:
    campana_seleccionada = st.selectbox(
        "Selecciona la Campaña Triguera a simular/predecir:",
        options=campanas_a_predecir,
        index=len(campanas_a_predecir) - 2 if len(campanas_a_predecir) >= 2 else 0,
        help="El modelo se entrenará con toda la historia previa y simulará la temporada completa."
    )

# Calcular fecha de corte (1 de junio del año de inicio)
año_inicio = int(campana_seleccionada.split('/')[0])
fecha_corte = pd.to_datetime(f"{año_inicio}-06-01").date()

# Calcular semanas de avance disponibles
fecha_inicio_campana = pd.to_datetime(f"{año_inicio}-06-01")
fecha_fin_campana = pd.to_datetime(f"{año_inicio + 1}-02-01")
df_campana_test = df_raw[(df_raw['fecha'] >= fecha_inicio_campana) & (df_raw['fecha'] <= fecha_fin_campana)].sort_values('fecha')
semanas_opciones = df_campana_test['fecha'].dt.date.tolist()

if len(semanas_opciones) == 0:
    semanas_opciones = [fecha_corte]

with col_c2:
    if campana_seleccionada == '2026/27':
        fecha_proyeccion = fecha_corte
        st.write("")
        st.info("📅 **Modo En Vivo:** Proyección forward out-of-sample a 35 semanas (hasta Febrero de 2027).")
    else:
        opciones_avance = {
            "Simular 100% a ciegas desde el inicio (Junio)": semanas_opciones[0],
        }
        for sem in semanas_opciones:
            dt_sem = pd.to_datetime(sem)
            if dt_sem.month == 8 and dt_sem.day <= 7:
                opciones_avance["Mitad de Siembra (Agosto)"] = sem
            elif dt_sem.month == 10 and dt_sem.day <= 7:
                opciones_avance["Periodo Crítico / Espigazón (Octubre)"] = sem
            elif dt_sem.month == 11 and dt_sem.day <= 20 and dt_sem.day >= 10:
                opciones_avance["Cosecha Abierta / Presión Oferta (Noviembre)"] = sem

        for idx, sem in enumerate(semanas_opciones):
            opciones_avance[f"Semana {idx+1}: {sem.strftime('%d-%b-%Y')}"] = sem

        avance_seleccionado = st.selectbox(
            "Punto de partida en la campaña (Fecha de Inicio):",
            options=list(opciones_avance.keys()),
            index=0,
            help="El simulador inyectará los datos reales hasta esta fecha, y a partir de ella generará predicciones endógenas."
        )
        fecha_proyeccion = opciones_avance[avance_seleccionado]

# 3. MODO EN VIVO 2026/27 (Escenarios agronómicos/financieros)
clima_scenario = "Neutral Promedio"
chicago_scenario_val = None
devaluacion_mensual_pct = 2.0

if campana_seleccionada == '2026/27':
    st.markdown("---")
    st.subheader("🌾 Configuración de Escenario - Modo En Vivo 2026/27 (Predecir 2027)")
    st.warning("⚠️ El simulador proyectará la campaña 2026/27 out-of-sample. Ajusta los parámetros base del escenario:")
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        clima_scenario = st.selectbox(
            "Escenario Climático Agronómico:",
            options=["Neutral Promedio", "Niña Moderada (Estrés / Lluvias Bajas)", "Niño Favorable (Clima Excelente)"],
            index=0,
            help="Determina el comportamiento del NDVI satelital simulado y lluvias estimadas en primavera (período crítico)."
        )
    with col_l2:
        chicago_scenario_val = st.number_input(
            "Precio Trigo Chicago Proyectado (USD/tn):",
            min_value=100.0,
            max_value=500.0,
            value=210.0,
            step=5.0,
            help="Define la señal internacional constante durante la temporada."
        )
    with col_l3:
        devaluacion_mensual_pct = st.slider(
            "Tasa de Devaluación Oficial Mensual (%):",
            0.0, 10.0, 2.0, 0.5,
            help="Define el ritmo de deslizamiento cambiario para simular la brecha y el tipo de cambio oficial."
        )

# 4. Variables exógenas y opciones
st.markdown("---")
st.subheader("2. Parámetros del Simulador y Ensamble")
col_exo, col_sim = st.columns(2)

cols_base = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != 'fecha' and not c.startswith('anomalia')]

with col_exo:
    exogenas = st.multiselect(
        "Variables Exógenas (Futuro Conocido):",
        options=cols_base,
        default=['tipo_cambio', 'precio_chicago_usd', 'lluvia_mm', 'temp_media'],
        help="El sistema inyectará la realidad conocida de estas variables para guiar el resto de predicciones endógenas."
    )

with col_sim:
    modo_pesos = st.radio(
        "Ensamble de modelos:",
        options=["Dinámico Bayesiano (DMA - Autoadaptativo)", "Stress Testing (Ponderación Estática Manual)"],
        index=0,
        horizontal=True
    )

stress_weights = None
if modo_pesos == "Stress Testing (Ponderación Estática Manual)":
    st.info("Ajusta los pesos para cada uno de los 7 modelos de ML del simulador:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        w_vecm = st.slider("VECM (Cointegración)", 0.0, 1.0, 0.15, 0.05)
        w_ms = st.slider("Markov Switching", 0.0, 1.0, 0.10, 0.05)
    with col2:
        w_hgbr = st.slider("HGBR (Árboles DMS)", 0.0, 1.0, 0.20, 0.05)
        w_en = st.slider("Elastic Net (Kalman)", 0.0, 1.0, 0.15, 0.05)
    with col3:
        w_mlp = st.slider("MLP Neural Net", 0.0, 1.0, 0.10, 0.05)
        w_gpr = st.slider("Gaussian Process (GPR)", 0.0, 1.0, 0.15, 0.05)
    with col4:
        w_foundation = st.slider("Fundacionales (Chronos)", 0.0, 1.0, 0.15, 0.05)
    
    raw_weights = [w_vecm, w_ms, w_hgbr, w_en, w_mlp, w_gpr, w_foundation]
    sum_w = sum(raw_weights)
    if sum_w > 0:
        stress_weights = [float(w / sum_w) for w in raw_weights]
    else:
        stress_weights = [1/7] * 7

st.markdown("---")

# Nota informativa sobre tiempo de ejecución
st.warning("""
⏱️ **Nota de Ejecución:** Debido a que el simulador entrena 7 modelos de Machine Learning en paralelo (incluyendo VECM, GARCH, Markov Switching, Elastic Net, Redes Neuronales MLP, Procesos Gaussianos y Modelos Fundacionales) para realizar una simulación recursiva paso a paso, el proceso tomará **entre 1 y 2 minutos**. Por favor, no recargue la página mientras se ejecuta.

💡 **Nota de Producción:** *En un entorno de producción real, este proceso tardaría menos de 5 segundos*, ya que los modelos estarían pre-entrenados y persistidos en un servidor. Aquí, al no contar con un servidor con estado para guardar los pesos de los modelos, cada ejecución requiere entrenar todo el stack desde cero en la nube de Streamlit.
""")

# 4. Botón de ejecución
if st.button("Ejecutar Backtesting / Simulación con Ensamble ML", type="primary", use_container_width=True):
    st.session_state.resultados_backtest_integral = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(pct, msg):
        progress_bar.progress(pct)
        status_text.text(f"⏳ {msg} ({pct}%)")
        
    try:
        resultados = entrenar_y_predecir_todo(
            df_raw,
            str(fecha_corte),
            variables_exogenas=exogenas,
            predecir_diferencias=False,
            fecha_proyeccion=str(fecha_proyeccion),
            clima_scenario=clima_scenario,
            chicago_scenario_val=chicago_scenario_val,
            devaluacion_mensual_pct=devaluacion_mensual_pct,
            stress_weights=stress_weights,
            progress_callback=update_progress
        )
        st.session_state.resultados_backtest_integral = resultados
        progress_bar.empty()
        status_text.empty()
        st.success("Simulación completada con éxito.")
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error durante la simulación: {e}")

# 5. Renderizar resultados con gráficos premium
if st.session_state.get('resultados_backtest_integral'):
    res = st.session_state.resultados_backtest_integral
    
    st.markdown("---")
    st.header("Resultados de Backtesting: Simulación vs Realidad")
    
    if campana_seleccionada != '2026/27':
        # Tabla de métricas de precisión
        st.subheader("Métricas de Error Fuera de Muestra (Out of Sample)")
        metricas = []
        for target, data in res.items():
            if not isinstance(data, dict) or 'es_exogena' not in data:
                continue
            if data.get('es_exogena', False):
                metricas.append({
                    'Variable': f"{target} (Exógena)",
                    '🎯 MAPE Test': "0.0% (Inyectado)",
                    'Calidad': "Exógena",
                    'MAE Test': "N/A",
                    'R² Aislado': "N/A",
                })
                continue
                
            mape = data['mape_test']
            semaforo = "✅ Excelente" if mape < 10 else "🟡 Aceptable" if mape < 25 else "🔴 Pobre"
            
            metricas.append({
                'Variable': target,
                '🎯 MAPE Test': f"{mape:.1f}%" if not pd.isna(mape) else "N/A",
                'Calidad': semaforo,
                'MAE Test': f"{data['mae_test']:.2f}",
                'R² Aislado': f"{data['r2_train']:.2f}",
            })
            
        st.table(pd.DataFrame(metricas))
    else:
        st.info("📢 **Modo Proyectivo En Vivo / Proyección 2027**\n\n"
                "Dado que estás proyectando la campaña futura 2026/27, no existen datos reales observados todavía para "
                "contrastar (MAE/MAPE/R² no disponibles). Se presentan a continuación las trayectorias proyectadas por el Ensamble y sus bandas de confianza.")

    # Multiselect de modelos individuales para mostrar
    st.markdown("---")
    st.subheader("Visualización del Ensamble y Modelos del Stack")
    modelos_disponibles = ["VECM", "Markov Switching", "HGBR (Direct)", "Elastic Net", "MLP Neural Network", "Gaussian Process", "Modelos Fundacionales (Zero-Shot)"]
    mostrar_individuales = st.multiselect(
        "Selecciona modelos individuales para superponer en los gráficos (líneas punteadas):",
        options=modelos_disponibles,
        default=modelos_disponibles[:3]  # Por defecto mostramos algunos para evitar ruido visual
    )
    
    # Mostrar gráficos principales de trayectoria
    st.subheader("Gráficos de Proyección vs Realidad")
    
    # 1. Precios (FAS USD)
    if 'precio_fas_usd' in res:
        df_comp_fas = res['precio_fas_usd']['df_comparacion']
        fig_fas = plot_backtest_single(df_comp_fas, "Precio Trigo FAS Local (USD/tn) - Simulación vs Realidad", "Precio (USD/tn)", fecha_proyeccion=fecha_proyeccion, modelos_a_mostrar=mostrar_individuales)
        st.plotly_chart(fig_fas, use_container_width=True)
        
    # 2. Precios (FOB USD)
    if 'precio_fob_usd' in res:
        df_comp_fob = res['precio_fob_usd']['df_comparacion']
        fig_fob = plot_backtest_single(df_comp_fob, "Precio Trigo FOB Oficial (USD/tn) - Simulación vs Realidad", "Precio (USD/tn)", fecha_proyeccion=fecha_proyeccion, modelos_a_mostrar=mostrar_individuales)
        st.plotly_chart(fig_fob, use_container_width=True)

    # 3. Flujo Logístico (Camiones)
    col_log1, col_log2 = st.columns(2)
    with col_log1:
        if 'descargas_camiones' in res:
            df_comp_cam = res['descargas_camiones']['df_comparacion']
            fig_cam = plot_backtest_single(df_comp_cam, "Descarga de Camiones Diaria (Puerto)", "Camiones/Día", fecha_proyeccion=fecha_proyeccion, modelos_a_mostrar=[])
            st.plotly_chart(fig_cam, use_container_width=True)
            
    # 4. Rendimiento Estimado (tn/ha)
    with col_log2:
        if 'rendimiento_estimado_tn_ha' in res:
            df_comp_rinde = res['rendimiento_estimado_tn_ha']['df_comparacion']
            fig_rinde = plot_backtest_single(df_comp_rinde, "Rendimiento Estimado (tn/ha)", "tn/ha", fecha_proyeccion=fecha_proyeccion, modelos_a_mostrar=[])
            st.plotly_chart(fig_rinde, use_container_width=True)

    # Volatilidad GARCH(1,1)
    if 'df_garch' in res:
        st.markdown("---")
        st.subheader("📈 Volatilidad Condicional GARCH(1,1) de Precios")
        fig_garch = ui.charts.plot_garch_volatility(res['df_garch'], "Desvío Estándar Condicional Estimado (σ_t)")
        st.plotly_chart(fig_garch, use_container_width=True)
        
    # Evolución del Ensamble DMA
    if 'pesos_dma_fob' in res and len(res['pesos_dma_fob']) > 0:
        import plotly.graph_objects as go
        st.markdown("---")
        st.subheader("⚖️ Evolución de Pesos del Ensamble en Tiempo Real (DMA)")
        col_w1, col_w2 = st.columns(2)
        
        df_comp_fob = res['precio_fob_usd']['df_comparacion']
        n_steps = len(res['pesos_dma_fob'])
        fechas_steps = df_comp_fob['fecha'].iloc[-n_steps:].values
        
        modelos_nombres = ['VECM', 'Markov Switching', 'HGBR (Direct)', 'Kalman (EN)', 'MLP Neural Network', 'Gaussian Process (GPR)', 'Modelos Fundacionales (Zero-Shot)']
        colores_dma = ['#00FFFF', '#FF33FF', '#00FF00', '#E0E0E0', '#FF9800', '#00E676', '#FFD600']
        
        with col_w1:
            df_pesos_fob = pd.DataFrame(res['pesos_dma_fob'], columns=modelos_nombres)
            df_pesos_fob.insert(0, 'fecha', fechas_steps)
            fig_w_fob = go.Figure()
            for col_name, color in zip(modelos_nombres, colores_dma):
                fig_w_fob.add_trace(go.Scatter(
                    x=df_pesos_fob['fecha'], y=df_pesos_fob[col_name],
                    mode='lines', stackgroup='one', name=col_name,
                    line=dict(width=0.5, color=color)
                ))
            fig_w_fob.update_layout(
                title="FOB USD: Evolución de Ponderaciones DMA",
                xaxis_title="Fecha", yaxis_title="Peso",
                yaxis=dict(range=[0, 1]), hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_w_fob, use_container_width=True)

        with col_w2:
            df_pesos_fas = pd.DataFrame(res['pesos_dma_fas'], columns=modelos_nombres)
            df_pesos_fas.insert(0, 'fecha', fechas_steps)
            fig_w_fas = go.Figure()
            for col_name, color in zip(modelos_nombres, colores_dma):
                fig_w_fas.add_trace(go.Scatter(
                    x=df_pesos_fas['fecha'], y=df_pesos_fas[col_name],
                    mode='lines', stackgroup='one', name=col_name,
                    line=dict(width=0.5, color=color)
                ))
            fig_w_fas.update_layout(
                title="FAS USD: Evolución de Ponderaciones DMA",
                xaxis_title="Fecha", yaxis_title="Peso",
                yaxis=dict(range=[0, 1]), hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_w_fas, use_container_width=True)
