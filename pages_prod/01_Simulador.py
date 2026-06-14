import streamlit as st
import pandas as pd
import numpy as np
import time
from engine.simulation import MotorSimulacion
from engine.montecarlo import correr_montecarlo
from engine.models import EventoProgramado
from ui.charts import plot_time_series, plot_montecarlo
from ui.style_prod import inject_custom_css

inject_custom_css()

st.title("1. Simulación de Escenarios")

if not st.session_state.get('reglas_activas'):
  st.warning(" Necesitas generar reglas primero (Pestaña 2 o 3).")
  st.stop()

# Configuración del escenario
st.sidebar.header("Estado Inicial (Hoy)")

if st.session_state.get('estado_base') is not None:
  ultimo_estado = st.session_state.get('estado_base')
elif st.session_state.get('df_proc') is not None:
  # Usar el último registro como base, pero excluir lags y promedios móviles
  ultimo_estado = st.session_state.get('df_proc').iloc[-1].to_dict()
  ultimo_estado = {k: v for k, v in ultimo_estado.items() if '_lag_' not in k and '_rolling_' not in k}
else:
  ultimo_estado = {'precio_bb_ars': 200, 'lluvia_mm': 50, 'descargas_camiones': 1000}

estado_inicial = {}
for key, val in ultimo_estado.items():
  if key == 'fecha':
    continue # skip date column
  if isinstance(val, (bool, np.bool_)):
    continue # skip boolean dummies from get_dummies
  if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
    estado_inicial[key] = st.sidebar.number_input(f"{key}", value=float(val))
  else:
    estado_inicial[key] = val

dias_simular = st.sidebar.slider("Días a simular", min_value=30, max_value=365, value=180)

st.sidebar.markdown("---")
st.sidebar.header("Eventos Discretos (Shocks)")
eventos_manuales = []
if st.sidebar.checkbox("Agregar Paro de Camioneros"):
  inicio_paro = st.sidebar.slider("Día de inicio del paro", 1, dias_simular, 30)
  duracion = st.sidebar.slider("Duración del paro (días)", 1, 30, 7)
  impacto_camiones = st.sidebar.number_input("Impacto en camiones/día", value=-500)
  
  # Evento de inicio
  eventos_manuales.append(EventoProgramado(dia_ejecucion=inicio_paro, variable='descargas_camiones', impacto=impacto_camiones, origen="Shock: Paro Inicia"))
  # Evento de fin
  eventos_manuales.append(EventoProgramado(dia_ejecucion=inicio_paro+duracion, variable='descargas_camiones', impacto=-impacto_camiones, origen="Shock: Paro Termina"))

st.sidebar.markdown("---")
st.sidebar.header("Futuros (Variables Exógenas)")
st.sidebar.caption("Proyecta variables cuyo comportamiento futuro ya conoces (ej: futuros del dólar o chicago).")

datos_futuros = {}
cols_posibles = list(estado_inicial.keys())

var_exogena = st.sidebar.selectbox("Proyectar Variable", options=["Ninguna"] + cols_posibles)
if var_exogena != "Ninguna":
  tasa_diaria = st.sidebar.number_input(f"Crecimiento diario esperado (%)", value=0.1, step=0.01)
  
  # Construir la curva para los días simulados
  valor_actual = estado_inicial[var_exogena]
  curva = []
  for d in range(dias_simular):
    curva.append(valor_actual * ((1 + tasa_diaria/100)**d))
    
  datos_futuros[var_exogena] = curva

# Controles de Simulación
col1, col2 = st.columns(2)
run_deterministic = col1.button("Ejecutar Simulación Central", type="primary")
run_montecarlo = col2.button("Correr Monte Carlo (Probabilidades)")

# Contenedores para UI en tiempo real
grafico_container = st.empty()
log_container = st.empty()

if run_deterministic:
  motor = MotorSimulacion(estado_inicial, st.session_state.get('reglas_activas', []))
  
  with st.spinner("Simulando..."):
    # Animación de progreso para dar sensación de "tick por tick" (opcional para el pitch)
    progress_bar = st.progress(0)
    
    historial = motor.correr(dias=dias_simular, eventos_manuales=eventos_manuales, datos_futuros_conocidos=datos_futuros)
    progress_bar.progress(100)
    
    df_sim = pd.DataFrame([s.valores for s in historial])
    df_sim['dia'] = df_sim.index
    
    st.session_state.resultados_simulacion = df_sim
    
    # Mostrar gráfico
    columnas_numericas = [c for c in df_sim.columns if c != 'dia']
    # Por defecto mostramos precio y camiones
    cols_default = [c for c in columnas_numericas if 'precio' in c or 'camiones' in c]
    if not cols_default:
      cols_default = [columnas_numericas[0]] if columnas_numericas else []
      
    fig = plot_time_series(df_sim, 'dia', cols_default, "Simulación: Escenario Base")
    grafico_container.plotly_chart(fig, width='stretch')
    
    # Mostrar Log
    with log_container.expander("Ver Log del Motor", expanded=False):
      for s in historial:
        if s.reglas_disparadas or s.eventos_ejecutados:
          st.text(f"Día {s.dia}:")
          for r in s.reglas_disparadas:
            st.text(f" → Regla disparada: {r}")
          for e in s.eventos_ejecutados:
            st.text(f" → Impacto aplicado por: {e}")

if run_montecarlo:
  motor = MotorSimulacion(estado_inicial, st.session_state.get('reglas_activas', []))
  
  with st.spinner("Corriendo 1000 simulaciones..."):
    # Variamos variables clave (lluvia, temperatura, precio chicago) un +/- 15% de desvío
    variables_variacion = {}
    for k, v in estado_inicial.items():
      if isinstance(v, (int, float)) and ('lluvia' in k or 'temp' in k or 'chicago' in k):
        variables_variacion[k] = (v, abs(v * 0.15))
        
    resultados_mc = correr_montecarlo(motor, variables_variacion, dias_simular, n_simulaciones=100)
    
    # Graficar variable principal (precio bb)
    var_plot = 'precio_bb_ars' if 'precio_bb_ars' in resultados_mc else list(resultados_mc.keys())[0]
    
    fig_mc = plot_montecarlo(resultados_mc[var_plot], f"Simulación Monte Carlo: {var_plot}", "Valor")
    grafico_container.plotly_chart(fig_mc, width='stretch')
    
    st.info(" **A diferencia de una predicción estática**, el abanico muestra cómo la incertidumbre se expande con el tiempo basándose en la variabilidad climática y de mercado.")
