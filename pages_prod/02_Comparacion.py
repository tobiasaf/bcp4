import streamlit as st
import pandas as pd
from engine.simulation import MotorSimulacion
from ui.charts import plot_time_series
import plotly.graph_objects as go
from ui.style_prod import inject_custom_css

inject_custom_css()

st.title("2. Comparar Escenarios")

if not st.session_state.get('reglas_activas'):
  st.warning("No hay reglas activas. Por favor, asegúrese de que el modelo esté desplegado en la base de datos.")
  st.stop()

# Obtener estado base (desde la BD, desde df_proc o fallback)
if st.session_state.get('estado_base') is not None:
  estado_base = st.session_state.get('estado_base')
elif st.session_state.get('df_proc') is not None:
  estado_base = st.session_state.get('df_proc').iloc[-1].to_dict()
  estado_base = {k: v for k, v in estado_base.items() if '_lag_' not in k and '_rolling_' not in k and k != 'fecha'}
else:
  estado_base = {'precio_bb_ars': 200, 'lluvia_mm': 50, 'descargas_camiones': 1000}

lluvia_default = float(estado_base.get('lluvia_mm', 50.0))
desc_default = float(estado_base.get('descargas_camiones', 1000.0))

st.markdown("Compara cómo impactan distintas condiciones iniciales (ej: Niña Fuerte vs Neutral) a lo largo del tiempo.")

col1, col2 = st.columns(2)

with col1:
  st.subheader("Escenario A: Niña Fuerte")
  lluvia_a = st.number_input("Lluvia mensual (mm)", value=max(0.0, lluvia_default * 0.4), key="lluv_a")
  desc_a = st.number_input("Camiones en puerto", value=desc_default, key="cam_a")

with col2:
  st.subheader("Escenario B: Neutral")
  lluvia_b = st.number_input("Lluvia mensual (mm)", value=lluvia_default, key="lluv_b")
  desc_b = st.number_input("Camiones en puerto", value=desc_default, key="cam_b")

if st.button("Comparar Escenarios", type="primary"):
  estado_a = estado_base.copy()
  estado_a['lluvia_mm'] = lluvia_a
  estado_a['descargas_camiones'] = desc_a
  
  estado_b = estado_base.copy()
  estado_b['lluvia_mm'] = lluvia_b
  estado_b['descargas_camiones'] = desc_b
  
  import copy
  reglas_a = copy.deepcopy(st.session_state.get('reglas_activas', []))
  reglas_b = copy.deepcopy(st.session_state.get('reglas_activas', []))
  motor_a = MotorSimulacion(estado_a, reglas_a)
  motor_b = MotorSimulacion(estado_b, reglas_b)
  
  with st.spinner("Corriendo simulaciones..."):
    hist_a = motor_a.correr(dias=180)
    hist_b = motor_b.correr(dias=180)
    
    df_a = pd.DataFrame([s.valores for s in hist_a])
    df_b = pd.DataFrame([s.valores for s in hist_b])
    
    # Combinar para graficar
    var_grafico = 'precio_bb_ars' if 'precio_bb_ars' in df_a.columns else df_a.columns[0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_a.index, y=df_a[var_grafico], name="Escenario A (Niña Fuerte)", line=dict(color="#E74C3C")))
    fig.add_trace(go.Scatter(x=df_b.index, y=df_b[var_grafico], name="Escenario B (Neutral)", line=dict(color="#2E86C1")))
    
    y_unit = "ARS/Tn" if "ars" in var_grafico.lower() else "USD/Tn" if "usd" in var_grafico.lower() else "Valor"
    fig.update_layout(
      title=f"Comparación de {var_grafico}",
      template="plotly_dark",
      hovermode="x unified",
      xaxis_title="Días simulados",
      yaxis_title=f"Precio ({y_unit})" if "precio" in var_grafico.lower() else f"Valor ({y_unit})"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa fin de simulación
    st.subheader("Resultados al final de la simulación (Día 180)")
    res_a = df_a.iloc[-1]
    res_b = df_b.iloc[-1]
    
    df_comp = pd.DataFrame({
      'Escenario A': res_a,
      'Escenario B': res_b,
      'Diferencia (%)': ((res_a - res_b) / res_b * 100).round(2)
    })
    
    st.dataframe(df_comp)
