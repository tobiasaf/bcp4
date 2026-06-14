import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# Colores institucionales / dashboard (Dark Fintech Premium)
COLOR_PRIMARY = "#2962FF" # Azul Eléctrico
COLOR_SECONDARY = "#00E676" # Esmeralda
COLOR_DANGER = "#FF3D00" # Coral
COLOR_WARNING = "#F9A826" # Amarillo
COLOR_BG = "rgba(0,0,0,0)" # Fondo Transparente

def apply_transparent_layout(fig):
    """Layout claro/transparente por defecto (para productores)."""
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Segoe UI, Helvetica, sans-serif", size=12, color="#333333"),
        xaxis=dict(showgrid=True, gridcolor="#E0E0E0", linecolor="#CCCCCC"),
        yaxis=dict(showgrid=True, gridcolor="#E0E0E0", linecolor="#CCCCCC")
    )
    return fig

def apply_bloomberg_layout(fig):
    """Layout estilo Consola Bloomberg de alta fidelidad (para economistas)."""
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(family="'Roboto Mono', monospace", size=11, color="#FF9900"),
        xaxis=dict(
            showgrid=True, 
            gridcolor="#112211", # Grilla verde muy oscura
            linecolor="#333333", 
            tickfont=dict(color="#00FF00") # Ticks en verde neón
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="#112211", 
            linecolor="#333333", 
            tickfont=dict(color="#00FF00")
        ),
        legend=dict(
            font=dict(color="#00FFFF", size=10), # Leyendas en cian
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#333333",
            borderwidth=1
        )
    )
    return fig

def apply_layout_bcp(fig):
    """Aplica el layout correcto según el contexto del usuario en session_state."""
    if st.session_state.get('es_economista', False):
        return apply_bloomberg_layout(fig)
    else:
        return apply_transparent_layout(fig)

def plot_time_series(df: pd.DataFrame, x_col: str, y_cols: list, title: str):
    """Grafica una o más series de tiempo."""
    fig = go.Figure()
    
    is_eco = st.session_state.get('es_economista', False)
    COLORS_BLOOMBERG = ["#00FF00", "#00FFFF", "#FF9900", "#FFFFFF", "#FF33FF", "#FFFF00"]
    
    for i, col in enumerate(y_cols):
        if is_eco:
            color = COLORS_BLOOMBERG[i % len(COLORS_BLOOMBERG)]
            width = 1.2 # Líneas más delgadas estilo terminal
        else:
            color = None
            width = 2
            
        fig.add_trace(go.Scatter(
            x=df[x_col], 
            y=df[col], 
            mode='lines', 
            name=col,
            line=dict(color=color, width=width)
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark" if is_eco else "plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return apply_layout_bcp(fig)

def plot_montecarlo(df_percentiles: pd.DataFrame, title: str, ylabel: str):
    """Grafica el abanico de probabilidades de Monte Carlo."""
    fig = go.Figure()
    
    dias = df_percentiles.index
    is_eco = st.session_state.get('es_economista', False)
    
    # Configurar colores según contexto
    if is_eco:
        color_band_90 = 'rgba(0, 255, 0, 0.08)' # Verde translúcido tenue
        color_band_50 = 'rgba(0, 255, 0, 0.2)'
        color_median = '#00FF00'
        width_median = 2
    else:
        color_band_90 = 'rgba(46, 134, 193, 0.15)'
        color_band_50 = 'rgba(46, 134, 193, 0.3)'
        color_median = COLOR_PRIMARY
        width_median = 3
        
    # Banda 5% - 95%
    fig.add_trace(go.Scatter(
        x=dias.tolist() + dias.tolist()[::-1],
        y=df_percentiles['p95'].tolist() + df_percentiles['p5'].tolist()[::-1],
        fill='toself',
        fillcolor=color_band_90,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Banda 90% Confianza'
    ))
    
    # Banda 25% - 75%
    fig.add_trace(go.Scatter(
        x=dias.tolist() + dias.tolist()[::-1],
        y=df_percentiles['p75'].tolist() + df_percentiles['p25'].tolist()[::-1],
        fill='toself',
        fillcolor=color_band_50,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Banda 50% Confianza'
    ))
    
    # Escenario Central (Mediana)
    fig.add_trace(go.Scatter(
        x=dias,
        y=df_percentiles['p50'],
        line=dict(color=color_median, width=width_median),
        name='Escenario Central (Mediana)'
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        xaxis_title="Días simulados",
        template="plotly_dark" if is_eco else "plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return apply_layout_bcp(fig)

def plot_tornado(df_importances: pd.DataFrame, title: str = "Importancia de Variables"):
    """Grafica un Tornado Chart de importancias."""
    is_eco = st.session_state.get('es_economista', False)
    
    # Escala de color estilo terminal o corporativa
    scale = [[0, '#001100'], [1, '#00FF00']] if is_eco else 'Blues'
    
    fig = px.bar(
        df_importances, 
        x='importancia', 
        y='feature', 
        orientation='h',
        title=title,
        color='importancia',
        color_continuous_scale=scale
    )
    fig.update_layout(
        template="plotly_dark" if is_eco else "plotly_white", 
        yaxis={'categoryorder':'total ascending'},
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return apply_layout_bcp(fig)

def plot_backtest_single(df_comp: pd.DataFrame, title: str, ylabel: str, fecha_proyeccion: str = None, modelos_a_mostrar: list = None):
    """Grafica la línea real (sólida) vs la predicción (punteada) con bandas de confianza."""
    fig = go.Figure()
    
    is_eco = st.session_state.get('es_economista', False)
    
    # Colores estilo consola Bloomberg
    color_real = "#FF9900" if is_eco else COLOR_PRIMARY   # Ámbar
    color_pred = "#00FF00" if is_eco else COLOR_WARNING   # Verde neón
    
    # Shaded confidence band (dibujada primero para quedar al fondo)
    if 'lower' in df_comp.columns and 'upper' in df_comp.columns:
        color_band = 'rgba(0, 255, 255, 0.12)' if is_eco else 'rgba(41, 98, 255, 0.12)'
        df_clean = df_comp.dropna(subset=['lower', 'upper']).copy()
        
        fig.add_trace(go.Scatter(
            x=df_clean['fecha'].tolist() + df_clean['fecha'].tolist()[::-1],
            y=df_clean['upper'].tolist() + df_clean['lower'].tolist()[::-1],
            fill='toself',
            fillcolor=color_band,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Banda Confianza (80%)'
        ))
    
    fig.add_trace(go.Scatter(
        x=df_comp['fecha'], 
        y=df_comp['real'], 
        mode='lines', 
        name='Real',
        line=dict(color=color_real, width=1.2 if is_eco else 2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_comp['fecha'], 
        y=df_comp['prediccion'], 
        mode='lines', 
        name='Predicción Ensemble',
        line=dict(color=color_pred, width=1.2 if is_eco else 2, dash='dash')
    ))
    
    # Modelos individuales (si existen en el DataFrame y están solicitados/permitidos)
    modelos_map = {
        'pred_vecm': ('VECM', '#00FFFF' if is_eco else '#0288D1'),
        'pred_ms': ('Markov Switching', '#FF33FF' if is_eco else '#7B1FA2'),
        'pred_hgbr': ('HGBR (Direct)', '#00FF00' if is_eco else '#388E3C'),
        'pred_en': ('Elastic Net', '#E0E0E0' if is_eco else '#78909C'),
        'pred_mlp': ('MLP Neural Network', '#FF9800' if is_eco else '#F57C00'),
        'pred_gpr': ('Gaussian Process', '#00E676' if is_eco else '#2E7D32'),
        'pred_foundation': ('Modelos Fundacionales (Zero-Shot)', '#FFD600' if is_eco else '#FBC02D')
    }
    
    for col, (label, color) in modelos_map.items():
        if col in df_comp.columns:
            visible = 'legendonly'
            if modelos_a_mostrar:
                multiselect_labels = {
                    'VECM': 'pred_vecm',
                    'Markov Switching': 'pred_ms',
                    'HGBR (Direct)': 'pred_hgbr',
                    'Elastic Net': 'pred_en',
                    'MLP Neural Network': 'pred_mlp',
                    'Gaussian Process': 'pred_gpr',
                    'Modelos Fundacionales (Zero-Shot)': 'pred_foundation'
                }
                label_name = [k for k, v in multiselect_labels.items() if v == col][0]
                if label_name in modelos_a_mostrar:
                    visible = True
                
            fig.add_trace(go.Scatter(
                x=df_comp['fecha'],
                y=df_comp[col],
                mode='lines',
                name=label,
                visible=visible,
                line=dict(color=color, width=1.0 if is_eco else 1.5, dash='dot')
            ))
    
    # Agregar línea vertical que marca el inicio de la proyección a ciegas
    if fecha_proyeccion:
        fig.add_vline(
            x=pd.to_datetime(fecha_proyeccion),
            line_width=1.5,
            line_dash="dashdot",
            line_color="#E040FB" if not is_eco else "#00FFFF",  # Magenta neón o Cian neón
        )
        fig.add_annotation(
            x=pd.to_datetime(fecha_proyeccion),
            y=1.0,
            yref="paper",  # Posicionamiento respecto al tope del gráfico
            text=" Proyección a ciegas",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="#E040FB" if not is_eco else "#00FFFF", size=9)
        )
    
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template="plotly_dark" if is_eco else "plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return apply_layout_bcp(fig)

def plot_backtest_with_overlays(df: pd.DataFrame, target: str, overlay_cols: list, title: str):
    """Grafica la línea real vs predicción (eje principal) y superpone predictoras (eje secundario)."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    is_eco = st.session_state.get('es_economista', False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Colores estilo consola Bloomberg
    color_real = "#FF9900" if is_eco else COLOR_PRIMARY   # Ámbar o Azul
    color_pred = "#00FF00" if is_eco else COLOR_SECONDARY # Verde neón o Esmeralda
    
    # Eje principal: Real
    fig.add_trace(
        go.Scatter(
            x=df['fecha'], 
            y=df['real'], 
            mode='lines', 
            name=f'{target} (Real)',
            line=dict(color=color_real, width=1.5 if is_eco else 2.5)
        ),
        secondary_y=False
    )
    
    # Eje principal: Predicción
    fig.add_trace(
        go.Scatter(
            x=df['fecha'], 
            y=df['prediccion'], 
            mode='lines', 
            name=f'{target} (Predicción)',
            line=dict(color=color_pred, width=1.5 if is_eco else 2.5, dash='dash')
        ),
        secondary_y=False
    )
    
    # Colores contrastantes para variables superpuestas
    COLORS_OVERLAY = ["#00FFFF", "#FF33FF", "#FFFF00", "#FFFFFF"] if is_eco else ["#FF3D00", "#F9A826", "#9C27B0", "#607D8B"]
    
    for i, col in enumerate(overlay_cols):
        color = COLORS_OVERLAY[i % len(COLORS_OVERLAY)]
        fig.add_trace(
            go.Scatter(
                x=df['fecha'], 
                y=df[col], 
                mode='lines', 
                name=f'{col} (Der)',
                line=dict(color=color, width=1.0 if is_eco else 1.8, dash='dot')
            ),
            secondary_y=True
        )
        
    fig.update_layout(
        title=title,
        template="plotly_dark" if is_eco else "plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor="rgba(0,0,0,0.8)" if is_eco else "rgba(255,255,255,0.8)",
            bordercolor="#333333" if is_eco else "#CCCCCC",
            borderwidth=1
        )
    )
    
    # Títulos de ejes
    fig.update_yaxes(title_text="Eje Principal (Valor)", secondary_y=False)
    if overlay_cols:
        fig.update_yaxes(title_text="Eje Secundario (Predictoras)", secondary_y=True)
        
    return apply_layout_bcp(fig)


def plot_local_attribution(df_contrib: pd.DataFrame, title: str):
    """
    Grafica la contribución local de cada variable para una fecha dada en formato de barras horizontales.
    """
    is_eco = st.session_state.get('es_economista', False)
    
    # Clonar para evitar modificar el original
    df_plot = df_contrib.copy()
    
    # Calcular contribución absoluta para ordenar
    df_plot['abs_contribucion'] = df_plot['contribucion'].abs()
    df_plot = df_plot.sort_values('abs_contribucion', ascending=True) # Ascendente para que la barra más larga quede arriba
    
    # Asignar colores según el signo de la contribución
    colors = [
        '#00FF00' if c >= 0 else '#FF3333' 
        for c in df_plot['contribucion']
    ] if is_eco else [
        '#00E676' if c >= 0 else '#FF3D00'
        for c in df_plot['contribucion']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot['feature'],
        x=df_plot['contribucion'],
        orientation='h',
        marker_color=colors,
        text=[f"{c:+.2f}" for c in df_plot['contribucion']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Contribución: %{x:+.2f}<br>Valor actual: %{customdata[0]:.2f}<br>Media histórica: %{customdata[1]:.2f}<extra></extra>",
        customdata=df_plot[['valor_actual', 'valor_medio']].values
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Contribución al Valor Predicho",
        yaxis_title="Variable Predictora",
        template="plotly_dark" if is_eco else "plotly_white",
        margin=dict(l=20, r=60, t=45, b=20),
        showlegend=False
    )
    
    # Dibujar una línea vertical en x=0 para referencia
    fig.add_vline(x=0.0, line_width=1, line_dash="dash", line_color="#555555" if is_eco else "#888888")
    
    return apply_layout_bcp(fig)


def plot_garch_volatility(df_vol: pd.DataFrame, title: str):
    """
    Grafica la volatilidad condicional calculada por GARCH(1,1) para FOB y FAS.
    """
    fig = go.Figure()
    is_eco = st.session_state.get('es_economista', False)
    
    color_fob = "#00FFFF" if is_eco else "#0288D1"  # Cian o Azul
    color_fas = "#FF33FF" if is_eco else "#7B1FA2"  # Magenta o Púrpura
    
    if 'vol_fob' in df_vol.columns:
        fig.add_trace(go.Scatter(
            x=df_vol['fecha'], 
            y=df_vol['vol_fob'], 
            mode='lines', 
            name='Volatilidad Condicional FOB (GARCH)',
            line=dict(color=color_fob, width=1.5 if is_eco else 2)
        ))
        
    if 'vol_fas' in df_vol.columns:
        fig.add_trace(go.Scatter(
            x=df_vol['fecha'], 
            y=df_vol['vol_fas'], 
            mode='lines', 
            name='Volatilidad Condicional FAS (GARCH)',
            line=dict(color=color_fas, width=1.5 if is_eco else 2)
        ))
        
    fig.update_layout(
        title=title,
        yaxis_title="Desvío Estándar Condicional (σ_t)",
        template="plotly_dark" if is_eco else "plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20)
    )
    return apply_layout_bcp(fig)


