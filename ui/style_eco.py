import streamlit as st

def inject_custom_css():
    """
    Inyecta CSS personalizado para transformar la apariencia de Streamlit 
    en una Consola Bloomberg auténtica (alta densidad, fondo negro, ámbar y verde neón).
    """
    css = """
    <style>
    /* 1. Ingesta de Tipografía Monospace y Global Override */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Roboto Mono', monospace !important;
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    /* 2. Forzar Fondo Negro Absoluto en Contenedores */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    
    /* 3. Ajuste de Spacing Extremo (Alta Densidad Bloomberg) */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 95% !important;
    }
    
    /* 4. Barra Lateral (Sidebar) estilo Terminal de Comandos */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #333333 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #FF9900 !important; /* Ámbar para textos en sidebar */
        font-size: 13px !important;
    }
    
    /* Estilo para los botones de navegación de la barra lateral */
    [data-testid="stSidebar"] button {
        background-color: transparent !important;
        color: #00FF00 !important; /* Menú en verde neón */
        border: none !important;
        text-align: left !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
        border-radius: 0px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #002200 !important; /* Highlight verde oscuro */
        color: #FF9900 !important; /* Cambio a ámbar en hover */
    }
    
    /* 5. Tarjetas de Métricas de Bloomberg (Bloomberg Tickers) */
    [data-testid="stMetric"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
        padding: 10px !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: #00FF00 !important; /* Borde verde neón en hover */
    }
    
    /* Etiqueta de la Métrica */
    [data-testid="stMetricLabel"] div {
        color: #FF9900 !important; /* Ámbar */
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    
    /* Valor de la Métrica */
    [data-testid="stMetricValue"] div {
        color: #00FF00 !important; /* Verde Neón */
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    /* Cambio (Delta) de la Métrica */
    [data-testid="stMetricDelta"] {
        font-size: 11px !important;
    }

    /* 6. Botones de Comando (Bloomberg Commands) */
    .stButton>button {
        background-color: #000000 !important;
        color: #FF9900 !important; /* Ámbar */
        border: 1px solid #FF9900 !important;
        border-radius: 0px !important;
        padding: 4px 14px !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        transition: none !important; /* Instantáneo */
    }
    
    .stButton>button:hover, .stButton>button:active {
        background-color: #00FF00 !important; /* Verde Neón */
        color: #000000 !important; /* Texto negro */
        border-color: #00FF00 !important;
    }
    
    /* Botones primarios específicos */
    .stButton>button[kind="primary"] {
        background-color: #000000 !important;
        color: #00FFFF !important; /* Cian para llamadas a la acción principales */
        border-color: #00FFFF !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #00FFFF !important;
        color: #000000 !important;
    }

    /* 7. Inputs, Selectboxes e Inputs Numéricos estilo Terminal */
    input, select, textarea, [data-baseweb="select"], [data-baseweb="base-input"] {
        background-color: #000000 !important;
        color: #00FF00 !important; /* Texto que escribe es verde neón */
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 13px !important;
    }
    
    /* Enfocar controles */
    input:focus, select:focus, textarea:focus, [data-baseweb="select"]:focus {
        border-color: #FF9900 !important; /* Ámbar al enfocar */
        outline: none !important;
    }
    
    /* Etiquetas de Inputs */
    label, [data-testid="stWidgetLabel"] p {
        color: #FF9900 !important; /* Ámbar para etiquetas */
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    /* 8. Tablas de Datos (Bloomberg Data Grid) */
    [data-testid="stTable"], [data-testid="stDataFrame"] {
        background-color: #000000 !important;
        border: 1px solid #222222 !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    th {
        background-color: #0A0A0A !important;
        border-bottom: 2px solid #FF9900 !important; /* Borde inferior de headers en ámbar */
        color: #00FFFF !important; /* Headers de tabla en cian */
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    
    td {
        background-color: #000000 !important;
        border-bottom: 1px solid #222222 !important;
        color: #FFFFFF !important;
        font-size: 12px !important;
    }
    
    tr:hover td {
        background-color: #001100 !important; /* Fila seleccionada en verde muy oscuro */
        color: #00FF00 !important;
    }

    /* 9. Estilizar Expanders (Bloomberg Panels) */
    [data-testid="stExpander"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }
    [data-testid="stExpander"] summary {
        color: #FF9900 !important; /* Título expander en ámbar */
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #00FF00 !important;
    }

    /* 10. Estilizar Pestañas (Tabs) */
    button[data-baseweb="tab"] {
        background-color: #000000 !important;
        color: #888888 !important;
        border: none !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        padding: 8px 16px !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #00FF00 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF9900 !important; /* Ámbar para pestaña seleccionada */
        border-bottom: 2px solid #FF9900 !important;
    }

    /* 11. Títulos y Encabezados */
    h1 {
        color: #FF9900 !important; /* Título principal en Ámbar */
        font-size: 24px !important;
        font-weight: 700 !important;
        border-bottom: 2px double #FF9900 !important;
        padding-bottom: 5px !important;
        margin-bottom: 15px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    h2 {
        color: #00FFFF !important; /* Subtítulos en Cian */
        font-size: 18px !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #333333 !important;
        padding-bottom: 4px !important;
        margin-top: 20px !important;
        text-transform: uppercase !important;
    }
    
    h3 {
        color: #FF9900 !important; /* Títulos menores en Ámbar */
        font-size: 14px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }

    /* 12. Avisos y Alertas estilo Consola */
    .stAlert {
        background-color: #000000 !important;
        border: 1px solid #FF9900 !important; /* Borde ámbar para alertas */
        border-radius: 0px !important;
    }
    .stAlert [data-testid="stMarkdownContainer"] p {
        color: #FF9900 !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 12px !important;
    }
    
    /* Alertas de error en rojo terminal */
    .stAlert[data-testid="stNotificationError"] {
        border-color: #FF3333 !important;
    }
    .stAlert[data-testid="stNotificationError"] p {
        color: #FF3333 !important;
    }
    
    /* Alertas de éxito en verde terminal */
    .stAlert[data-testid="stNotificationSuccess"] {
        border-color: #00FF00 !important;
    }
    .stAlert[data-testid="stNotificationSuccess"] p {
        color: #00FF00 !important;
    }

    /* 13. Sliders Estilo Consola */
    .stSlider {
        padding-bottom: 10px !important;
    }
    .stSlider [data-testid="stWidgetLabel"] {
        color: #FF9900 !important;
    }
    
    /* 14. Código y Terminal Output */
    code, pre {
        background-color: #0A0A0A !important;
        color: #00FF00 !important; /* Código en verde neón */
        border: 1px solid #222222 !important;
        border-radius: 0px !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
