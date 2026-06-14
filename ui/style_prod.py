import streamlit as st

def inject_custom_css():
    """
    Inyecta CSS personalizado para transformar la apariencia de Streamlit 
    en una interfaz profesional tipo Fintech (Dark Mode Premium).
    """
    css = """
    <style>
    /* 1. Tipografía Global Corporativa (Helvetica/Segoe UI) */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif !important;
        background-color: #F8F9FA !important;
        color: #333333 !important;
    }

    /* 2. Reducir Padding Superior */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* 3. Tarjetas para las Métricas (Cards) Limpias */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    [data-testid="stMetric"]:hover {
        border-color: #0056B3; /* Azul corporativo */
    }

    /* 4. Botones Corporativos (Sólidos) */
    .stButton>button {
        background: #0056B3 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: opacity 0.2s ease !important;
    }
    
    .stButton>button:hover {
        opacity: 0.9 !important;
    }

    /* 5. Estilizar Expanders (Acordeones) */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 4px !important;
    }

    /* 6. Barra lateral (Sidebar) clara */
    [data-testid="stSidebar"] {
        background-color: #F1F3F5 !important;
        border-right: 1px solid #E0E0E0 !important;
    }

    /* 7. Tablas limpias corporativas */
    [data-testid="stTable"] {
        background-color: #FFFFFF !important;
    }
    th {
        border-bottom: 2px solid #0056B3 !important;
        color: #333333 !important;
    }
    td {
        border-bottom: 1px solid #E0E0E0 !important;
    }

    /* 8. Fix para Sliders */
    .stSlider [data-testid="stTickBar"] {
        background-color: #2A2F3D !important;
    }

    /* 9. Títulos con peso moderno */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
