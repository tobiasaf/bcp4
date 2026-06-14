import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import json




COLS_NO_LAG = [
    'precio_bb_ars',        
    'precio_fas_usd',       
    'precio_pizarra_usd',   
    'basis_usd',            
    'descargas_camiones_tn',
    'descargas_vagones_tn', 
    'temp_media',           
    'compras_se',           
    'compras_si',           
    'compras_totales',      
    'compras_sin_precio_tot',
    'delta_compras_si',     
]


COLS_LAG_CORTO = [
    'precio_fob_usd', 'precio_fas_ars', 'precio_chicago_usd', 'tipo_cambio',
    'brecha_cambiaria_pct', 'precio_urea_usd', 'precio_map_usd',
    'compras_sin_precio_pct', 'rofex_precio_usd',
    
    'precio_blue_usd', 'precio_mep_usd', 'brecha_blue_pct', 'brecha_ccl_pct',
    'riesgo_pais_embi', 'tasa_politica_pct', 'dxy_index', 'petroleo_wti_usd',
    'cbot_maiz_usd', 'cbot_soja_usd'
]


COLS_LAG_MEDIO = [
    'descargas_camiones', 'descargas_vagones', 'embarques_tn',
    'lluvia_mm', 'delta_compras_se', 'delta_compras_totales',
    
    'cot_managed_money_net', 'cot_commercial_net', 'nivel_parana_m'
]


COLS_LAG_LARGO = [
    'rendimiento_estimado_tn_ha', 'superficie_cosechada_ha', 'ndvi_anomalia_pct',
    
    'wasde_stocks_to_use', 'wasde_arg_export_mt', 'gtrends_vender_trigo',
    'gtrends_dolar', 'nino34_anomalia'
]


def crear_lags_optimizados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de lag de forma eficiente usando pd.concat.
    Diferencia los horizontes de lag según el tipo de variable.
    """
    nuevas_cols = {}

    for col in COLS_LAG_CORTO:
        if col in df.columns:
            for lag in range(1, 5):  
                nuevas_cols[f'{col}_lag_{lag}'] = df[col].shift(lag)

    for col in COLS_LAG_MEDIO:
        if col in df.columns:
            for lag in range(1, 9):  
                nuevas_cols[f'{col}_lag_{lag}'] = df[col].shift(lag)

    for col in COLS_LAG_LARGO:
        if col in df.columns:
            for lag in [1, 2, 4, 8, 12]:  
                nuevas_cols[f'{col}_lag_{lag}'] = df[col].shift(lag)

    if nuevas_cols:
        df_lags = pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)
    else:
        df_lags = df.copy()

    return df_lags


def crear_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea rolling mean y rolling std para variables clave con shift(1)
    para evitar data leakage. Aumenta min_periods a 4 (o min(window, 4)).
    """
    nuevas_cols = {}

    
    for col in ['precio_fob_usd', 'precio_chicago_usd', 'rofex_precio_usd']:
        if col in df.columns:
            for w in [4, 8]:
                min_p = min(w, 4)
                nuevas_cols[f'{col}_rolling_mean_{w}'] = df[col].shift(1).rolling(window=w, min_periods=min_p).mean()
                nuevas_cols[f'{col}_rolling_std_{w}'] = df[col].shift(1).rolling(window=w, min_periods=min_p).std().fillna(0)

    
    for col in ['lluvia_mm', 'descargas_camiones', 'descargas_vagones', 'embarques_tn', 'rofex_volumen']:
        if col in df.columns:
            for w in [4, 8]:
                min_p = min(w, 4)
                nuevas_cols[f'{col}_rolling_mean_{w}'] = df[col].shift(1).rolling(window=w, min_periods=min_p).mean()

    if nuevas_cols:
        return pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)
    return df.copy()


def crear_features_estacionales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de estacionalidad basadas en la fecha.
    Usa codificación cíclica (sin/cos) para que semana 52 ≈ semana 1.
    """
    if 'fecha' not in df.columns:
        return df

    nuevas_cols = {}
    fecha = pd.to_datetime(df['fecha'])

    
    semana = fecha.dt.isocalendar().week.astype(float)
    nuevas_cols['semana_sin'] = np.sin(2 * np.pi * semana / 52)
    nuevas_cols['semana_cos'] = np.cos(2 * np.pi * semana / 52)

    
    mes = fecha.dt.month.astype(float)
    nuevas_cols['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    nuevas_cols['mes_cos'] = np.cos(2 * np.pi * mes / 12)

    
    
    
    
    
    mes_int = fecha.dt.month
    condiciones = [
        mes_int.isin([5, 6, 7]),
        mes_int.isin([8, 9, 10]),
        mes_int.isin([11, 12, 1]),
        mes_int.isin([2, 3, 4]),
    ]
    valores = [0, 1, 2, 3]  
    fase = np.select(condiciones, valores, default=0).astype(float)
    nuevas_cols['fase_campaña_sin'] = np.sin(2 * np.pi * fase / 4)
    nuevas_cols['fase_campaña_cos'] = np.cos(2 * np.pi * fase / 4)

    return pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)


def crear_features_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de ratio escala-invariante que no dependen de la inflación.
    Estos ratios son estables entre train y test (~5% de variación).
    Aplica shift(1) a retornos semanales para erradicar data leakage.
    """
    nuevas_cols = {}

    
    if 'precio_fob_usd' in df.columns and 'precio_chicago_usd' in df.columns:
        fob_premium = df['precio_fob_usd'] / df['precio_chicago_usd'].replace(0, np.nan)
        nuevas_cols['fob_premium'] = fob_premium
        
        for lag in [1, 2, 4]:
            nuevas_cols[f'fob_premium_lag_{lag}'] = fob_premium.shift(lag)

    
    if 'precio_fas_ars' in df.columns and 'tipo_cambio' in df.columns and 'precio_chicago_usd' in df.columns:
        fas_usd = df['precio_fas_ars'] / df['tipo_cambio'].replace(0, np.nan)
        fas_discount = fas_usd / df['precio_chicago_usd'].replace(0, np.nan)
        nuevas_cols['fas_discount'] = fas_discount
        for lag in [1, 2, 4]:
            nuevas_cols[f'fas_discount_lag_{lag}'] = fas_discount.shift(lag)

    
    if 'precio_fob_usd' in df.columns:
        nuevas_cols['fob_retorno_1w'] = df['precio_fob_usd'].pct_change(1).shift(1)
        nuevas_cols['fob_retorno_4w'] = df['precio_fob_usd'].pct_change(4).shift(1)

    
    if 'precio_chicago_usd' in df.columns:
        nuevas_cols['chicago_retorno_1w'] = df['precio_chicago_usd'].pct_change(1).shift(1)
        nuevas_cols['chicago_retorno_4w'] = df['precio_chicago_usd'].pct_change(4).shift(1)

    
    
    

    
    if 'precio_chicago_usd' in df.columns:
        if 'cbot_maiz_usd' in df.columns:
            ratio_maiz = df['precio_chicago_usd'] / df['cbot_maiz_usd'].replace(0, np.nan)
            nuevas_cols['ratio_trigo_maiz'] = ratio_maiz
            for lag in [1, 2, 4]:
                nuevas_cols[f'ratio_trigo_maiz_lag_{lag}'] = ratio_maiz.shift(lag)
        if 'cbot_soja_usd' in df.columns:
            ratio_soja = df['precio_chicago_usd'] / df['cbot_soja_usd'].replace(0, np.nan)
            nuevas_cols['ratio_trigo_soja'] = ratio_soja
            for lag in [1, 2, 4]:
                nuevas_cols[f'ratio_trigo_soja_lag_{lag}'] = ratio_soja.shift(lag)

    
    if 'nivel_parana_m' in df.columns:
        anom_rio = df['nivel_parana_m'] - 3.4
        nuevas_cols['nivel_parana_anomalia'] = anom_rio
        for lag in [1, 2, 4]:
            nuevas_cols[f'nivel_parana_anomalia_lag_{lag}'] = anom_rio.shift(lag)

    
    if 'gtrends_vender_trigo' in df.columns:
        
        nuevas_cols['gtrends_venta_momentum'] = df['gtrends_vender_trigo'].pct_change(1).shift(1).fillna(0.0)

    
    if 'cot_managed_money_net' in df.columns:
        
        nuevas_cols['cot_mm_net_change_1w'] = df['cot_managed_money_net'].diff(1).shift(1).fillna(0.0)

    if nuevas_cols:
        return pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)
    return df.copy()


def crear_features_regimen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables dummy de régimen y variables de interacción
    para capturar dinámicas específicas de mercado (e.g. cosecha vs comercialización).
    Unifica mes de Mayo (5) a régimen de Siembra para alineación estacional.
    """
    if 'fecha' not in df.columns:
        return df

    nuevas_cols = {}
    mes = pd.to_datetime(df['fecha']).dt.month

    
    
    regimen_cosecha = mes.isin([11, 12, 1]).astype(float)
    nuevas_cols['regimen_cosecha'] = regimen_cosecha

    
    regimen_comercializacion = mes.isin([2, 3, 4]).astype(float)
    nuevas_cols['regimen_comercializacion'] = regimen_comercializacion

    
    regimen_siembra = mes.isin([5, 6, 7, 8, 9, 10]).astype(float)
    nuevas_cols['regimen_siembra'] = regimen_siembra

    
    if 'fas_discount_lag_1' in df.columns:
        nuevas_cols['fas_discount_lag_1_x_regimen_cosecha'] = df['fas_discount_lag_1'] * regimen_cosecha
        nuevas_cols['fas_discount_lag_1_x_regimen_comercializacion'] = df['fas_discount_lag_1'] * regimen_comercializacion

    
    if 'semana_sin' in df.columns:
        nuevas_cols['semana_sin_x_regimen_cosecha'] = df['semana_sin'] * regimen_cosecha

    
    if 'ndvi_anomalia_pct' in df.columns:
        nuevas_cols['ndvi_anomalia_octubre_x_regimen_cosecha'] = df['ndvi_anomalia_pct'] * regimen_cosecha

    if nuevas_cols:
        return pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)
    return df.copy()



_ANOMALIAS_LLM_CACHE = None

def obtener_anomalias_desde_llm(fechas: list) -> dict:
    """
    Llama a Gemini para detectar dinámicamente anomalías macro y cisnes negros
    históricos en Argentina basados en la memoria cognitiva del modelo para las fechas dadas.
    Carga de forma segura API keys desde entorno o Streamlit Secrets.
    """
    global _ANOMALIAS_LLM_CACHE
    if _ANOMALIAS_LLM_CACHE is not None:
        return _ANOMALIAS_LLM_CACHE

    
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass

    
    if not api_key or api_key == "AIzaSyBzwUZZklyEAFde6GWoMel8o-WfrZobmLI":
        return {}

    try:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            _test_model = 'gemini-2.5-flash'
        except Exception:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                _test_model = 'gemini-1.5-flash'
            except Exception:
                model = genai.GenerativeModel('gemini-2.0-flash')
                _test_model = 'gemini-2.0-flash'
        
        fechas_unicas = sorted(list(set([str(f)[:10] for f in fechas])))
        fechas_str = ", ".join(fechas_unicas)
        
        prompt = f"""
        Analizá la historia macroeconómica y logística agrícola de Argentina de los últimos años. 
        Para las siguientes fechas semanales, identificá si alguna de ellas cae dentro de un período de 
        SHOCK EXCEPCIONAL o cisne negro histórico de la economía de Argentina.
        
        Devolveme ÚNICAMENTE un objeto JSON plano donde las claves sean las fechas en formato 'YYYY-MM-DD' 
        y los valores sean la etiqueta del evento exacto:
        - 'anomalia_devaluacion_2023' (para el shock cambiario de dic 2023 a principios de ene 2024)
        - 'anomalia_sequia_2022' (para el colapso productivo por sequía histórica de nov 2022 a apr 2023)
        - 'anomalia_logistica_parana' (para la bajante récord del Río Paraná de may 2021 a nov 2021)
        - 'anomalia_helada_tardia_2022' (para las heladas tardías extremas y atípicas de octubre 2022 que dañaron el trigo)
        - 'anomalia_golpe_calor_2023' (para la ola de calor histórica y estrés térmico del llenado de grano en nov-dic 2023)
        
        Si la fecha no corresponde a ningún shock histórico excepcional relevante para el trigo o los puertos, no la incluyas.
        No agregues explicaciones ni formato markdown (como ```json), devolvé solo el texto JSON plano.
        
        Fechas a evaluar: {fechas_str}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.startswith("json"):
                text = text[4:].strip()
                
        anomalias_dict = json.loads(text)
        _ANOMALIAS_LLM_CACHE = anomalias_dict
        print(f"  -> ¡Anomalías cognitivas extraídas dinámicamente con {_test_model} de forma exitosa!")
        return _ANOMALIAS_LLM_CACHE
    except Exception as e:
        print(f"  [LLM Warning] No se pudo procesar dinámicamente con Gemini: {e}. Activando fallback determinista de alta fidelidad.")
        return {}


def crear_anomalias_cognitivas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece el DataFrame con variables indicadoras de shocks macroeconómicos y anomalías logísticas,
    detectadas dinámicamente por Gemini Flash o autogeneradas mediante un fallback local vectorizado y veloz.
    """
    if 'fecha' not in df.columns:
        return df

    df = df.copy()

    
    eventos = [
        'anomalia_devaluacion_2023',
        'anomalia_sequia_2022',
        'anomalia_logistica_parana',
        'anomalia_helada_tardia_2022',
        'anomalia_golpe_calor_2023'
    ]
    for ev in eventos:
        df[ev] = 0.0

    
    fechas = pd.to_datetime(df['fecha'])
    llm_anomalias = obtener_anomalias_desde_llm(fechas)

    if llm_anomalias:
        
        fecha_str = fechas.dt.strftime('%Y-%m-%d')
        for ev in eventos:
            mask = fecha_str.map(llm_anomalias) == ev
            df.loc[mask, ev] = 1.0
    else:
        
        f = pd.to_datetime(df['fecha'])
        df.loc[(f >= '2021-05-01') & (f <= '2021-11-30'), 'anomalia_logistica_parana'] = 1.0
        df.loc[(f >= '2022-11-01') & (f <= '2023-04-30'), 'anomalia_sequia_2022'] = 1.0
        df.loc[(f >= '2023-12-10') & (f <= '2024-01-05'), 'anomalia_devaluacion_2023'] = 1.0
        df.loc[(f >= '2022-10-01') & (f <= '2022-10-31'), 'anomalia_helada_tardia_2022'] = 1.0
        df.loc[(f >= '2023-11-15') & (f <= '2023-12-15'), 'anomalia_golpe_calor_2023'] = 1.0

    return df


def procesar_datos_bcp(df: pd.DataFrame, max_lag_semanas: int = 12, lag_todos_precios: bool = True) -> pd.DataFrame:
    """
    Procesamiento optimizado para el Challenge BCP.
    Firma mantenida por retrocompatibilidad.
    Elimina ffill().fillna(0) generalizado para evitar lags inválidos a 0.
    Aplica ffill(limit=4) y bfill() generalizado al dataframe final.
    """
    
    patrones_sinteticos = [
        '_lag_', '_rolling_', 'fob_premium', 'fas_discount', 'fase_campaña', 'retorno', 'regimen', '_x_', 
        'anomalia_devaluacion', 'anomalia_sequia', 'anomalia_logistica_parana', 'anomalia_helada', 'anomalia_golpe_calor',
        'ratio_trigo_', 'nivel_parana_anomalia', 'gtrends_venta_momentum', 'cot_mm_net_change_1w'
    ]
    cols_a_mantener = [
        c for c in df.columns 
        if not any(p in c for p in patrones_sinteticos)
        and not c.endswith('_cos')
        and not c.endswith('_sin')
    ]
    df_proc = df[cols_a_mantener].copy()

    
    if 'fecha' in df_proc.columns:
        df_proc['fecha'] = pd.to_datetime(df_proc['fecha'])
        df_proc = df_proc.sort_values('fecha').reset_index(drop=True)

    
    df_proc = crear_features_estacionales(df_proc)

    
    df_proc = crear_features_ratio(df_proc)

    
    df_proc = crear_lags_optimizados(df_proc)

    
    df_proc = crear_rolling_stats(df_proc)

    
    df_proc = crear_anomalias_cognitivas(df_proc)

    
    df_proc = crear_features_regimen(df_proc)

    
    if 'fase_enso' in df_proc.columns:
        df_proc['fase_enso_Neutral'] = (df_proc['fase_enso'] == 'Neutral').astype(int)
        df_proc['fase_enso_Niño'] = (df_proc['fase_enso'] == 'Niño').astype(int)
        df_proc['fase_enso_Niña'] = (df_proc['fase_enso'] == 'Niña').astype(int)
        df_proc = df_proc.drop(columns=['fase_enso'])

    
    
    
    
    cols_sinteticas = [c for c in df_proc.columns if '_lag_' in c or '_rolling_' in c or 'retorno' in c]
    if cols_sinteticas:
        df_proc[cols_sinteticas] = df_proc[cols_sinteticas].ffill(limit=4)
    
    df_proc = df_proc.ffill(limit=4).bfill(limit=4).reset_index(drop=True)

    return df_proc
