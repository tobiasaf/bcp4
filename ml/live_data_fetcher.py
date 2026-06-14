import pandas as pd
import numpy as np
import os
import requests
import time
import json
from datetime import datetime, timedelta
import concurrent.futures
import yfinance as yf
from pytrends.request import TrendReq
import cot_reports as cot

# Configuración y Constantes
CACHE_DIR = "data"
PYTRENDS_CACHE_FILE = os.path.join(CACHE_DIR, "pytrends_cache.json")
COT_CACHE_FILE = os.path.join(CACHE_DIR, "cot_cache.json")

# Asegurar que el directorio de datos existe
os.makedirs(CACHE_DIR, exist_ok=True)

def get_usda_api_key():
    """Obtiene la API key de USDA desde los secretos de Streamlit o variables de entorno."""
    try:
        import streamlit as st
        if "usda" in st.secrets and "api_key" in st.secrets["usda"]:
            return st.secrets["usda"]["api_key"]
    except:
        pass
    return os.environ.get("USDA_API_KEY", "")

# ══════════════════════════════════════════════════
# TIER 1: APIs públicas y gratuitas sin credenciales
# ══════════════════════════════════════════════════

def fetch_dolar_live(timeout=5):
    """
    Obtiene las cotizaciones del dólar (Oficial, Blue, MEP, CCL) y calcula las brechas.
    Fuente: dolarapi.com
    """
    try:
        r = requests.get("https://dolarapi.com/v1/dolares", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            # Mapear respuesta
            valores = {}
            for item in data:
                casa = item.get("casa", "").lower()
                venta = float(item.get("venta", 0.0))
                if casa == "oficial":
                    valores["precio_oficial_usd"] = venta
                elif casa == "blue":
                    valores["precio_blue_usd"] = venta
                elif casa == "mep":
                    valores["precio_mep_usd"] = venta
                elif casa == "contadoconliqui":
                    valores["precio_ccl_usd"] = venta

            oficial = valores.get("precio_oficial_usd", 1.0)
            blue = valores.get("precio_blue_usd", oficial)
            ccl = valores.get("precio_ccl_usd", oficial)

            valores["brecha_blue_pct"] = ((blue - oficial) / oficial) * 100.0
            valores["brecha_ccl_pct"] = ((ccl - oficial) / oficial) * 100.0
            return valores, True
    except Exception as e:
        print(f"[Warning] Error obteniendo cotizaciones de dólar: {e}")
    
    # Fallback si falla el endpoint unificado, intentando individuales
    try:
        r_oficial = requests.get("https://dolarapi.com/v1/dolares/oficial", timeout=2)
        r_ccl = requests.get("https://dolarapi.com/v1/dolares/contadoconliqui", timeout=2)
        if r_oficial.status_code == 200 and r_ccl.status_code == 200:
            val_of = float(r_oficial.json()['venta'])
            val_ccl = float(r_ccl.json()['venta'])
            return {
                "precio_oficial_usd": val_of,
                "precio_blue_usd": val_ccl * 0.98,  # Proxy si falla blue
                "precio_mep_usd": val_ccl * 0.95,
                "precio_ccl_usd": val_ccl,
                "brecha_blue_pct": ((val_ccl * 0.98 - val_of) / val_of) * 100.0,
                "brecha_ccl_pct": ((val_ccl - val_of) / val_of) * 100.0
            }, True
    except:
        pass
        
    return {}, False

def fetch_riesgo_pais_live(timeout=5):
    """
    Obtiene el último valor del Riesgo País de Argentina (EMBI+).
    Fuente: ArgentinaDatos API
    """
    try:
        r = requests.get("https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo", timeout=timeout)
        if r.status_code == 200:
            val = float(r.json().get("valor", 0.0))
            return {"riesgo_pais_embi": val}, True
    except Exception as e:
        print(f"[Warning] Error obteniendo Riesgo País: {e}")
    return {}, False

def fetch_bcra_live(timeout=5):
    """
    Obtiene la Tasa de Política Monetaria del BCRA (ex LELIQ / Pases).
    Fuente: API pública BCRA principales variables.
    """
    # Usar verify=False porque a menudo el sitio del BCRA tiene problemas de certificados SSL
    try:
        r = requests.get("https://api.bcra.gob.ar/estadisticas/v1.0/principalesvariables", timeout=timeout, verify=False)
        if r.status_code == 200:
            data = r.json().get("results", [])
            for item in data:
                # ID 29 es Tasa de Política Monetaria o similar (también ID 34 plazos fijos)
                if item.get("idVariable") in [29, 34]:
                    val = float(item.get("valor", 0.0))
                    return {"tasa_politica_pct": val}, True
    except Exception as e:
        print(f"[Warning] Error obteniendo tasa BCRA: {e}")
    return {}, False

def fetch_cbot_live(timeout=8):
    """
    Obtiene cotizaciones de Chicago (trigo, maíz, soja) y variables macro globales.
    Fuente: yfinance
    """
    try:
        # ZW=F (Wheat), ZC=F (Corn), ZS=F (Soy), CL=F (WTI), DX-Y.NYB (DXY)
        tickers = {
            "precio_chicago_usd": "ZW=F",
            "cbot_maiz_usd": "ZC=F",
            "cbot_soja_usd": "ZS=F",
            "petroleo_wti_usd": "CL=F",
            "dxy_index": "DX-Y.NYB"
        }
        res = {}
        # Descargamos los últimos 5 días para asegurar cotización en fin de semana / feriados
        df = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)
        if not df.empty:
            # Mapear cada ticker a su valor de cierre más reciente no nulo
            for name, ticker in tickers.items():
                col_name = ("Close", ticker)
                if col_name in df.columns:
                    col_series = df[col_name].dropna()
                    if not col_series.empty:
                        # yfinance cotiza trigo en centavos de dólar por bushel. 
                        # Para ZW=F, ZC=F, ZS=F convertimos a USD por tonelada métrica
                        # Trigo: centavos/bushel * 0.367437 = USD/tn
                        # Maíz: centavos/bushel * 0.3936825 = USD/tn (aprox 0.39378)
                        # Soja: centavos/bushel * 0.367437 = USD/tn
                        val = float(col_series.iloc[-1])
                        if name == "precio_chicago_usd":
                            res[name] = val * 0.367437
                        elif name == "cbot_maiz_usd":
                            res[name] = val * 0.39368
                        elif name == "cbot_soja_usd":
                            res[name] = val * 0.367437
                        else:
                            res[name] = val
            
            # Verificar si logramos obtener al menos el trigo
            if "precio_chicago_usd" in res:
                return res, True
    except Exception as e:
        print(f"[Warning] Error obteniendo cotizaciones de yfinance: {e}")
    return {}, False

def fetch_fob_magyp_live(timeout=5):
    """
    Obtiene el precio FOB del trigo oficial publicado hoy por la Secretaría de Agricultura.
    Fuente: SIO Granos monitor precios FOB.
    """
    # Hacemos una búsqueda hacia atrás de hasta 5 días para encontrar el reporte más reciente
    for i in range(5):
        fecha_str = (datetime.now() - timedelta(days=i)).strftime("%d/%m/%Y")
        url = f"http://monitorsiogranos.magyp.gob.ar/ws/ssma/precios_fob.php?Fecha={fecha_str}"
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    # Buscar el trigo ("Trigo Pan", "Trigo")
                    for item in data:
                        producto = item.get("producto", "").lower()
                        if "trigo" in producto and "pan" in producto:
                            val = float(item.get("precio", 0.0))
                            if val > 0:
                                return {"precio_fob_usd": val}, True
        except:
            pass
    return {}, False

def fetch_enso_live(timeout=5):
    """
    Obtiene el índice ONI de la NOAA más reciente para deducir la fase ENSO (El Niño / La Niña).
    Fuente: NOAA CPC sstoi.indices (archivo de texto plano más ligero y confiable)
    """
    try:
        r = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices", timeout=timeout)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            # Filtrar líneas vacías y encabezados
            data_lines = [l.strip() for l in lines if l.strip() and not l.startswith("YR")]
            if data_lines:
                last_line = data_lines[-1]
                parts = last_line.split()
                # sstoi.indices tiene las columnas: YR, MON, NINO1+2, ANOM, NINO3, ANOM, NINO4, ANOM, NINO3.4, ANOM
                # El 10mo elemento (index 9) es el desvío/anomalía de Niño 3.4
                if len(parts) >= 10:
                    nino34_anom = float(parts[9])
                    
                    # Clasificación ENSO
                    if nino34_anom >= 0.5:
                        fase = "EL NIÑO"
                    elif nino34_anom <= -0.5:
                        fase = "LA NIÑA"
                    else:
                        fase = "NEUTRAL"
                        
                    return {
                        "nino34_anomalia": nino34_anom,
                        "fase_enso": fase
                    }, True
    except Exception as e:
        print(f"[Warning] Error obteniendo ENSO de NOAA: {e}")
    return {}, False

def fetch_parana_live(timeout=5):
    """
    Obtiene la altura hidrométrica del Río Paraná en Rosario.
    Fuente: Alerta Cuenca del Plata API (INA)
    """
    # Series ID 5893 representa la altura del río en la estación Rosario (puntual)
    url = "https://alerta.ina.gob.ar/pub/api/getObservaciones?tipo=puntual&series_id=5893"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                # El último elemento es la observación más fresca
                last_obs = data[-1]
                val = float(last_obs.get("valor", 0.0))
                return {"nivel_parana_rosario_m": val}, True
    except Exception as e:
        print(f"[Warning] Error obteniendo altura Paraná (Rosario) de INA: {e}")
    return {}, False

def fetch_cftc_cot_live():
    """
    Obtiene la posición neta de los especuladores (Managed Money) y comerciales en trigo CBOT.
    Fuente: CFTC disaggregated report usando la librería cot-reports.
    Con caché semanal en disco para optimizar performance y red.
    """
    # 1. Intentar cargar desde el caché
    if os.path.exists(COT_CACHE_FILE):
        try:
            with open(COT_CACHE_FILE, "r") as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached["timestamp"])
            # Si tiene menos de 7 días, usarlo directamente
            if datetime.now() - cached_time < timedelta(days=7):
                return cached["data"], True
        except Exception as e:
            print(f"[Warning] No se pudo leer el caché de COT: {e}")

    # 2. Si no hay caché o expiró, descargar
    year = datetime.now().year
    try:
        # disaggregated_fut es más liviano que disaggregated_futopt y tiene los contratos puros de futuros
        df = cot.cot_year(year, cot_report_type='disaggregated_fut')
        
        # Filtrar el trigo de Chicago (Soft Red Winter)
        df_trigo = df[
            df['Market_and_Exchange_Names'].str.contains('WHEAT', case=False, na=False) &
            df['Market_and_Exchange_Names'].str.contains('CHICAGO BOARD OF TRADE', case=False, na=False)
        ]
        
        if df_trigo.empty:
            # Fallback a búsqueda más amplia de trigo si cambia el nombre
            df_trigo = df[df['Market_and_Exchange_Names'].str.contains('WHEAT', case=False, na=False)]
            
        if not df_trigo.empty:
            # Ordenar por fecha y tomar el último
            # Las columnas de fecha comunes en cot-reports son 'Report_Date_as_YYYY-MM-DD'
            date_col = 'Report_Date_as_YYYY-MM-DD'
            if date_col not in df_trigo.columns:
                date_cols = [c for c in df_trigo.columns if 'date' in c.lower()]
                date_col = date_cols[0] if date_cols else df_trigo.columns[0]
                
            df_trigo = df_trigo.sort_values(by=date_col)
            latest = df_trigo.iloc[-1]
            
            # Extraer Managed Money Long & Short
            mm_long = float(latest.get('M_Money_Positions_Long_All', latest.get('M_Money_Positions_Long_Fut', 0.0)))
            mm_short = float(latest.get('M_Money_Positions_Short_All', latest.get('M_Money_Positions_Short_Fut', 0.0)))
            
            # Extraer Comerciales (Productores + Swap Dealers)
            comm_long = float(latest.get('Prod_Merc_Positions_Long_All', 0.0)) + float(latest.get('Swap_Positions_Long_All', 0.0))
            comm_short = float(latest.get('Prod_Merc_Positions_Short_All', 0.0)) + float(latest.get('Swap_Positions_Short_All', 0.0))
            
            cot_data = {
                "cot_managed_money_net": mm_long - mm_short,
                "cot_commercial_net": comm_long - comm_short
            }
            
            # Guardar en caché
            try:
                with open(COT_CACHE_FILE, "w") as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "data": cot_data}, f)
            except Exception as cache_err:
                print(f"[Warning] Error guardando caché de COT: {cache_err}")
                
            return cot_data, True
            
    except Exception as e:
        print(f"[Warning] Error descargando reporte COT de la CFTC: {e}")
        
    # 3. Degradación amigable: Cargar caché viejo si existía aunque haya expirado
    if os.path.exists(COT_CACHE_FILE):
        try:
            with open(COT_CACHE_FILE, "r") as f:
                cached = json.load(f)
            print("[Info] Cargando datos COT desde el caché expirado debido a falla de descarga.")
            return cached["data"], True
        except:
            pass
            
    return {}, False

def fetch_gtrends_live():
    """
    Obtiene la tendencia de búsqueda en Google Trends para "vender trigo" y "dólar" en Argentina.
    Limita a 1 consulta semanal y guarda en un caché persistente local para evitar bloqueos por IP.
    """
    # 1. Intentar cargar desde el caché
    if os.path.exists(PYTRENDS_CACHE_FILE):
        try:
            with open(PYTRENDS_CACHE_FILE, "r") as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached["timestamp"])
            # Si tiene menos de 7 días, usarlo directamente (Caché Semanal Estricto)
            if datetime.now() - cached_time < timedelta(days=7):
                return cached["data"], True
        except Exception as e:
            print(f"[Warning] No se pudo leer el caché de PyTrends: {e}")

    # 2. Si no hay caché o expiró, intentar hacer la llamada web
    try:
        # Configurar pytrends con timeout e IP rotada si fuese necesario (hl=español, tz=Argentina)
        pytrends = TrendReq(hl='es-419', tz=180, timeout=10)
        kw_list = ["vender trigo", "dólar"]
        pytrends.build_payload(kw_list, cat=0, timeframe='today 3-m', geo='AR', gprop='')
        df = pytrends.interest_over_time()
        
        if not df.empty:
            # Obtener el último valor válido (normalmente representa la última semana)
            # Evitamos filas que puedan tener flags parciales
            latest_row = df.dropna().iloc[-1]
            
            trends_data = {
                "gtrends_vender_trigo": float(latest_row.get("vender trigo", 50.0)),
                "gtrends_dolar": float(latest_row.get("dólar", 50.0))
            }
            
            # Guardar en caché
            try:
                with open(PYTRENDS_CACHE_FILE, "w") as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "data": trends_data}, f)
            except Exception as cache_err:
                print(f"[Warning] Error guardando caché de PyTrends: {cache_err}")
                
            return trends_data, True
            
    except Exception as e:
        print(f"[Warning] Error consultando Google Trends (bloqueo o timeout): {e}")

    # 3. Degradación amigable: Si falla Google (429), usar caché viejo sin importar antigüedad
    if os.path.exists(PYTRENDS_CACHE_FILE):
        try:
            with open(PYTRENDS_CACHE_FILE, "r") as f:
                cached = json.load(f)
            print("[Info] Cargando datos de PyTrends desde caché viejo por bloqueo o falla de red.")
            return cached["data"], True
        except:
            pass

    return {}, False

# ══════════════════════════════════════════════════
# TIER 2: APIs con credenciales gratuitas
# ══════════════════════════════════════════════════

def fetch_usda_wasde_live(timeout=5):
    """
    Obtiene el balance de oferta y demanda mundial de trigo de la USDA FAS PSD API.
    Calcula la relación stock-consumo (Stocks-to-Use) global y las exportaciones de Argentina.
    """
    api_key = get_usda_api_key()
    if not api_key:
        print("[Warning] No hay API Key del USDA cargada en secrets.toml. WASDE omitido.")
        return {}, False
        
    headers = {"X-Api-Key": api_key}
    wheat_code = "0410000"
    
    # Intentamos obtener la campaña actual (ej. 2025/2026, si no funciona probamos campaña previa 2024/2025)
    current_year = datetime.now().year
    years_to_try = [current_year, current_year - 1]
    
    for yr in years_to_try:
        url = f"https://api.fas.usda.gov/api/psd/commodity/{wheat_code}/country/all/year/{yr}"
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if len(data) > 0:
                    # Atributos:
                    # 125: Domestic Consumption
                    # 176: Ending Stocks
                    # 88: Exports
                    
                    # Consumo y Stocks Globales (sumamos todas las regiones)
                    total_consumption = sum(float(x.get("value", 0.0)) for x in data if x.get("attributeId") == 125)
                    total_stocks = sum(float(x.get("value", 0.0)) for x in data if x.get("attributeId") == 176)
                    
                    # Exportaciones estimadas para Argentina
                    arg_exports = sum(float(x.get("value", 0.0)) for x in data if x.get("countryCode") == "AR" and x.get("attributeId") == 88)
                    
                    # Evitar división por cero
                    stocks_to_use = total_stocks / total_consumption if total_consumption > 0 else 0.30
                    
                    return {
                        "wasde_stocks_to_use": stocks_to_use,
                        "wasde_arg_export_mt": arg_exports
                    }, True
        except Exception as e:
            print(f"[Warning] Error consultando USDA para el año {yr}: {e}")
            
    return {}, False

# ══════════════════════════════════════════════════
# TIER 3: APIs con credenciales opcionales
# ══════════════════════════════════════════════════

def fetch_rofex_live(timeout=5):
    """
    Obtiene datos de futuros en vivo de MatbaRofex usando la librería pyRofex.
    Esta función es 100% opcional y requiere credenciales activas del broker en secrets.toml.
    Si fallan o están vacías, se degrada a proxies de yfinance/Chicago.
    """
    try:
        import pyRofex
        import streamlit as st
        
        # Verificar credenciales en secretos
        if ("rofex" in st.secrets and 
            st.secrets["rofex"]["user"] and 
            st.secrets["rofex"]["password"] and 
            st.secrets["rofex"]["account"]):
            
            user = st.secrets["rofex"]["user"]
            pw = st.secrets["rofex"]["password"]
            acc = st.secrets["rofex"]["account"]
            
            # Inicializar entorno (se asume live, o sandbox según credencial)
            pyRofex.initialize(
                user=user,
                password=pw,
                account=acc,
                environment=pyRofex.Environment.LIVE
            )
            
            # Intentar obtener cotizaciones del trigo más cercano
            # Tradicionalmente: W/ENE26, W/MAY26, etc.
            # Haremos una consulta rápida de cotización del trigo del mes actual o cercano
            instrumento_trigo = "I.TRIGO"  # Índice oficial de trigo MatbaRofex
            market_data = pyRofex.get_market_data(
                ticker=instrumento_trigo,
                depth=1
            )
            
            if market_data and market_data.get("status") == "OK":
                bids = market_data["marketData"].get("BI", [])
                asks = market_data["marketData"].get("OF", [])
                price = None
                if bids:
                    price = float(bids[0]["price"])
                elif asks:
                    price = float(asks[0]["price"])
                    
                if price:
                    return {
                        "rofex_precio_usd": price,
                        "rofex_volumen": float(market_data["marketData"].get("volume", 1500.0)),
                        "rofex_interes_abierto": float(market_data["marketData"].get("openInterest", 25000.0))
                    }, True
    except Exception as e:
        print(f"[Warning] Error consultando pyRofex (degradando a fallbacks de Chicago): {e}")
        
    return {}, False

# ══════════════════════════════════════════════════
# ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════

def fetch_all_live(timeout=5):
    """
    Orquestador que ejecuta todas las APIs en paralelo usando hilos de ejecución.
    Proporciona timeouts independientes y soporte completo de fallbacks.
    
    Retorna un diccionario unificado con los datos en tiempo real de BCP Studio y
    un segundo diccionario con los flags de estado de cada consulta (API Status).
    """
    print(f"📡 Iniciando orquestación paralela de APIs en vivo ({datetime.now().strftime('%H:%M:%S')})...")
    
    # Definición de las tareas a ejecutar
    # (función, clave de estado, timeout opcional)
    tasks = {
        "dolar": (fetch_dolar_live, "Dólar API"),
        "riesgo_pais": (fetch_riesgo_pais_live, "ArgentinaDatos EMBI"),
        "bcra": (fetch_bcra_live, "BCRA Monetarias"),
        "cbot": (fetch_cbot_live, "yfinance CBOT"),
        "fob_magyp": (fetch_fob_magyp_live, "SIO Granos FOB"),
        "enso": (fetch_enso_live, "NOAA CPC ENSO"),
        "parana": (fetch_parana_live, "INA Paraná"),
        "cot": (fetch_cftc_cot_live, "CFTC COT Report"),
        "gtrends": (fetch_gtrends_live, "Google Trends"),
        "usda": (fetch_usda_wasde_live, "USDA WASDE"),
        "rofex": (fetch_rofex_live, "MatbaRofex")
    }
    
    unified_data = {}
    status_flags = {}
    
    # Usar ThreadPoolExecutor para paralelizar
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        # Encolar las ejecuciones
        future_to_key = {}
        for key, (func, label) in tasks.items():
            # Algunos métodos no aceptan timeout
            if key in ["cot", "gtrends"]:
                future = executor.submit(func)
            else:
                future = executor.submit(func, timeout=timeout)
            future_to_key[future] = (key, label)
            
        # Recolectar resultados a medida que finalizan
        for future in concurrent.futures.as_completed(future_to_key):
            key, label = future_to_key[future]
            try:
                res_dict, success = future.result()
                status_flags[label] = "✅ Online" if success else "⚠️ Desconectado"
                if success and res_dict:
                    unified_data.update(res_dict)
            except Exception as exc:
                print(f"[Error] La tarea '{label}' lanzó una excepción: {exc}")
                status_flags[label] = "❌ Falla crítica"
                
    # ══════════════════════════════════════════════════
    # DEGRADACIÓN Y FALLBACKS DE CÁLCULO
    # ══════════════════════════════════════════════════
    
    # A. Fallback si MatbaRofex está apagado
    if "rofex_precio_usd" not in unified_data:
        # Usamos Chicago como proxy: Chicago - 20 USD/ton base histórica
        cbot_price = unified_data.get("precio_chicago_usd", 230.0)
        unified_data["rofex_precio_usd"] = cbot_price - 20.0
        unified_data["rofex_volumen"] = 1200.0  # Media de volumen histórica
        unified_data["rofex_interes_abierto"] = 22000.0  # Open Interest histórico promedio
        status_flags["MatbaRofex"] = "ℹ️ Proxy (Chicago - $20)"
        
    # B. Fallback para USDA WASDE
    if "wasde_stocks_to_use" not in unified_data:
        unified_data["wasde_stocks_to_use"] = 0.325  # Relación histórica promedio
        unified_data["wasde_arg_export_mt"] = 11500.0 # Exportación promedio de trigo en miles de toneladas
        if "USDA WASDE" not in status_flags or status_flags["USDA WASDE"] != "✅ Online":
            status_flags["USDA WASDE"] = "ℹ️ Fallback Histórico"
            
    # C. Fallback para Paraná
    if "nivel_parana_rosario_m" not in unified_data:
        unified_data["nivel_parana_rosario_m"] = 3.5  # Altura promedio hidrométrica en Rosario
        status_flags["INA Paraná"] = "ℹ️ Fallback Histórico"
        
    # D. Fallback para ENSO
    if "nino34_anomalia" not in unified_data:
        unified_data["nino34_anomalia"] = 0.0
        unified_data["fase_enso"] = "NEUTRAL"
        status_flags["NOAA CPC ENSO"] = "ℹ️ Fallback Histórico"

    # E. Fallback para Google Trends
    if "gtrends_vender_trigo" not in unified_data:
        unified_data["gtrends_vender_trigo"] = 50.0
        unified_data["gtrends_dolar"] = 50.0
        status_flags["Google Trends"] = "ℹ️ Fallback Histórico"
        
    # F. Fallback para CFTC COT
    if "cot_managed_money_net" not in unified_data:
        unified_data["cot_managed_money_net"] = -25000.0  # Históricamente los especuladores son net short
        unified_data["cot_commercial_net"] = 15000.0
        status_flags["CFTC COT Report"] = "ℹ️ Fallback Histórico"
        
    # G. Fallback para Riesgo País y Tasa BCRA
    if "riesgo_pais_embi" not in unified_data:
        unified_data["riesgo_pais_embi"] = 1000.0
    if "tasa_politica_pct" not in unified_data:
        unified_data["tasa_politica_pct"] = 40.0
        
    print(f"📡 Orquestación paralela finalizada con éxito. {len(unified_data)} variables en vivo consolidadas.")
    return unified_data, status_flags

if __name__ == "__main__":
    # Test rápido de ejecución
    res, status = fetch_all_live(timeout=5)
    print("\n--- RESULTADO INTEGRADO ---")
    for k, v in res.items():
        print(f"{k}: {v}")
    print("\n--- ESTADO DE APIS ---")
    for k, v in status.items():
        print(f"{k}: {v}")
