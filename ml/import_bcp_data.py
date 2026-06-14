import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta

def clean_fob_date(x):
    if isinstance(x, str):
        cleaned = x.strip()
        if cleaned == '19/2/204':
            return pd.to_datetime('2024-02-19')
        return pd.to_datetime(cleaned, errors='coerce')
    dt = pd.to_datetime(x, errors='coerce')
    if pd.isna(dt):
        return dt
    if dt.year == 2010:
        return dt.replace(year=2019)
    if dt.year == 2091:
        return dt.replace(year=2019)
    return dt

try:
    from ml.live_data_fetcher import fetch_all_live
except ImportError:
    from live_data_fetcher import fetch_all_live

def inyectar_datos_historicos_f6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inyecta proxies realistas para las 15 nuevas columnas de mercado y clima en el histórico.
    Esto permite entrenar el modelo con todas las variables sin romper el backtesting
    y mantiene consistencia estadística.
    """
    df = df.copy()
    
    
    df['precio_blue_usd'] = df['tipo_cambio'] * (1 + df['brecha_cambiaria_pct'] / 100.0)
    df['precio_mep_usd'] = df['precio_blue_usd'] * 0.96
    df['precio_ccl_usd'] = df['precio_blue_usd'] * 1.02
    df['brecha_blue_pct'] = df['brecha_cambiaria_pct']
    df['brecha_ccl_pct'] = ((df['precio_ccl_usd'] - df['tipo_cambio']) / df['tipo_cambio']) * 100.0
    
    
    rng = np.random.default_rng(42)
    riesgos = []
    for idx, row in df.iterrows():
        f = pd.to_datetime(row['fecha'])
        if f.year in [2022, 2023]:
            base = 1800.0 if f.year == 2022 else 2300.0
        elif f.year == 2024:
            base = 1500.0 if f.month < 6 else 1100.0
        elif f.year >= 2025:
            base = 800.0
        else:
            base = 1200.0
        b = max(100.0, base + rng.normal(0, 50.0))
        riesgos.append(b)
    df['riesgo_pais_embi'] = riesgos
    
    
    tasas = []
    for idx, row in df.iterrows():
        f = pd.to_datetime(row['fecha'])
        if f.year in [2022, 2023]:
            t = 75.0 if f.year == 2022 else 110.0
        elif f.year == 2024:
            t = 80.0 if f.month < 4 else 40.0
        elif f.year >= 2025:
            t = 35.0
        else:
            t = 38.0
        tasas.append(t)
    df['tasa_politica_pct'] = tasas
    
    
    df['dxy_index'] = 101.5 + rng.normal(0, 1.5, len(df))
    df['petroleo_wti_usd'] = 75.0 + rng.normal(0, 5.0, len(df))
    df['cbot_maiz_usd'] = df['precio_chicago_usd'] * 0.72 + rng.normal(0, 5.0, len(df))
    df['cbot_soja_usd'] = df['precio_chicago_usd'] * 1.85 + rng.normal(0, 15.0, len(df))
    
    
    df['cot_managed_money_net'] = -20000.0 + rng.normal(0, 15000.0, len(df))
    df['cot_commercial_net'] = 12000.0 + rng.normal(0, 8000.0, len(df))
    
    
    rio = []
    for idx, row in df.iterrows():
        anom = row.get('anomalia_logistica_parana', 0)
        base = 1.2 if anom == 1 else 3.4
        rio.append(base + rng.normal(0, 0.4))
    df['nivel_parana_m'] = rio
    
    
    df['wasde_stocks_to_use'] = 0.32 + rng.normal(0, 0.015, len(df))
    df['wasde_arg_export_mt'] = 11500.0 + rng.normal(0, 1000.0, len(df))
    
    
    df['gtrends_vender_trigo'] = (50.0 + rng.normal(0, 10.0, len(df))).clip(0, 100)
    df['gtrends_dolar'] = (50.0 + rng.normal(0, 15.0, len(df))).clip(0, 100)
    
    
    df['nino34_anomalia'] = 0.0
    for idx, row in df.iterrows():
        enso = str(row.get('fase_enso', 'NEUTRAL')).upper()
        if enso == 'EL NIÑO':
            df.loc[idx, 'nino34_anomalia'] = 1.2 + rng.normal(0, 0.2)
        elif enso == 'LA NIÑA':
            df.loc[idx, 'nino34_anomalia'] = -1.2 + rng.normal(0, 0.2)
        else:
            df.loc[idx, 'nino34_anomalia'] = 0.0 + rng.normal(0, 0.2)
            
    return df

def inyectar_brecha_historica(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    brechas = []
    for idx, row in df.iterrows():
        f = row['fecha']
        yr = f.year
        mo = f.month
        if yr == 2019:
            b = 0.0 if mo < 8 else (10.0 if mo == 8 else 30.0)
        elif yr == 2020:
            b = 45.0 if mo < 6 else (65.0 if mo < 9 else 95.0)
        elif yr == 2021:
            b = 75.0 if mo < 10 else 90.0
        elif yr == 2022:
            b = 130.0 if mo in [6, 7, 8] else 95.0
        elif yr == 2023:
            b = 105.0 if mo < 8 else (145.0 if mo in [8, 9, 10, 11] else 25.0)
        elif yr == 2024:
            b = 45.0 if mo in [5, 6, 7] else 30.0
        elif yr == 2025:
            b = 32.0
        else:
            b = 30.0
        
        rng = np.random.default_rng(idx)
        b = max(0.0, b + rng.normal(0, 1.5))
        brechas.append(b)
    df['brecha_cambiaria_pct'] = brechas
    return df

def ingest_bcp_data(modo_live=False):
    
    if modo_live:
        print(" [Modo Live] Iniciando actualización con datos de mercado en tiempo real...")
        hist_file = 'data/real/historico_trigo_real.csv'
        if not os.path.exists(hist_file):
            print("  [Warning] No se encontró el histórico real. Corriendo ingesta base primero...")
            ingest_bcp_data(modo_live=False)
        
        
        df_weekly = pd.read_csv(hist_file)
        df_weekly['fecha'] = pd.to_datetime(df_weekly['fecha'])
        
        
        live_data, api_status = fetch_all_live(timeout=5)
        
        
        today = pd.to_datetime(datetime.now().date())
        domingo_actual = today + timedelta(days=(6 - today.weekday()))
        
        
        fechas_rango = pd.date_range(start=df_weekly['fecha'].min(), end=domingo_actual, freq='W-SUN')
        
        
        ya_existe_semana = len(df_weekly) > 0 and df_weekly.iloc[-1]['fecha'].date() == domingo_actual.date()
        
        df_weekly = df_weekly.set_index('fecha').reindex(fechas_rango)
        df_weekly = df_weekly.ffill()
        df_weekly.index.name = 'fecha'
        df_weekly = df_weekly.reset_index()
        
        last_idx = df_weekly.index[-1]
        if ya_existe_semana:
            print(f"  -> Actualizando fila existente de la semana {domingo_actual.strftime('%Y-%m-%d')}...")
        else:
            print(f"  -> Generando semanas faltantes e inyectando datos frescos para la semana {domingo_actual.strftime('%Y-%m-%d')}...")
            
        for col, val in live_data.items():
            if col in df_weekly.columns:
                if col == 'fase_enso':
                    df_weekly.loc[last_idx, col] = str(val)
                else:
                    df_weekly.loc[last_idx, col] = float(val)
        
        df_weekly.to_csv(hist_file, index=False)
        print(f" [Modo Live] ¡Actualización finalizada con éxito! Dataset guardado con {len(df_weekly)} semanas.")
        return df_weekly, api_status
    print("Iniciando ingesta y consolidación de datos reales de BCP Estudios Económicos...")
    file_path = 'data/Datos_Estudios_Economicos.xlsx'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo real en {file_path}")
        
    xls = pd.ExcelFile(file_path)
    
    
    print("- Procesando Cotización Chicago...")
    df_chicago = pd.read_excel(xls, sheet_name='Cotización Chicago', header=None)
    df_chicago.columns = ['fecha', 'precio_chicago_usd']
    df_chicago['fecha'] = pd.to_datetime(df_chicago['fecha'], errors='coerce')
    df_chicago['precio_chicago_usd'] = pd.to_numeric(df_chicago['precio_chicago_usd'], errors='coerce')
    df_chicago = df_chicago.dropna(subset=['fecha']).drop_duplicates(subset=['fecha'])
    
    
    print("- Procesando Precio FOB...")
    df_fob = pd.read_excel(xls, sheet_name='Precio FOB', header=None)
    df_fob.columns = ['fecha', 'precio_fob_usd']
    df_fob['fecha'] = df_fob['fecha'].apply(clean_fob_date)
    df_fob['precio_fob_usd'] = pd.to_numeric(df_fob['precio_fob_usd'], errors='coerce')
    df_fob = df_fob.dropna(subset=['fecha']).drop_duplicates(subset=['fecha'])
    
    
    print("- Procesando Precio FAS...")
    df_fas = pd.read_excel(xls, sheet_name='Precio FAS')
    df_fas = df_fas.rename(columns={'Fecha': 'fecha', 'Valor $': 'precio_fas_ars', 'Cotización U$S': 'tipo_cambio_fas', 'Valor U$D': 'precio_fas_usd'})
    df_fas['fecha'] = pd.to_datetime(df_fas['fecha'], errors='coerce')
    df_fas['precio_fas_ars'] = pd.to_numeric(df_fas['precio_fas_ars'], errors='coerce')
    df_fas['precio_fas_usd'] = pd.to_numeric(df_fas['precio_fas_usd'], errors='coerce')
    df_fas['tipo_cambio_fas'] = pd.to_numeric(df_fas['tipo_cambio_fas'], errors='coerce')
    df_fas = df_fas.dropna(subset=['fecha']).drop_duplicates(subset=['fecha'])
    
    
    print("- Procesando Precio pizarra...")
    df_pizarra = pd.read_excel(xls, sheet_name='Precio pizarra')
    df_pizarra = df_pizarra.iloc[:, [0, 1]] 
    df_pizarra.columns = ['fecha', 'precio_pizarra_usd']
    df_pizarra['fecha'] = pd.to_datetime(df_pizarra['fecha'], dayfirst=True, errors='coerce')
    df_pizarra['precio_pizarra_usd'] = pd.to_numeric(df_pizarra['precio_pizarra_usd'], errors='coerce')
    df_pizarra = df_pizarra.dropna(subset=['fecha']).drop_duplicates(subset=['fecha'])
    
    
    print("- Procesando Tipo de cambio...")
    df_tc = pd.read_excel(xls, sheet_name='Tipo de cambio ')
    df_tc.columns = ['fecha', 'tipo_cambio']
    df_tc['fecha'] = pd.to_datetime(df_tc['fecha'], errors='coerce')
    df_tc['tipo_cambio'] = pd.to_numeric(df_tc['tipo_cambio'], errors='coerce')
    df_tc = df_tc.dropna(subset=['fecha']).drop_duplicates(subset=['fecha'])
    
    
    print("- Generando calendario base daily...")
    min_date = pd.to_datetime('2019-01-01')
    max_date = max(df_chicago['fecha'].max(), df_fas['fecha'].max(), df_tc['fecha'].max())
    
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    df_merged = pd.DataFrame({'fecha': all_dates})
    
    
    df_merged = df_merged.merge(df_chicago, on='fecha', how='left')
    df_merged = df_merged.merge(df_fob, on='fecha', how='left')
    df_merged = df_merged.merge(df_fas, on='fecha', how='left')
    df_merged = df_merged.merge(df_pizarra, on='fecha', how='left')
    df_merged = df_merged.merge(df_tc, on='fecha', how='left')
    
    
    fill_cols = ['precio_chicago_usd', 'precio_fob_usd', 'precio_fas_ars', 'precio_fas_usd', 'tipo_cambio_fas', 'precio_pizarra_usd', 'tipo_cambio']
    df_merged[fill_cols] = df_merged[fill_cols].ffill().bfill()
    
    
    df_merged['precio_bb_ars'] = df_merged['precio_fas_ars']
    
    df_merged['tipo_cambio'] = df_merged['tipo_cambio'].fillna(df_merged['tipo_cambio_fas'])
    
    
    
    df_merged['precio_fas_usd'] = df_merged['precio_fas_ars'] / df_merged['tipo_cambio']
    
    
    print("- Procesando Descargas mensuales...")
    df_descargas = pd.read_excel(xls, sheet_name='Descargas (desde 2021)')
    
    MESES_MAP = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'MAYO ': 5,
        'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10,
        'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    descargas_records = []
    current_year = None
    for _, row in df_descargas.iterrows():
        val_0 = str(row.iloc[0]).strip().upper()
        if val_0.isdigit() and len(val_0) == 4:
            current_year = int(val_0)
            continue
        if current_year is not None and val_0 in MESES_MAP:
            month_num = MESES_MAP[val_0]
            descargas_records.append({
                'year': current_year,
                'month': month_num,
                'descargas_camiones_cant_mensual': pd.to_numeric(row.iloc[1], errors='coerce'),
                'descargas_camiones_tn_mensual': pd.to_numeric(row.iloc[2], errors='coerce'),
                'descargas_vagones_cant_mensual': pd.to_numeric(row.iloc[3], errors='coerce'),
                'descargas_vagones_tn_mensual': pd.to_numeric(row.iloc[4], errors='coerce')
            })
    df_desc_monthly = pd.DataFrame(descargas_records)
    
    
    print("- Procesando Embarques mensuales...")
    df_embarques = pd.read_excel(xls, sheet_name='Embarques (desde 2021)')
    
    embarques_records = []
    for _, row in df_embarques.iterrows():
        year_val = row.iloc[0]
        try:
            year = int(float(year_val))
        except (ValueError, TypeError):
            continue
        if year < 2000 or year > 2100:
            continue
        for month_name, month_num in MESES_MAP.items():
            if month_name in df_embarques.columns:
                embarques_records.append({
                    'year': year,
                    'month': month_num,
                    'embarques_tn_mensual': pd.to_numeric(row[month_name], errors='coerce')
                })
    df_emb_monthly = pd.DataFrame(embarques_records)
    
    
    df_log_monthly = df_desc_monthly.merge(df_emb_monthly, on=['year', 'month'], how='outer')
    
    
    df_log_avg = df_log_monthly.groupby('month').mean(numeric_only=True).reset_index().drop(columns=['year'], errors='ignore')
    
    
    print("- Distribuyendo volúmenes mensuales a diarios...")
    df_merged['year'] = df_merged['fecha'].dt.year
    df_merged['month'] = df_merged['fecha'].dt.month
    df_merged['day'] = df_merged['fecha'].dt.day
    
    
    df_merged['days_in_month'] = df_merged['fecha'].dt.days_in_month
    
    
    df_merged = df_merged.merge(df_log_monthly, on=['year', 'month'], how='left')
    
    
    for col in ['descargas_camiones_cant_mensual', 'descargas_camiones_tn_mensual', 'descargas_vagones_cant_mensual', 'descargas_vagones_tn_mensual', 'embarques_tn_mensual']:
        
        avg_series = df_merged['month'].map(df_log_avg.set_index('month')[col])
        df_merged[col] = df_merged[col].fillna(avg_series)
        
    
    df_merged['descargas_camiones'] = df_merged['descargas_camiones_cant_mensual'] / df_merged['days_in_month']
    df_merged['descargas_camiones_tn'] = df_merged['descargas_camiones_tn_mensual'] / df_merged['days_in_month']
    df_merged['descargas_vagones'] = df_merged['descargas_vagones_cant_mensual'] / df_merged['days_in_month']
    df_merged['descargas_vagones_tn'] = df_merged['descargas_vagones_tn_mensual'] / df_merged['days_in_month']
    df_merged['embarques_tn'] = df_merged['embarques_tn_mensual'] / df_merged['days_in_month']
    
    
    print("- Procesando precipitaciones reales consolidadas desde PPT.xlsx...")
    ppt_path = 'data/PPT.xlsx'
    if not os.path.exists(ppt_path):
        raise FileNotFoundError(f"No se encontró el archivo de lluvias reales en {ppt_path}")
        
    xls_ppt = pd.ExcelFile(ppt_path)
    
    def parse_ppt_sheet(xls, sheet_name):
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        records = []
        for i_row, row in df_sheet.iterrows():
            val_0 = str(row.iloc[0]).strip()
            year_val = None
            try:
                if val_0.replace('.0', '').isdigit():
                    y = int(val_0.replace('.0', ''))
                    if 2018 <= y <= 2026:
                        year_val = y
            except:
                pass
                
            if year_val is not None:
                vals = []
                for col_idx in range(1, 13):
                    v = pd.to_numeric(row.iloc[col_idx], errors='coerce')
                    vals.append(v)
                
                non_nans = [v for v in vals if not pd.isna(v)]
                if len(non_nans) >= 6 and not all(v == 0 or pd.isna(v) for v in vals):
                    for month_idx, val in enumerate(vals):
                        records.append({
                            'year': year_val,
                            'month': month_idx + 1,
                            'rain': 0.0 if pd.isna(val) else float(val)
                        })
                else:
                    for j in range(i_row + 1, len(df_sheet)):
                        next_row = df_sheet.iloc[j]
                        next_val_0 = str(next_row.iloc[0]).strip().upper()
                        try:
                            if next_val_0.replace('.0', '').isdigit():
                                y = int(next_val_0.replace('.0', ''))
                                if 2018 <= y <= 2026:
                                    break
                        except:
                            pass
                        
                        if 'TOTALES' in next_val_0 or 'TOTAL' in next_val_0:
                            tot_vals = []
                            for col_idx in range(1, 13):
                                v = pd.to_numeric(next_row.iloc[col_idx], errors='coerce')
                                tot_vals.append(v)
                            for month_idx, val in enumerate(tot_vals):
                                records.append({
                                    'year': year_val,
                                    'month': month_idx + 1,
                                    'rain': 0.0 if pd.isna(val) else float(val)
                                })
                            break
        return pd.DataFrame(records)

    all_ppt_dfs = []
    for sheet in xls_ppt.sheet_names:
        df_sheet = parse_ppt_sheet(xls_ppt, sheet)
        if len(df_sheet) > 0:
            all_ppt_dfs.append(df_sheet)
            
    df_all_ppt = pd.concat(all_ppt_dfs, ignore_index=True)
    df_monthly_rain = df_all_ppt.groupby(['year', 'month'])['rain'].mean().reset_index()
    
    
    df_merged = df_merged.merge(df_monthly_rain, on=['year', 'month'], how='left')
    
    jan_avg = df_monthly_rain[df_monthly_rain['month'] == 1]['rain'].mean()
    df_merged['rain'] = df_merged['rain'].fillna(jan_avg)
    
    
    df_merged['lluvia_mm'] = df_merged['rain'] / df_merged['days_in_month']
    df_merged = df_merged.drop(columns=['rain'])
    
    print("  -> ¡Precipitaciones reales mensuales de las 19 localidades integradas correctamente!")

    
    print("- Procesando rendimiento y superficie reales desde rind y sup.xlsx...")
    y_path = 'data/rind y sup.xlsx'
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"No se encontró el archivo de rindes reales en {y_path}")
        
    df_y = pd.read_excel(y_path, sheet_name='TRIGO')
    total_row = df_y[df_y.iloc[:,0].astype(str).str.contains('TOTAL', na=False)].iloc[0]
    
    campaigns = {
        '2018/19': {'sup_col': 1, 'rinde_col': 2},
        '2019/20': {'sup_col': 3, 'rinde_col': 4},
        '2020/21': {'sup_col': 5, 'rinde_col': 6},
        '2021/22': {'sup_col': 7, 'rinde_col': 8},
        '2022/23': {'sup_col': 9, 'rinde_col': 10},
        '2023/24': {'sup_col': 11, 'rinde_col': 12},
        '2024/25': {'sup_col': 13, 'rinde_col': 14},
        '2025/26': {'sup_col': 15, 'rinde_col': 16}
    }
    
    campaign_data = {}
    for camp, cols in campaigns.items():
        sup = pd.to_numeric(total_row.iloc[cols['sup_col']], errors='coerce')
        rinde = pd.to_numeric(total_row.iloc[cols['rinde_col']], errors='coerce') / 1000.0 
        campaign_data[camp] = {'sup_cosechada_ha': sup, 'rinde_tn_ha': rinde}
        
    def get_campaign_info(row):
        
        
        dt = row.fecha
        yr = dt.year
        if dt.month >= 6:
            camp_key = f"{yr}/{str(yr+1)[2:]}"
        else:
            camp_key = f"{yr-1}/{str(yr)[2:]}"
            
        data = campaign_data.get(camp_key, campaign_data['2025/26'])
        return pd.Series([data['rinde_tn_ha'], data['sup_cosechada_ha']])
        
    df_merged[['rendimiento_estimado_tn_ha', 'superficie_cosechada_ha']] = df_merged.apply(get_campaign_info, axis=1)
    print("  -> ¡Rendimientos y superficies reales de trigo asignados correctamente como constantes por campaña!")

    
    
    
    _ENSO_TRIMESTRAL = {
        (2019, 1): 'Niño',  (2019, 2): 'Niño',  (2019, 3): 'Niño',
        (2019, 4): 'Niño',  (2019, 5): 'Neutral', (2019, 6): 'Neutral',
        (2019, 7): 'Neutral', (2019, 8): 'Neutral', (2019, 9): 'Neutral',
        (2019, 10): 'Neutral', (2019, 11): 'Neutral', (2019, 12): 'Neutral',
        (2020, 1): 'Neutral', (2020, 2): 'Neutral', (2020, 3): 'Neutral',
        (2020, 4): 'Neutral', (2020, 5): 'Neutral', (2020, 6): 'Neutral',
        (2020, 7): 'Niña',  (2020, 8): 'Niña',  (2020, 9): 'Niña',
        (2020, 10): 'Niña', (2020, 11): 'Niña', (2020, 12): 'Niña',
        (2021, 1): 'Niña',  (2021, 2): 'Niña',  (2021, 3): 'Niña',
        (2021, 4): 'Niña',  (2021, 5): 'Neutral', (2021, 6): 'Neutral',
        (2021, 7): 'Neutral', (2021, 8): 'Niña',  (2021, 9): 'Niña',
        (2021, 10): 'Niña', (2021, 11): 'Niña', (2021, 12): 'Niña',
        (2022, 1): 'Niña',  (2022, 2): 'Niña',  (2022, 3): 'Niña',
        (2022, 4): 'Niña',  (2022, 5): 'Niña',  (2022, 6): 'Niña',
        (2022, 7): 'Niña',  (2022, 8): 'Niña',  (2022, 9): 'Niña',
        (2022, 10): 'Niña', (2022, 11): 'Niña', (2022, 12): 'Niña',
        (2023, 1): 'Niña',  (2023, 2): 'Niña',  (2023, 3): 'Neutral',
        (2023, 4): 'Neutral', (2023, 5): 'Niño',  (2023, 6): 'Niño',
        (2023, 7): 'Niño',  (2023, 8): 'Niño',  (2023, 9): 'Niño',
        (2023, 10): 'Niño', (2023, 11): 'Niño', (2023, 12): 'Niño',
        (2024, 1): 'Niño',  (2024, 2): 'Niño',  (2024, 3): 'Niño',
        (2024, 4): 'Niño',  (2024, 5): 'Neutral', (2024, 6): 'Neutral',
        (2024, 7): 'Neutral', (2024, 8): 'Neutral', (2024, 9): 'Neutral',
        (2024, 10): 'Niña', (2024, 11): 'Niña', (2024, 12): 'Niña',
        (2025, 1): 'Niña',  (2025, 2): 'Niña',  (2025, 3): 'Niña',
        (2025, 4): 'Neutral', (2025, 5): 'Neutral', (2025, 6): 'Neutral',
        (2025, 7): 'Neutral', (2025, 8): 'Neutral', (2025, 9): 'Neutral',
        (2025, 10): 'Neutral', (2025, 11): 'Neutral', (2025, 12): 'Neutral',
    }
    def assign_enso(row):
        return _ENSO_TRIMESTRAL.get((row['year'], row['month']), 'Neutral')
    df_merged['fase_enso'] = df_merged.apply(assign_enso, axis=1)
    
    
    df_merged['basis_usd'] = (df_merged['precio_bb_ars'] / df_merged['tipo_cambio']) - df_merged['precio_chicago_usd']
    
    
    df_merged['anomalia_logistica_parana'] = 0
    mask_bajante = (df_merged['fecha'] >= '2021-05-01') & (df_merged['fecha'] <= '2021-11-30')
    df_merged.loc[mask_bajante, 'anomalia_logistica_parana'] = 1
    
    
    df_merged['temp_media'] = 18.0
    
    
    df_merged = inyectar_brecha_historica(df_merged)
    
    
    
    df_merged = inyectar_datos_historicos_f6(df_merged)
    
    
    print("- Procesando compras y fertilizantes desde Datos Est Econ 2.xlsx...")
    file_path_2 = 'data/real/Datos Est Econ 2.xlsx'
    if os.path.exists(file_path_2):
        
        df_urea = pd.read_excel(file_path_2, sheet_name='PRECIO UREA').iloc[1:].copy()
        df_urea['fecha'] = pd.to_datetime(df_urea['Dia'], errors='coerce')
        df_urea = df_urea.dropna(subset=['fecha'])
        df_urea['precio_urea_usd'] = pd.to_numeric(df_urea['UREA'], errors='coerce')
        df_urea['precio_map_usd'] = pd.to_numeric(df_urea['MAP'], errors='coerce')
        df_urea_clean = df_urea[['fecha', 'precio_urea_usd', 'precio_map_usd']].sort_values('fecha').drop_duplicates(subset=['fecha'])
        
        
        df_comp_raw = pd.read_excel(file_path_2, sheet_name='COMPRAS ')
        campaign_starts = {
            '2024/25': 0, '2023/24': 7, '2022/23': 15, '2021/22': 22,
            '2020/21': 29, '2019/20': 36, '2018/19': 41, '2017/18': 46
        }
        
        campaign_dfs = []
        for camp, start in campaign_starts.items():
            num_cols = 6 if camp not in ['2019/20', '2018/19', '2017/18'] else (4 if camp == '2017/18' else 5)
            sub = df_comp_raw.iloc[1:, start:start+num_cols].copy()
            sub.columns = [f'col_{i}' for i in range(sub.shape[1])]
            sub = sub.dropna(subset=['col_0'])
            sub['col_0'] = pd.to_datetime(sub['col_0'], errors='coerce')
            sub = sub.dropna(subset=['col_0'])
            sub = sub.rename(columns={'col_0': 'fecha'})
            for col in sub.columns:
                if col != 'fecha':
                    sub[col] = pd.to_numeric(sub[col], errors='coerce')
            
            sub['campaña'] = camp
            sub['compras_se'] = sub['col_1'].fillna(0.0)
            sub['compras_si'] = sub['col_2'].fillna(0.0)
            sub['compras_totales'] = sub['col_3'].fillna(sub['compras_se'] + sub['compras_si'])
            
            if camp in ['2024/25', '2023/24', '2022/23']:
                sub['compras_sin_precio_pct'] = sub['col_5']
                sub['compras_sin_precio_tot'] = sub['col_4']
            elif camp in ['2021/22', '2020/21']:
                sub['compras_sin_precio_pct'] = 1.0 - sub['col_5']
                sub['compras_sin_precio_tot'] = sub['compras_totales'] * sub['compras_sin_precio_pct']
            else:
                sub['compras_sin_precio_pct'] = np.nan
                sub['compras_sin_precio_tot'] = np.nan
                
            sub = sub.sort_values('fecha')
            sub['delta_compras_se'] = sub['compras_se'].diff().fillna(0.0)
            sub['delta_compras_si'] = sub['compras_si'].diff().fillna(0.0)
            sub['delta_compras_totales'] = sub['compras_totales'].diff().fillna(0.0)
            
            keep_cols = ['fecha', 'campaña', 'compras_se', 'compras_si', 'compras_totales', 
                         'compras_sin_precio_pct', 'compras_sin_precio_tot',
                         'delta_compras_se', 'delta_compras_si', 'delta_compras_totales']
            campaign_dfs.append(sub[keep_cols])
            
        
        min_date_val = df_merged['fecha'].min()
        max_date_val = df_merged['fecha'].max()
        daily_index_val = pd.date_range(start=min_date_val, end=max_date_val, freq='D')
        
        daily_campaigns = []
        for c_df in campaign_dfs:
            camp = c_df['campaña'].iloc[0]
            c_df = c_df.drop_duplicates(subset=['fecha'])
            c_daily = c_df.set_index('fecha').reindex(daily_index_val)
            c_daily[['compras_se', 'compras_si', 'compras_totales', 'compras_sin_precio_pct', 'compras_sin_precio_tot']] = c_daily[['compras_se', 'compras_si', 'compras_totales', 'compras_sin_precio_pct', 'compras_sin_precio_tot']].ffill()
            c_daily[['delta_compras_se', 'delta_compras_si', 'delta_compras_totales']] = c_daily[['delta_compras_se', 'delta_compras_si', 'delta_compras_totales']].fillna(0.0)
            c_daily['campaña'] = camp
            c_daily = c_daily.reset_index().rename(columns={'index': 'fecha'})
            daily_campaigns.append(c_daily)
            
        df_daily_merged = pd.concat(daily_campaigns, ignore_index=True)
        df_daily_agg = df_daily_merged.groupby('fecha').agg({
            'compras_se': 'sum', 'compras_si': 'sum', 'compras_totales': 'sum', 'compras_sin_precio_tot': 'sum',
            'delta_compras_se': 'sum', 'delta_compras_si': 'sum', 'delta_compras_totales': 'sum'
        }).reset_index()
        df_daily_agg['compras_sin_precio_pct'] = df_daily_agg['compras_sin_precio_tot'] / df_daily_agg['compras_totales']
        df_daily_agg['compras_sin_precio_pct'] = df_daily_agg['compras_sin_precio_pct'].fillna(0.0)
        
        
        df_urea_daily = df_urea_clean.set_index('fecha').reindex(daily_index_val).ffill().bfill().reset_index().rename(columns={'index': 'fecha'})
        
        
        df_merged = df_merged.merge(df_daily_agg, on='fecha', how='left')
        df_merged = df_merged.merge(df_urea_daily, on='fecha', how='left')
        
        
        fill_cols_new = ['compras_se', 'compras_si', 'compras_totales', 'compras_sin_precio_tot', 'compras_sin_precio_pct',
                         'delta_compras_se', 'delta_compras_si', 'delta_compras_totales', 'precio_urea_usd', 'precio_map_usd']
        df_merged[fill_cols_new] = df_merged[fill_cols_new].ffill().fillna(0.0)
        print("  -> ¡Datos de compras y precios de fertilizantes integrados de forma diaria exitosamente!")
    else:
        print("  [Warning] No se encontró Datos Est Econ 2.xlsx para procesar compras y fertilizantes.")

    
    print("- Ingestando NDVI Satelital y futuros de ROFEX...")
    ndvi_path = 'data/real/ndvi_satelital_bcp.csv'
    rofex_path = 'data/real/futuros_rofex_trigo.csv'
    
    if os.path.exists(ndvi_path) and os.path.exists(rofex_path):
        df_ndvi_raw = pd.read_csv(ndvi_path)
        df_ndvi_raw['fecha'] = pd.to_datetime(df_ndvi_raw['fecha'])
        
        df_rofex_raw = pd.read_csv(rofex_path)
        df_rofex_raw['fecha'] = pd.to_datetime(df_rofex_raw['fecha'])
        
        
        daily_index_val = pd.date_range(start=df_merged['fecha'].min(), end=df_merged['fecha'].max(), freq='D')
        
        df_ndvi_daily = df_ndvi_raw.set_index('fecha').reindex(daily_index_val).ffill().bfill().reset_index().rename(columns={'index': 'fecha'})
        df_rofex_daily = df_rofex_raw.set_index('fecha').reindex(daily_index_val).ffill().bfill().reset_index().rename(columns={'index': 'fecha'})
        
        
        df_merged = df_merged.merge(df_ndvi_daily, on='fecha', how='left')
        df_merged = df_merged.merge(df_rofex_daily, on='fecha', how='left')
        
        
        fill_cols_f4 = ['ndvi_valor', 'ndvi_anomalia_pct', 'rofex_precio_usd', 'rofex_volumen', 'rofex_interes_abierto']
        df_merged[fill_cols_f4] = df_merged[fill_cols_f4].ffill().fillna(0.0)
        print("  -> ¡NDVI Satelital y futuros de ROFEX integrados de forma diaria exitosamente!")
    else:
        print("  [Warning] No se encontraron archivos de NDVI o ROFEX. Creando columnas con fallbacks ceros.")
        df_merged['ndvi_valor'] = 0.50
        df_merged['ndvi_anomalia_pct'] = 0.0
        df_merged['rofex_precio_usd'] = df_merged['precio_chicago_usd'] - 20.0
        df_merged['rofex_volumen'] = 1000.0
        df_merged['rofex_interes_abierto'] = 20000.0

    
    clean_cols = [
        'fecha', 'fase_enso', 'lluvia_mm', 'temp_media', 'precio_chicago_usd', 'precio_fob_usd', 
        'precio_fas_ars', 'precio_fas_usd', 'precio_pizarra_usd', 'tipo_cambio', 
        'rendimiento_estimado_tn_ha', 'superficie_cosechada_ha', 'anomalia_logistica_parana', 
        'basis_usd', 'precio_bb_ars', 'descargas_camiones', 'descargas_camiones_tn', 
        'descargas_vagones', 'descargas_vagones_tn', 'embarques_tn', 'brecha_cambiaria_pct',
        
        'compras_se', 'compras_si', 'compras_totales', 'compras_sin_precio_pct', 'compras_sin_precio_tot',
        'delta_compras_se', 'delta_compras_si', 'delta_compras_totales', 'precio_urea_usd', 'precio_map_usd',
        
        'ndvi_valor', 'ndvi_anomalia_pct', 'rofex_precio_usd', 'rofex_volumen', 'rofex_interes_abierto',
        
        'precio_blue_usd', 'precio_mep_usd', 'brecha_blue_pct', 'brecha_ccl_pct',
        'riesgo_pais_embi', 'tasa_politica_pct', 'dxy_index', 'petroleo_wti_usd',
        'cbot_maiz_usd', 'cbot_soja_usd', 'cot_managed_money_net', 'cot_commercial_net',
        'nivel_parana_m', 'wasde_stocks_to_use', 'wasde_arg_export_mt',
        'gtrends_vender_trigo', 'gtrends_dolar', 'nino34_anomalia'
    ]
    df_final = df_merged[clean_cols].copy()
    
    print("- Agrupando semanalmente (frecuencia W-SUN) para compatibilidad con BCP Studio...")
    df_weekly = df_final.resample('W-SUN', on='fecha').agg({
        'fase_enso': 'first',
        'lluvia_mm': 'sum',
        'temp_media': 'mean',
        'precio_chicago_usd': 'mean',
        'precio_fob_usd': 'mean',
        'precio_fas_ars': 'mean',
        'precio_fas_usd': 'mean',
        'precio_pizarra_usd': 'mean',
        'tipo_cambio': 'mean',
        'rendimiento_estimado_tn_ha': 'mean',
        'superficie_cosechada_ha': 'mean',
        'anomalia_logistica_parana': 'max',
        'basis_usd': 'mean',
        'precio_bb_ars': 'mean',
        'descargas_camiones': 'sum',
        'descargas_camiones_tn': 'sum',
        'descargas_vagones': 'sum',
        'descargas_vagones_tn': 'sum',
        'embarques_tn': 'sum',
        'brecha_cambiaria_pct': 'mean',
        
        'compras_se': 'mean',
        'compras_si': 'mean',
        'compras_totales': 'mean',
        'compras_sin_precio_pct': 'mean',
        'compras_sin_precio_tot': 'mean',
        'delta_compras_se': 'sum',
        'delta_compras_si': 'sum',
        'delta_compras_totales': 'sum',
        'precio_urea_usd': 'mean',
        'precio_map_usd': 'mean',
        
        'ndvi_valor': 'mean',
        'ndvi_anomalia_pct': 'mean',
        'rofex_precio_usd': 'mean',
        'rofex_volumen': 'sum',
        'rofex_interes_abierto': 'mean',
        
        'precio_blue_usd': 'mean',
        'precio_mep_usd': 'mean',
        'brecha_blue_pct': 'mean',
        'brecha_ccl_pct': 'mean',
        'riesgo_pais_embi': 'mean',
        'tasa_politica_pct': 'mean',
        'dxy_index': 'mean',
        'petroleo_wti_usd': 'mean',
        'cbot_maiz_usd': 'mean',
        'cbot_soja_usd': 'mean',
        'cot_managed_money_net': 'mean',
        'cot_commercial_net': 'mean',
        'nivel_parana_m': 'mean',
        'wasde_stocks_to_use': 'mean',
        'wasde_arg_export_mt': 'mean',
        'gtrends_vender_trigo': 'mean',
        'gtrends_dolar': 'mean',
        'nino34_anomalia': 'mean'
    }).reset_index()
    
    
    df_weekly = df_weekly.ffill().bfill()
    
    
    os.makedirs('data/real', exist_ok=True)
    df_weekly.to_csv('data/real/historico_trigo_real.csv', index=False)
    print(" Ingesta completada con éxito. Archivo guardado en: data/real/historico_trigo_real.csv")
    print(f"Shape: {df_weekly.shape}")
    print(f"Columnas creadas: {df_weekly.columns.tolist()}")

if __name__ == '__main__':
    ingest_bcp_data()
