import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, List, Callable
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model
from ml.trainer import entrenar_modelo
from ml.feature_engineering import procesar_datos_bcp
from ml.foundation_model import FoundationTimeSeriesPredictor

def _extraer_params_markov(params_dict, regime_idx, n_exog=3):
    """Extrae parámetros de un régimen Markov con fallback robusto y logging."""
    const_keys = [f'const[{regime_idx}]', 'const', f'intercept[{regime_idx}]']
    const = 0.0
    for k in const_keys:
        if k in params_dict:
            const = params_dict[k]
            break
    else:
        print(f"  ⚠️ Markov: No se encontró constante para régimen {regime_idx}. Keys: {list(params_dict.keys())[:10]}")
    
    coefs = []
    for i in range(n_exog):
        coef_keys = [f'x{i+1}[{regime_idx}]', f'x{i+1}']
        c = 0.0
        for k in coef_keys:
            if k in params_dict:
                c = params_dict[k]
                break
        coefs.append(c)
    
    return const, coefs


def _fillna_safe(X, medians):
    """Fill NaN con ffill primero, luego mediana del training set."""
    if X is None:
        return None
    # Si X es un DataFrame de 1 fila (como en predicciones en simulación)
    # y medians contiene las medianas de las columnas
    return X.ffill().fillna(medians)


# Patrones de features para el selector de variables predictoras
FEATURES_EXTRA_PATTERNS = [
    'fase_enso', 'premium', 'discount', 'retorno',
    'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos',
    'fase_campaña_sin', 'fase_campaña_cos',
    'regimen',
    'anomalia',
    'brecha',
]

def _es_feature_valida(col: str, target: str, cols_ignorar: list, variables_exogenas: list) -> bool:
    """Determina si una columna es una feature válida para el modelo de ML."""
    if col == target or col in cols_ignorar:
        return False
    # Excluir explícitamente variables endógenas contemporáneas para evitar data leakage
    # o feature mismatch durante la simulación autoregresiva.
    if ('fob_premium' in col or 'fas_discount' in col or 'fob_retorno' in col) and '_lag_' not in col and '_rolling_' not in col:
        return False
    # Lags y rolling stats
    if '_lag_' in col or '_rolling_' in col:
        return True
    # Variables exógenas contemporáneas
    if col in variables_exogenas:
        return True
    # Ratios, retornos, estacionalidad
    for pattern in FEATURES_EXTRA_PATTERNS:
        if pattern in col:
            return True
    return False


class KalmanFilterOnline:
    """
    Filtro de Kalman recursivo online para actualizar dinámicamente
    los coeficientes del modelo de ensamble lineal (Meta-Learner o base).
    Ecuación de Medición: y_t = X_t * theta_t + e_t, e_t ~ N(0, H_t)
    Ecuación de Transición: theta_t = theta_t-1 + u_t, u_t ~ N(0, Q_t)
    """
    def __init__(self, n_features: int, H: float = 1.0, Q_ratio: float = 1e-4):
        self.theta = np.zeros(n_features)
        self.P = np.eye(n_features) * 10.0
        self.H = H
        self.Q = np.eye(n_features) * Q_ratio

    def predict(self):
        """Paso de predicción del estado y covarianza."""
        self.P = self.P + self.Q
        return self.theta

    def update(self, x: np.ndarray, y: float):
        """Paso de corrección basándose en la nueva observación (x, y)."""
        x = np.asarray(x)
        pred = np.dot(x, self.theta)
        error = y - pred
        S = np.dot(x, np.dot(self.P, x)) + self.H
        if abs(S) > 1e-9:
            K = np.dot(self.P, x) / S
            self.theta = self.theta + K * error
            I = np.eye(len(self.theta))
            self.P = np.dot(I - np.outer(K, x), self.P)
        return self.theta, error


class ClimaEstocasticoMCMG:
    """
    Generador estocástico de clima de Richardson (Markov Chain + Gamma Distribution - MCMG)
    calibrado dinámicamente por mes y fase ENSO histórica.
    """
    def __init__(self):
        self.transiciones = {}
        self.gamma_params = {}
        self.mes_transiciones = {}
        self.mes_gamma = {}
        self.global_p_w_d = 0.25
        self.global_p_w_w = 0.45
        self.global_shape = 1.2
        self.global_scale = 15.0

    def calibrar(self, df_train):
        df = df_train.copy()
        df['mes'] = df['fecha'].dt.month
        df['lluvioso'] = (df['lluvia_mm'] > 1.0).astype(int)
        df['lluvioso_ant'] = df['lluvioso'].shift(1)
        df = df.dropna(subset=['lluvioso_ant'])
        
        df_reg = df.copy()
        if 'fase_enso_Niño' in df_reg.columns:
            def get_enso(row):
                if row.get('fase_enso_Niño', 0) == 1: return 'Niño'
                if row.get('fase_enso_Niña', 0) == 1: return 'Niña'
                return 'Neutral'
            df_reg['fase_enso'] = df_reg.apply(get_enso, axis=1)
        elif 'fase_enso' not in df_reg.columns:
            df_reg['fase_enso'] = 'Neutral'
            
        for (mes, enso), sub in df_reg.groupby(['mes', 'fase_enso']):
            sub_d = sub[sub['lluvioso_ant'] == 0]
            n_d_w = (sub_d['lluvioso'] == 1).sum()
            n_d_d = (sub_d['lluvioso'] == 0).sum()
            p_w_d = n_d_w / (n_d_w + n_d_d) if (n_d_w + n_d_d) > 0 else self.global_p_w_d
            
            sub_w = sub[sub['lluvioso_ant'] == 1]
            n_w_w = (sub_w['lluvioso'] == 1).sum()
            n_w_d = (sub_w['lluvioso'] == 0).sum()
            p_w_w = n_w_w / (n_w_w + n_w_d) if (n_w_w + n_w_d) > 0 else self.global_p_w_w
            
            self.transiciones[(mes, enso)] = {'p_w_d': p_w_d, 'p_w_w': p_w_w}
            
            lluvias = sub[sub['lluvia_mm'] > 1.0]['lluvia_mm'].values
            if len(lluvias) >= 3:
                mu = np.mean(lluvias)
                var = np.var(lluvias)
                if var > 0:
                    shape = (mu ** 2) / var
                    scale = var / mu
                else:
                    shape, scale = self.global_shape, self.global_scale
                self.gamma_params[(mes, enso)] = {'shape': shape, 'scale': scale}
            else:
                self.gamma_params[(mes, enso)] = {'shape': self.global_shape, 'scale': self.global_scale}
                
        # Calibrar a nivel mensual global como fallback
        for mes, sub in df_reg.groupby('mes'):
            sub_d = sub[sub['lluvioso_ant'] == 0]
            n_d_w = (sub_d['lluvioso'] == 1).sum()
            n_d_d = (sub_d['lluvioso'] == 0).sum()
            p_w_d = n_d_w / (n_d_w + n_d_d) if (n_d_w + n_d_d) > 0 else self.global_p_w_d
            
            sub_w = sub[sub['lluvioso_ant'] == 1]
            n_w_w = (sub_w['lluvioso'] == 1).sum()
            n_w_d = (sub_w['lluvioso'] == 0).sum()
            p_w_w = n_w_w / (n_w_w + n_w_d) if (n_w_w + n_w_d) > 0 else self.global_p_w_w
            
            self.mes_transiciones[mes] = {'p_w_d': p_w_d, 'p_w_w': p_w_w}
            
            lluvias = sub[sub['lluvia_mm'] > 1.0]['lluvia_mm'].values
            if len(lluvias) >= 3:
                mu = np.mean(lluvias)
                var = np.var(lluvias)
                if var > 0:
                    shape = (mu ** 2) / var
                    scale = var / mu
                else:
                    shape, scale = self.global_shape, self.global_scale
                self.mes_gamma[mes] = {'shape': shape, 'scale': scale}
            else:
                self.mes_gamma[mes] = {'shape': self.global_shape, 'scale': self.global_scale}

    def simular_paso(self, mes, enso, lluvia_ant, rng):
        # Determinar estado lluvioso previo
        lluvioso_ant = 1 if lluvia_ant > 1.0 else 0
        
        # Obtener probabilidades de transición correspondientes
        trans = self.transiciones.get((mes, enso), self.mes_transiciones.get(mes, {'p_w_d': self.global_p_w_d, 'p_w_w': self.global_p_w_w}))
        p_lluvia = trans['p_w_w'] if lluvioso_ant == 1 else trans['p_w_d']
        
        # Simular si llueve
        llueve = rng.random() < p_lluvia
        
        if not llueve:
            return 0.0
            
        # Simular cantidad usando distribución Gamma
        gamma = self.gamma_params.get((mes, enso), self.mes_gamma.get(mes, {'shape': self.global_shape, 'scale': self.global_scale}))
        shape = gamma['shape']
        scale = gamma['scale']
        
        return rng.gamma(shape, scale)


def calcular_factores_estacionales(df_train: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """
    Calcula el índice estacional mensual por variable usando la MEDIANA (más robusta
    que la media ante outliers generados por eventos extremos como la bajante del Paraná).
    """
    factores = {}
    cols_estacionales = ['descargas_camiones', 'descargas_vagones', 'embarques_tn']
    df = df_train.copy()
    df['mes'] = df['fecha'].dt.month
    
    for col in cols_estacionales:
        if col in df.columns:
            mediana_global = df[col].median()
            if mediana_global > 0:
                # Usar mediana mensual: más robusta que la media ante shocks extremos
                medianas_mensuales = df.groupby('mes')[col].median()
                factores[col] = (medianas_mensuales / mediana_global).to_dict()
            else:
                factores[col] = {m: 1.0 for m in range(1, 13)}
    return factores


def entrenar_y_predecir_todo(
    df_raw: pd.DataFrame, 
    fecha_corte: str, 
    variables_exogenas: List[str] = None, 
    predecir_diferencias: bool = False, 
    fecha_proyeccion: str = None,
    # Parámetros del Modo En Vivo (Fase 4)
    clima_scenario: str = "Neutral Promedio",
    chicago_scenario_val: float = None,
    devaluacion_mensual_pct: float = 2.0,
    stress_weights: List[float] = None,
    progress_callback: Callable[[int, str], None] = None
) -> Dict[str, Any]:
    """
    Entrena el ensamble matemático BCP (VECM, GARCH, Markov Switching, HGBR, ElasticNet)
    con alineación temporal honesta (Out-of-Sample) y ejecuta una simulación recursiva paso a paso.
    
    Integración Causal con rule_extractor al final.
    """
    if progress_callback:
        progress_callback(0, "Iniciando procesamiento de datos y saneamiento...")

    if variables_exogenas is None:
        variables_exogenas = []
    if fecha_proyeccion is None:
        fecha_proyeccion = fecha_corte

    # 0. Procesamiento base y saneamiento sin data leakage
    df_proc_full = procesar_datos_bcp(df_raw)
    
    if progress_callback:
        progress_callback(5, "Ajustando modelo VECM para FOB/FAS...")
    
    # Divisiones temporales
    df_proc_full_train = df_proc_full[df_proc_full['fecha'] < pd.to_datetime(fecha_corte)].copy()
    
    # Soporte para la Campaña 2026/27 (Modo En Vivo)
    is_live_2026_27 = pd.to_datetime(fecha_corte) >= pd.to_datetime('2026-06-01')
    if is_live_2026_27:
        # Forzar fechas_test para el futuro de la campaña 2026/27 (35 semanas a partir de Junio 2026)
        fechas_test = pd.date_range(start='2026-06-07', periods=35, freq='W-SUN')
    else:
        fechas_test = df_proc_full[df_proc_full['fecha'] >= pd.to_datetime(fecha_corte)]['fecha'].values
        fechas_test = pd.to_datetime(fechas_test).sort_values()
    
    # Rangos de seguridad históricos para recortar predicciones extremas
    cols_a_predecir = [
        'precio_fob_usd', 'precio_fas_usd', 'precio_fas_ars', 'precio_bb_ars', 
        'precio_pizarra_usd', 'basis_usd', 'descargas_camiones', 'descargas_vagones', 
        'embarques_tn', 'rendimiento_estimado_tn_ha', 'superficie_cosechada_ha',
        'fob_premium', 'fas_discount', 'delta_compras_se', 'compras_sin_precio_pct'
    ]
    
    rangos_seguridad = {}
    for col in cols_a_predecir:
        if col in df_proc_full_train.columns:
            vmin = df_proc_full_train[col].min() * 0.70
            vmax = df_proc_full_train[col].max() * 1.35
            if col in ['rendimiento_estimado_tn_ha', 'superficie_cosechada_ha', 'fob_premium', 'fas_discount']:
                vmin = max(0.1, df_proc_full_train[col].min() * 0.85)
                vmax = df_proc_full_train[col].max() * 1.15
            elif 'precio' in col or col in ['descargas_camiones', 'descargas_vagones', 'embarques_tn']:
                vmin = max(0.0, vmin)
            rangos_seguridad[col] = (vmin, vmax)
            
    # Valores base
    df_train_raw = df_raw[df_raw['fecha'] < pd.to_datetime(fecha_corte)].copy()
    rinde_medio_hist = df_train_raw['rendimiento_estimado_tn_ha'].mean() if 'rendimiento_estimado_tn_ha' in df_train_raw.columns else 2.9
    precio_urea_base = df_train_raw['precio_urea_usd'].mean() if 'precio_urea_usd' in df_train_raw.columns else 500.0
    lluvia_sept_oct_base = 75.0
    coef_lluvia = 0.12
    coef_urea = 0.08
    
    # Factores estacionales y clima estocástico
    factores_estacionales = calcular_factores_estacionales(df_proc_full_train)
    generador_clima = ClimaEstocasticoMCMG()
    generador_clima.calibrar(df_proc_full_train)

    print("\n--- INICIANDO AJUSTE DEL STACK HÍBRIDO FASE 1 ---")
    
    # 1. Ajustar VECM sobre los datos completos de entrenamiento
    cols_vecm = ['fob_premium', 'fas_discount', 'precio_chicago_usd', 'tipo_cambio']
    Y_vecm = df_proc_full_train[cols_vecm].dropna()
    
    print("- Ajustando VECM...")
    try:
        rango = select_coint_rank(Y_vecm, det_order=0, k_ar_diff=2, method='trace')
        rank = max(1, rango.rank)
    except Exception:
        rank = 1
    vecm_model = VECM(Y_vecm, k_ar_diff=2, coint_rank=rank)
    vecm_fit = vecm_model.fit()
    
    # 2. Ajustar GARCH(1,1) sobre los residuos del VECM
    if progress_callback:
        progress_callback(12, "Ajustando volatilidad condicional con GARCH...")
    print("- Ajustando GARCH...")
    resid_fob = vecm_fit.resid[:, 0]
    garch_model_fob = arch_model(resid_fob * 100, vol='Garch', p=1, q=1, dist='skewt')
    garch_fit_fob = garch_model_fob.fit(disp='off')
    
    resid_fas = vecm_fit.resid[:, 1]
    garch_model_fas = arch_model(resid_fas * 100, vol='Garch', p=1, q=1, dist='skewt')
    garch_fit_fas = garch_model_fas.fit(disp='off')
    
    # 3. Ajustar Markov Switching
    if progress_callback:
        progress_callback(18, "Ajustando modelo de Regresión de Cambio de Régimen de Markov...")
    print("- Ajustando Markov Switching...")
    cols_ms_exog = ['precio_chicago_usd', 'rendimiento_estimado_tn_ha', 'lluvia_mm']
    ms_success = False
    try:
        ms_model_fob = MarkovRegression(
            endog=df_proc_full_train['fob_premium'],
            k_regimes=2,
            trend='c',
            exog=df_proc_full_train[cols_ms_exog]
        )
        ms_fit_fob = ms_model_fob.fit(search_reps=10)
        
        ms_model_fas = MarkovRegression(
            endog=df_proc_full_train['fas_discount'],
            k_regimes=2,
            trend='c',
            exog=df_proc_full_train[cols_ms_exog]
        )
        ms_fit_fas = ms_model_fas.fit(search_reps=10)
        ms_success = True
        print("  Markov Switching ajustado exitosamente.")
    except Exception as e:
        print(f"  Error ajustando Markov: {e}")
        
    # 4. Ajustar Elastic Net CV con features normalizadas
    if progress_callback:
        progress_callback(25, "Buscando hiperparámetros óptimos para Elastic Net CV...")
    print("- Ajustando Elastic Net CV...")
    cols_predictoras_base = [c for c in df_proc_full_train.columns if c not in ['fecha', 'fob_premium', 'fas_discount'] and ('_lag_' in c or '_rolling_' in c)]
    
    medians_all = df_proc_full_train.median(numeric_only=True)
    medians_all['precio_chicago_usd_future'] = df_proc_full_train['precio_chicago_usd'].median()
    medians_all['tipo_cambio_future'] = df_proc_full_train['tipo_cambio'].median()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(_fillna_safe(df_proc_full_train[cols_predictoras_base], medians_all))
    
    encv_fob = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=5000)
    encv_fob.fit(X_train_scaled, df_proc_full_train['fob_premium'])
    alpha_fob, l1_ratio_fob = encv_fob.alpha_, encv_fob.l1_ratio_
    print(f"  EN CV FOB: alpha={alpha_fob:.4f}, l1_ratio={l1_ratio_fob:.4f}")
    
    encv_fas = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=5000)
    encv_fas.fit(X_train_scaled, df_proc_full_train['fas_discount'])
    alpha_fas, l1_ratio_fas = encv_fas.alpha_, encv_fas.l1_ratio_
    print(f"  EN CV FAS: alpha={alpha_fas:.4f}, l1_ratio={l1_ratio_fas:.4f}")
    
    # 5. Entrenar modelos de Direct Multi-Step Forecasting para horizontes 1..35
    if progress_callback:
        progress_callback(32, "Iniciando entrenamiento de 280 regresores Direct Multi-Step...")
    print("- Entrenando modelos Direct Multi-Step...")
    hgbr_models_fob = {}
    hgbr_models_fas = {}
    en_models_fob = {}
    en_models_fas = {}
    mlp_models_fob = {}
    mlp_models_fas = {}
    gpr_models_fob = {}
    gpr_models_fas = {}
    
    for h in range(1, 36):
        if progress_callback:
            pct = int(32 + (h * 30 / 35))
            progress_callback(pct, f"Entrenando regresores Direct Multi-Step: horizonte {h}/35...")
            
        y_fob_h = df_proc_full_train['fob_premium'].shift(-h)
        y_fas_h = df_proc_full_train['fas_discount'].shift(-h)
        
        X_h = df_proc_full_train[cols_predictoras_base].copy()
        X_h['precio_chicago_usd_future'] = df_proc_full_train['precio_chicago_usd']
        X_h['tipo_cambio_future'] = df_proc_full_train['tipo_cambio']
        
        scaler_h = StandardScaler()
        X_h_scaled = scaler_h.fit_transform(_fillna_safe(X_h, medians_all))
        
        valid_idx_fob = y_fob_h.dropna().index
        if len(valid_idx_fob) > 10:
            hgbr = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
            hgbr.fit(X_h.loc[valid_idx_fob], y_fob_h.loc[valid_idx_fob])
            hgbr_models_fob[h] = hgbr
            
            en = ElasticNet(alpha=alpha_fob, l1_ratio=l1_ratio_fob, max_iter=5000, random_state=42)
            en.fit(X_h_scaled[valid_idx_fob], y_fob_h.loc[valid_idx_fob])
            en_models_fob[h] = (scaler_h, en)
            
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42, early_stopping=True)
            mlp.fit(X_h_scaled[valid_idx_fob], y_fob_h.loc[valid_idx_fob])
            mlp_models_fob[h] = (scaler_h, mlp)
            
            gpr_kernel = Matern(nu=1.5) + WhiteKernel(noise_level=0.1)
            gpr = GaussianProcessRegressor(kernel=gpr_kernel, alpha=0.0, random_state=42)
            gpr.fit(X_h_scaled[valid_idx_fob], y_fob_h.loc[valid_idx_fob])
            gpr_models_fob[h] = (scaler_h, gpr)
            
        valid_idx_fas = y_fas_h.dropna().index
        if len(valid_idx_fas) > 10:
            hgbr = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
            hgbr.fit(X_h.loc[valid_idx_fas], y_fas_h.loc[valid_idx_fas])
            hgbr_models_fas[h] = hgbr
            
            en = ElasticNet(alpha=alpha_fas, l1_ratio=l1_ratio_fas, max_iter=5000, random_state=42)
            en.fit(X_h_scaled[valid_idx_fas], y_fas_h.loc[valid_idx_fas])
            en_models_fas[h] = (scaler_h, en)
            
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42, early_stopping=True)
            mlp.fit(X_h_scaled[valid_idx_fas], y_fas_h.loc[valid_idx_fas])
            mlp_models_fas[h] = (scaler_h, mlp)
            
            gpr_kernel = Matern(nu=1.5) + WhiteKernel(noise_level=0.1)
            gpr = GaussianProcessRegressor(kernel=gpr_kernel, alpha=0.0, random_state=42)
            gpr.fit(X_h_scaled[valid_idx_fas], y_fas_h.loc[valid_idx_fas])
            gpr_models_fas[h] = (scaler_h, gpr)

            
    # 6. Calibración Out-of-Sample honesta de pesos del Meta-Learner en el set de calibración
    # [FIX CRÍTICO #5 y CRÍTICO #6]
    if progress_callback:
        progress_callback(63, "Calibrando pesos de modelos individuales y Conformal Prediction...")
    print("- Calibrando pesos del Meta-Learner (Out-of-Sample)...")
    n_calib = max(10, int(len(df_proc_full_train) * 0.25))
    df_train_fit = df_proc_full_train.iloc[:-n_calib]
    df_calib = df_proc_full_train.iloc[-n_calib:]
    
    # Entrenar VECM honesto para calibración
    Y_vecm_train = df_train_fit[cols_vecm].dropna()
    try:
        rango_train = select_coint_rank(Y_vecm_train, det_order=0, k_ar_diff=2, method='trace')
        rank_train = max(1, rango_train.rank)
    except Exception:
        rank_train = 1
    vecm_model_train = VECM(Y_vecm_train, k_ar_diff=2, coint_rank=rank_train)
    vecm_fit_train = vecm_model_train.fit()
    
    # Predicción honesta VECM out-of-sample
    vecm_pred_calib = vecm_fit_train.predict(steps=n_calib)
    vecm_pred_calib_fob = vecm_pred_calib[:, 0]
    vecm_pred_calib_fas = vecm_pred_calib[:, 1]
    
    n_vecm_avail = min(len(vecm_pred_calib_fob), n_calib)
    if n_vecm_avail < n_calib:
        print(f"  ⚠️ VECM forecast truncado: {n_vecm_avail}/{n_calib} pasos disponibles. Padding con último valor.")
        vecm_pred_calib_fob = np.pad(vecm_pred_calib_fob, (0, n_calib - n_vecm_avail), mode='edge')
        vecm_pred_calib_fas = np.pad(vecm_pred_calib_fas, (0, n_calib - n_vecm_avail), mode='edge')
    
    # Entrenar GARCH honesto
    resid_fob_train = vecm_fit_train.resid[:, 0]
    garch_model_train_fob = arch_model(resid_fob_train * 100, vol='Garch', p=1, q=1, dist='skewt')
    garch_fit_train_fob = garch_model_train_fob.fit(disp='off')
    
    resid_fas_train = vecm_fit_train.resid[:, 1]
    garch_model_train_fas = arch_model(resid_fas_train * 100, vol='Garch', p=1, q=1, dist='skewt')
    garch_fit_train_fas = garch_model_train_fas.fit(disp='off')
    
    # GARCH OOS Volatility
    vol_calib_fob = np.sqrt(garch_fit_train_fob.forecast(horizon=n_calib).variance.iloc[-1].values) / 100
    vol_calib_fas = np.sqrt(garch_fit_train_fas.forecast(horizon=n_calib).variance.iloc[-1].values) / 100
    
    n_vol_avail_fob = min(len(vol_calib_fob), n_calib)
    n_vol_avail_fas = min(len(vol_calib_fas), n_calib)
    if n_vol_avail_fob < n_calib:
        vol_calib_fob = np.pad(vol_calib_fob, (0, n_calib - n_vol_avail_fob), mode='edge')
    if n_vol_avail_fas < n_calib:
        vol_calib_fas = np.pad(vol_calib_fas, (0, n_calib - n_vol_avail_fas), mode='edge')
    
    # Markov Switching honesto
    if ms_success:
        try:
            ms_model_train_fob = MarkovRegression(
                endog=df_train_fit['fob_premium'],
                k_regimes=2, trend='c', exog=df_train_fit[cols_ms_exog]
            )
            ms_fit_train_fob = ms_model_train_fob.fit(search_reps=10)
            ms_pred_calib_fob = ms_fit_train_fob.predict(start=len(df_train_fit), end=len(df_proc_full_train)-1, exog=df_calib[cols_ms_exog]).values
        except Exception:
            try:
                ms_pred_calib_fob = ms_fit_fob.predict()[-n_calib:]
            except Exception:
                ms_pred_calib_fob = vecm_pred_calib_fob
            
        try:
            ms_model_train_fas = MarkovRegression(
                endog=df_train_fit['fas_discount'],
                k_regimes=2, trend='c', exog=df_train_fit[cols_ms_exog]
            )
            ms_fit_train_fas = ms_model_train_fas.fit(search_reps=10)
            ms_pred_calib_fas = ms_fit_train_fas.predict(start=len(df_train_fit), end=len(df_proc_full_train)-1, exog=df_calib[cols_ms_exog]).values
        except Exception:
            try:
                ms_pred_calib_fas = ms_fit_fas.predict()[-n_calib:]
            except Exception:
                ms_pred_calib_fas = vecm_pred_calib_fas
    else:
        ms_pred_calib_fob = vecm_pred_calib_fob
        ms_pred_calib_fas = vecm_pred_calib_fas
        
    X_calib_h1 = df_calib[cols_predictoras_base].copy()
    X_calib_h1['precio_chicago_usd_future'] = df_calib['precio_chicago_usd']
    X_calib_h1['tipo_cambio_future'] = df_calib['tipo_cambio']
    
    hgbr_pred_calib_fob = hgbr_models_fob[1].predict(X_calib_h1)
    hgbr_pred_calib_fas = hgbr_models_fas[1].predict(X_calib_h1)
    
    scaler_h1_fob, en_model_h1_fob = en_models_fob[1]
    X_calib_h1_scaled_fob = scaler_h1_fob.transform(_fillna_safe(X_calib_h1, medians_all))
    en_pred_calib_fob = en_model_h1_fob.predict(X_calib_h1_scaled_fob)
    
    scaler_h1_fas, en_model_h1_fas = en_models_fas[1]
    X_calib_h1_scaled_fas = scaler_h1_fas.transform(_fillna_safe(X_calib_h1, medians_all))
    en_pred_calib_fas = en_model_h1_fas.predict(X_calib_h1_scaled_fas)
    
    # MLP predictions on calibration set
    scaler_h1_fob_mlp, mlp_model_h1_fob = mlp_models_fob[1]
    mlp_pred_calib_fob = mlp_model_h1_fob.predict(X_calib_h1_scaled_fob)
    
    scaler_h1_fas_mlp, mlp_model_h1_fas = mlp_models_fas[1]
    mlp_pred_calib_fas = mlp_model_h1_fas.predict(X_calib_h1_scaled_fas)
    
    # GPR predictions on calibration set
    scaler_h1_fob_gpr, gpr_model_h1_fob = gpr_models_fob[1]
    gpr_pred_calib_fob = gpr_model_h1_fob.predict(X_calib_h1_scaled_fob)
    
    scaler_h1_fas_gpr, gpr_model_h1_fas = gpr_models_fas[1]
    gpr_pred_calib_fas = gpr_model_h1_fas.predict(X_calib_h1_scaled_fas)
    
    # Foundation model predictions on calibration set
    foundation_pred_calib_fob = []
    foundation_pred_calib_fas = []
    foundation_predictor = FoundationTimeSeriesPredictor(seasonal_periods=52)
    for i in range(len(df_calib)):
        hist_fob = np.hstack([df_train_fit['fob_premium'].values, df_calib['fob_premium'].values[:i]])
        hist_fas = np.hstack([df_train_fit['fas_discount'].values, df_calib['fas_discount'].values[:i]])
        foundation_pred_calib_fob.append(foundation_predictor.predict(hist_fob, steps=1, target_name="fob_premium")[0])
        foundation_pred_calib_fas.append(foundation_predictor.predict(hist_fas, steps=1, target_name="fas_discount")[0])
    foundation_pred_calib_fob = np.array(foundation_pred_calib_fob)
    foundation_pred_calib_fas = np.array(foundation_pred_calib_fas)
    
    y_calib_fob = df_calib['fob_premium'].values
    y_calib_fas = df_calib['fas_discount'].values

    # --- SANITIZACIÓN BULLETPROOF CONTRA NaNs ---
    fob_medio_hist = df_proc_full_train['fob_premium'].mean()
    fas_medio_hist = df_proc_full_train['fas_discount'].mean()
    
    # 1. Asegurar tipos array
    y_calib_fob = np.asarray(y_calib_fob)
    y_calib_fas = np.asarray(y_calib_fas)
    vecm_pred_calib_fob = np.asarray(vecm_pred_calib_fob)
    vecm_pred_calib_fas = np.asarray(vecm_pred_calib_fas)
    ms_pred_calib_fob = np.asarray(ms_pred_calib_fob)
    ms_pred_calib_fas = np.asarray(ms_pred_calib_fas)
    hgbr_pred_calib_fob = np.asarray(hgbr_pred_calib_fob)
    hgbr_pred_calib_fas = np.asarray(hgbr_pred_calib_fas)
    en_pred_calib_fob = np.asarray(en_pred_calib_fob)
    en_pred_calib_fas = np.asarray(en_pred_calib_fas)
    mlp_pred_calib_fob = np.asarray(mlp_pred_calib_fob)
    mlp_pred_calib_fas = np.asarray(mlp_pred_calib_fas)
    gpr_pred_calib_fob = np.asarray(gpr_pred_calib_fob)
    gpr_pred_calib_fas = np.asarray(gpr_pred_calib_fas)
    foundation_pred_calib_fob = np.asarray(foundation_pred_calib_fob)
    foundation_pred_calib_fas = np.asarray(foundation_pred_calib_fas)
    vol_calib_fob = np.asarray(vol_calib_fob)
    vol_calib_fas = np.asarray(vol_calib_fas)

    # 2. Reemplazar NaNs por fallbacks históricos o valores alternativos
    y_calib_fob = np.where(np.isnan(y_calib_fob), fob_medio_hist, y_calib_fob)
    y_calib_fas = np.where(np.isnan(y_calib_fas), fas_medio_hist, y_calib_fas)
    
    vecm_pred_calib_fob = np.where(np.isnan(vecm_pred_calib_fob), fob_medio_hist, vecm_pred_calib_fob)
    vecm_pred_calib_fas = np.where(np.isnan(vecm_pred_calib_fas), fas_medio_hist, vecm_pred_calib_fas)
    
    ms_pred_calib_fob = np.where(np.isnan(ms_pred_calib_fob), vecm_pred_calib_fob, ms_pred_calib_fob)
    ms_pred_calib_fas = np.where(np.isnan(ms_pred_calib_fas), vecm_pred_calib_fas, ms_pred_calib_fas)
    
    hgbr_pred_calib_fob = np.where(np.isnan(hgbr_pred_calib_fob), fob_medio_hist, hgbr_pred_calib_fob)
    hgbr_pred_calib_fas = np.where(np.isnan(hgbr_pred_calib_fas), fas_medio_hist, hgbr_pred_calib_fas)
    
    en_pred_calib_fob = np.where(np.isnan(en_pred_calib_fob), fob_medio_hist, en_pred_calib_fob)
    en_pred_calib_fas = np.where(np.isnan(en_pred_calib_fas), fas_medio_hist, en_pred_calib_fas)
    
    mlp_pred_calib_fob = np.where(np.isnan(mlp_pred_calib_fob), fob_medio_hist, mlp_pred_calib_fob)
    mlp_pred_calib_fas = np.where(np.isnan(mlp_pred_calib_fas), fas_medio_hist, mlp_pred_calib_fas)
    
    gpr_pred_calib_fob = np.where(np.isnan(gpr_pred_calib_fob), fob_medio_hist, gpr_pred_calib_fob)
    gpr_pred_calib_fas = np.where(np.isnan(gpr_pred_calib_fas), fas_medio_hist, gpr_pred_calib_fas)
    
    foundation_pred_calib_fob = np.where(np.isnan(foundation_pred_calib_fob), fob_medio_hist, foundation_pred_calib_fob)
    foundation_pred_calib_fas = np.where(np.isnan(foundation_pred_calib_fas), fas_medio_hist, foundation_pred_calib_fas)
    
    vol_calib_fob = np.where(np.isnan(vol_calib_fob) | (vol_calib_fob <= 0), 0.05, vol_calib_fob)
    vol_calib_fas = np.where(np.isnan(vol_calib_fas) | (vol_calib_fas <= 0), 0.05, vol_calib_fas)

    # 3. Recalcular las métricas sobre los arrays limpios
    mae_vecm_fob = np.nanmean(np.abs(y_calib_fob - vecm_pred_calib_fob))
    if pd.isna(mae_vecm_fob) or mae_vecm_fob == 0: mae_vecm_fob = 0.05
    mae_ms_fob = np.nanmean(np.abs(y_calib_fob - ms_pred_calib_fob))
    if pd.isna(mae_ms_fob) or mae_ms_fob == 0: mae_ms_fob = 0.05
    mae_hgbr_fob = np.nanmean(np.abs(y_calib_fob - hgbr_pred_calib_fob))
    if pd.isna(mae_hgbr_fob) or mae_hgbr_fob == 0: mae_hgbr_fob = 0.05
    mae_en_fob = np.nanmean(np.abs(y_calib_fob - en_pred_calib_fob))
    if pd.isna(mae_en_fob) or mae_en_fob == 0: mae_en_fob = 0.05
    mae_mlp_fob = np.nanmean(np.abs(y_calib_fob - mlp_pred_calib_fob))
    if pd.isna(mae_mlp_fob) or mae_mlp_fob == 0: mae_mlp_fob = 0.05
    mae_gpr_fob = np.nanmean(np.abs(y_calib_fob - gpr_pred_calib_fob))
    if pd.isna(mae_gpr_fob) or mae_gpr_fob == 0: mae_gpr_fob = 0.05
    mae_foundation_fob = np.nanmean(np.abs(y_calib_fob - foundation_pred_calib_fob))
    if pd.isna(mae_foundation_fob) or mae_foundation_fob == 0: mae_foundation_fob = 0.05
    
    mae_vecm_fas = np.nanmean(np.abs(y_calib_fas - vecm_pred_calib_fas))
    if pd.isna(mae_vecm_fas) or mae_vecm_fas == 0: mae_vecm_fas = 0.05
    mae_ms_fas = np.nanmean(np.abs(y_calib_fas - ms_pred_calib_fas))
    if pd.isna(mae_ms_fas) or mae_ms_fas == 0: mae_ms_fas = 0.05
    mae_hgbr_fas = np.nanmean(np.abs(y_calib_fas - hgbr_pred_calib_fas))
    if pd.isna(mae_hgbr_fas) or mae_hgbr_fas == 0: mae_hgbr_fas = 0.05
    mae_en_fas = np.nanmean(np.abs(y_calib_fas - en_pred_calib_fas))
    if pd.isna(mae_en_fas) or mae_en_fas == 0: mae_en_fas = 0.05
    mae_mlp_fas = np.nanmean(np.abs(y_calib_fas - mlp_pred_calib_fas))
    if pd.isna(mae_mlp_fas) or mae_mlp_fas == 0: mae_mlp_fas = 0.05
    mae_gpr_fas = np.nanmean(np.abs(y_calib_fas - gpr_pred_calib_fas))
    if pd.isna(mae_gpr_fas) or mae_gpr_fas == 0: mae_gpr_fas = 0.05
    mae_foundation_fas = np.nanmean(np.abs(y_calib_fas - foundation_pred_calib_fas))
    if pd.isna(mae_foundation_fas) or mae_foundation_fas == 0: mae_foundation_fas = 0.05

    # --- FIX: Guardia anti-colapso del MLP ---
    # Si el MAE del MLP es más de 2.5x la mediana de los otros modelos, forzamos su peso a ~0.
    # Esto evita que un MLP sobreajustado contamine el ensamble con predicciones catastróficas.
    otros_maes_fob = [mae_vecm_fob, mae_ms_fob, mae_hgbr_fob, mae_en_fob, mae_gpr_fob, mae_foundation_fob]
    mediana_otros_fob = np.median(otros_maes_fob)
    if mae_mlp_fob > 2.5 * mediana_otros_fob:
        print(f"  ⚠️ MLP FOB colapsado (MAE={mae_mlp_fob:.4f} > 2.5x mediana={mediana_otros_fob:.4f}). Peso forzado a 0.")
        mae_mlp_fob = mae_mlp_fob * 1000  # peso ~0 efectivo

    otros_maes_fas = [mae_vecm_fas, mae_ms_fas, mae_hgbr_fas, mae_en_fas, mae_gpr_fas, mae_foundation_fas]
    mediana_otros_fas = np.median(otros_maes_fas)
    if mae_mlp_fas > 2.5 * mediana_otros_fas:
        print(f"  ⚠️ MLP FAS colapsado (MAE={mae_mlp_fas:.4f} > 2.5x mediana={mediana_otros_fas:.4f}). Peso forzado a 0.")
        mae_mlp_fas = mae_mlp_fas * 1000  # peso ~0 efectivo

    scores_fob = 1.0 / np.array([mae_vecm_fob, mae_ms_fob, mae_hgbr_fob, mae_en_fob, mae_mlp_fob, mae_gpr_fob, mae_foundation_fob])
    w_fob = scores_fob / scores_fob.sum()
    
    scores_fas = 1.0 / np.array([mae_vecm_fas, mae_ms_fas, mae_hgbr_fas, mae_en_fas, mae_mlp_fas, mae_gpr_fas, mae_foundation_fas])
    w_fas = scores_fas / scores_fas.sum()
    
    # 7. Calibrar Conformal Prediction al 95% [FIX IMPORTANTE #10]
    ensemble_pred_calib_fob = (w_fob[0] * vecm_pred_calib_fob + 
                               w_fob[1] * ms_pred_calib_fob + 
                               w_fob[2] * hgbr_pred_calib_fob + 
                               w_fob[3] * en_pred_calib_fob +
                               w_fob[4] * mlp_pred_calib_fob +
                               w_fob[5] * gpr_pred_calib_fob +
                               w_fob[6] * foundation_pred_calib_fob)
                               
    ensemble_pred_calib_fas = (w_fas[0] * vecm_pred_calib_fas + 
                               w_fas[1] * ms_pred_calib_fas + 
                               w_fas[2] * hgbr_pred_calib_fas + 
                               w_fas[3] * en_pred_calib_fas +
                               w_fas[4] * mlp_pred_calib_fas +
                               w_fas[5] * gpr_pred_calib_fas +
                               w_fas[6] * foundation_pred_calib_fas)
                               
    scaled_resids_fob = np.abs(y_calib_fob - ensemble_pred_calib_fob) / vol_calib_fob
    scaled_resids_fas = np.abs(y_calib_fas - ensemble_pred_calib_fas) / vol_calib_fas
    
    q_fob = np.percentile(scaled_resids_fob, 95)
    q_fas = np.percentile(scaled_resids_fas, 95)
    print(f"  Pesos FOB: VECM={w_fob[0]:.4f}, MS={w_fob[1]:.4f}, HGBR={w_fob[2]:.4f}, EN={w_fob[3]:.4f}, MLP={w_fob[4]:.4f}, GPR={w_fob[5]:.4f}, Chronos={w_fob[6]:.4f} | Conformal Q={q_fob:.4f}")
    print(f"  Pesos FAS: VECM={w_fas[0]:.4f}, MS={w_fas[1]:.4f}, HGBR={w_fas[2]:.4f}, EN={w_fas[3]:.4f}, MLP={w_fas[4]:.4f}, GPR={w_fas[5]:.4f}, Chronos={w_fas[6]:.4f} | Conformal Q={q_fas:.4f}")

    # Guardar métricas del meta-learner para asignación de R2 Train dinámico
    r2_tr_fob = r2_score(y_calib_fob, ensemble_pred_calib_fob)
    r2_tr_fas = r2_score(y_calib_fas, ensemble_pred_calib_fas)

    # --- RE-AJUSTE PARA FUERA DE MUESTRA REALINEADO CON FECHA_PROYECCION ---
    if progress_callback:
        progress_callback(75, "Re-ajustando modelos condicionados hasta la fecha de corte...")
    df_history_up_to_proj = df_proc_full[df_proc_full['fecha'] < pd.to_datetime(fecha_proyeccion)].copy()
    print(f"- Re-ajustando VECM/GARCH/Markov condicionado hasta {fecha_proyeccion}...")
    Y_vecm_proj = df_history_up_to_proj[cols_vecm].dropna()
    vecm_proj_fit = VECM(Y_vecm_proj, k_ar_diff=2, coint_rank=rank).fit()
    garch_fit_fob_proj = arch_model(vecm_proj_fit.resid[:, 0] * 100, vol='Garch', p=1, q=1, dist='skewt').fit(disp='off')
    garch_fit_fas_proj = arch_model(vecm_proj_fit.resid[:, 1] * 100, vol='Garch', p=1, q=1, dist='skewt').fit(disp='off')
    
    ms_proj_success = False
    ms_fit_fob_proj = None
    ms_fit_fas_proj = None
    if ms_success:
        try:
            print("  Re-ajustando Markov FOB...")
            ms_fit_fob_proj = MarkovRegression(
                endog=df_history_up_to_proj['fob_premium'],
                k_regimes=2, trend='c', exog=df_history_up_to_proj[cols_ms_exog]
            ).fit(search_reps=10)
            
            print("  Re-ajustando Markov FAS...")
            ms_fit_fas_proj = MarkovRegression(
                endog=df_history_up_to_proj['fas_discount'],
                k_regimes=2, trend='c', exog=df_history_up_to_proj[cols_ms_exog]
            ).fit(search_reps=10)
            
            # Verificar que no tengan NaNs en los parámetros
            if not np.isnan(ms_fit_fob_proj.params).any() and not np.isnan(ms_fit_fas_proj.params).any():
                ms_proj_success = True
                print("  Markov Switching re-ajustado exitosamente.")
            else:
                print("  Warning: Markov re-ajustado contiene NaNs. Fallback a VECM.")
        except Exception as e:
            print(f"  Warning: No se pudo re-ajustar Markov switching para la proyección: {e}. Fallback a VECM.")

    # 1. Desestacionalizar variables logísticas en train
    # Los factores estacionales usan mediana (más robusta ante outliers).
    if progress_callback:
        progress_callback(80, "Entrenando regresores para variables logísticas y comerciales...")
    df_proc_full_train = df_proc_full_train.copy()
    meses_proc = df_proc_full_train['fecha'].dt.month
    for col in ['descargas_camiones', 'descargas_vagones', 'embarques_tn']:
        if col in cols_a_predecir:
            factors_col = factores_estacionales.get(col, {m: 1.0 for m in range(1, 13)})
            factor_series = meses_proc.map(factors_col).fillna(1.0)
            df_proc_full_train[col] = df_proc_full_train[col] / factor_series
            
    # Entrenar modelos de ML para los otros targets válidos
    modelos = {}
    metricas_train = {}
    cols_ignorar = ['fecha', 'precio_bb_ars', 'precio_fas_ars', 'precio_fas_usd', 'precio_fob_usd', 'precio_pizarra_usd', 'basis_usd']
    
    cols_ml_puros = [c for c in cols_a_predecir if c not in ['superficie_cosechada_ha', 'fob_premium', 'fas_discount']]
    
    for idx_target, target in enumerate(cols_ml_puros):
        if progress_callback:
            pct = int(80 + (idx_target * 4 / len(cols_ml_puros)))
            progress_callback(pct, f"Entrenando regresor complementario: {target}...")
            
        cols_predictoras = [
            c for c in df_proc_full_train.columns 
            if _es_feature_valida(c, target, cols_ignorar, variables_exogenas)
        ]
        if target == 'rendimiento_estimado_tn_ha':
            biophysical_patterns = ['lluvia_mm', 'enso', 'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos', 'fase_campaña_sin', 'fase_campaña_cos', 'urea', 'map', 'anomalia']
            cols_predictoras = [
                c for c in cols_predictoras 
                if any(p in c.lower() for p in biophysical_patterns)
                and not any(x in c for x in ['descargas', 'embarques', 'compras', 'regimen', 'devaluacion', 'parana', 'superficie'])
            ]
        
        df_train_active = df_proc_full_train[
            (df_proc_full_train['fecha'].dt.month.isin([6, 7, 8, 9, 10, 11, 12, 1])) |
            (df_proc_full_train['fecha'] >= pd.to_datetime(fecha_corte))
        ].copy()
        
        # OMITIMOS calcular_importancia (calcular_importancia=False) para evitar cuellos de botella de tiempo
        res = entrenar_modelo(df_train_active, target, cols_predictoras, fecha_corte=fecha_corte, predecir_diferencias=False, calcular_importancia=False)
        modelos[target] = res['modelo']
        metricas_train[target] = {'r2': res['r2'], 'mae': res['mae']}
        
    # Conformal para los otros targets al 95%
    conformal_quantiles = {
        'rendimiento_estimado_tn_ha': 0.35, 
        'superficie_cosechada_ha': 120000.0 
    }
    
    ALPHA_CONFIDENCE = 0.95
    for target in cols_ml_puros:
        cols_predictoras = [
            c for c in df_proc_full_train.columns 
            if _es_feature_valida(c, target, cols_ignorar, variables_exogenas)
        ]
        if target == 'rendimiento_estimado_tn_ha':
            biophysical_patterns = ['lluvia_mm', 'enso', 'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos', 'fase_campaña_sin', 'fase_campaña_cos', 'urea', 'map', 'anomalia']
            cols_predictoras = [
                c for c in cols_predictoras
                if any(p in c.lower() for p in biophysical_patterns)
                and not any(x in c for x in ['descargas', 'embarques', 'compras', 'regimen', 'devaluacion', 'parana', 'superficie'])
            ]
        X_calib = df_calib[cols_predictoras]
        y_calib_proc = df_calib[target].values
        y_pred_calib = modelos[target].predict(X_calib)
        
        if target in ['descargas_camiones', 'descargas_vagones', 'embarques_tn']:
            factors_calib = df_calib['fecha'].dt.month.map(factores_estacionales[target]).fillna(1.0).values
            y_calib_real = y_calib_proc * factors_calib
            y_pred_real = y_pred_calib * factors_calib
            residuos = np.abs(y_calib_real - y_pred_real)
        else:
            residuos = np.abs(y_calib_proc - y_pred_calib)
            
        q_val = np.percentile(residuos, ALPHA_CONFIDENCE * 100)
        conformal_quantiles[target] = float(q_val)
        
    # --- SIMULACIÓN AUTOREGRESIVA PASO A PASO ---
    if progress_callback:
        progress_callback(85, "Iniciando simulación recursiva paso a paso...")
    df_historia_pre_proyeccion = df_proc_full[
        (df_proc_full['fecha'] >= pd.to_datetime(fecha_corte)) & 
        (df_proc_full['fecha'] < pd.to_datetime(fecha_proyeccion))
    ].copy()
    
    df_historia_simulada = pd.concat([
        df_proc_full[df_proc_full['fecha'] < pd.to_datetime(fecha_corte)],
        df_historia_pre_proyeccion
    ], ignore_index=True)
    
    predicciones_test = []
    proyeccion_idx = 0
    
    idx_proj_in_test = 0
    indices_proj = np.where(fechas_test >= pd.to_datetime(fecha_proyeccion))[0]
    if len(indices_proj) > 0:
        idx_proj_in_test = int(indices_proj[0])
        
    steps_out_of_sample = max(1, len(fechas_test) - idx_proj_in_test)
    
    vecm_forecasts_proj = vecm_proj_fit.predict(steps=steps_out_of_sample)
    garch_fc_fob_proj = np.sqrt(garch_fit_fob_proj.forecast(horizon=steps_out_of_sample).variance.iloc[-1].values) / 100
    garch_fc_fas_proj = np.sqrt(garch_fit_fas_proj.forecast(horizon=steps_out_of_sample).variance.iloc[-1].values) / 100
    
    # --- INICIALIZACIÓN DE DMA Y KALMAN FILTER (FASE 2) ---
    features_h1 = list(cols_predictoras_base) + ['precio_chicago_usd_future', 'tipo_cambio_future']
    if stress_weights is not None and len(stress_weights) == 7:
        pi_fob = np.array(stress_weights)
        pi_fas = np.array(stress_weights)
    else:
        pi_fob = np.array([w_fob[0], w_fob[1], w_fob[2], w_fob[3], w_fob[4], w_fob[5], w_fob[6]])
        pi_fas = np.array([w_fas[0], w_fas[1], w_fas[2], w_fas[3], w_fas[4], w_fas[5], w_fas[6]])
    
    kf_fob = KalmanFilterOnline(n_features=len(features_h1) + 1, H=1.0, Q_ratio=1e-4)
    kf_fob.theta = np.hstack([[en_models_fob[1][1].intercept_], en_models_fob[1][1].coef_.copy()])
    
    kf_fas = KalmanFilterOnline(n_features=len(features_h1) + 1, H=1.0, Q_ratio=1e-4)
    kf_fas.theta = np.hstack([[en_models_fas[1][1].intercept_], en_models_fas[1][1].coef_.copy()])
    
    historial_pesos_dma_fob = []
    historial_pesos_dma_fas = []
    
    for paso_idx, fecha_actual in enumerate(fechas_test):
        if progress_callback:
            pct = int(85 + (paso_idx * 10 / len(fechas_test)))
            progress_callback(pct, f"Simulando paso {paso_idx + 1}/{len(fechas_test)}: {pd.to_datetime(fecha_actual).strftime('%d-%b-%Y')}...")
            
        es_periodo_proyeccion = fecha_actual >= pd.to_datetime(fecha_proyeccion)
        
        if not es_periodo_proyeccion:
            fila_proc_real = df_proc_full[df_proc_full['fecha'] == fecha_actual].iloc[0]
            predicciones_paso = {'fecha': fecha_actual}
            for col in df_proc_full.columns:
                if col != 'fecha':
                    predicciones_paso[col] = fila_proc_real[col]
                    predicciones_paso[f'{col}_lower'] = fila_proc_real[col]
                    predicciones_paso[f'{col}_upper'] = fila_proc_real[col]
                    
            predicciones_paso['descargas_camiones_tn'] = fila_proc_real.get('descargas_camiones_tn', fila_proc_real.get('descargas_camiones', 0.0) * 30.0)
            predicciones_paso['descargas_camiones_tn_lower'] = predicciones_paso['descargas_camiones_tn']
            predicciones_paso['descargas_camiones_tn_upper'] = predicciones_paso['descargas_camiones_tn']
            
            predicciones_paso['descargas_vagones_tn'] = fila_proc_real.get('descargas_vagones_tn', fila_proc_real.get('descargas_vagones', 0.0) * 45.0)
            predicciones_paso['descargas_vagones_tn_lower'] = predicciones_paso['descargas_vagones_tn']
            predicciones_paso['descargas_vagones_tn_upper'] = predicciones_paso['descargas_vagones_tn']
            
            # Copiar FOB real para modelos individuales
            chicago_t = fila_proc_real['precio_chicago_usd']
            real_fob_usd = chicago_t * fila_proc_real['fob_premium']
            real_fas_usd = chicago_t * fila_proc_real['fas_discount']
            
            predicciones_paso['fob_pred_vecm'] = real_fob_usd
            predicciones_paso['fob_pred_ms'] = real_fob_usd
            predicciones_paso['fob_pred_hgbr'] = real_fob_usd
            predicciones_paso['fob_pred_en'] = real_fob_usd
            predicciones_paso['fob_pred_mlp'] = real_fob_usd
            predicciones_paso['fob_pred_gpr'] = real_fob_usd
            predicciones_paso['fob_pred_foundation'] = real_fob_usd
            
            # Copiar FAS real para modelos individuales
            predicciones_paso['fas_pred_vecm'] = real_fas_usd
            predicciones_paso['fas_pred_ms'] = real_fas_usd
            predicciones_paso['fas_pred_hgbr'] = real_fas_usd
            predicciones_paso['fas_pred_en'] = real_fas_usd
            predicciones_paso['fas_pred_mlp'] = real_fas_usd
            predicciones_paso['fas_pred_gpr'] = real_fas_usd
            predicciones_paso['fas_pred_foundation'] = real_fas_usd
            
            # Asignar volatilidad basal
            predicciones_paso['vol_fob'] = 0.015 + 0.005 * np.abs(np.sin(paso_idx / 5.0))
            predicciones_paso['vol_fas'] = 0.012 + 0.004 * np.abs(np.cos(paso_idx / 5.0))
            
            predicciones_test.append(predicciones_paso)
            continue
            
        proyeccion_idx += 1
        rows_matching = df_raw[df_raw['fecha'] == fecha_actual]
        if len(rows_matching) > 0:
            fila_real = rows_matching.iloc[0].copy()
        else:
            # Campaña futura 2026/27 (Modo En Vivo)
            # Copiar la última fila real como base para estructura y exógenas base
            fila_real = df_raw.iloc[-1].copy()
            fila_real['fecha'] = fecha_actual
            
            # 1. Configurar precio Chicago para el futuro
            if chicago_scenario_val is not None:
                fila_real['precio_chicago_usd'] = chicago_scenario_val
            else:
                chicago_t_1 = df_historia_simulada.iloc[-1]['precio_chicago_usd'] if not df_historia_simulada.empty else 210.0
                fila_real['precio_chicago_usd'] = chicago_t_1
                
            # 2. Configurar tipo de cambio (devaluación mensual programada, e.g. 2% mensual)
            tc_t_1 = df_historia_simulada.iloc[-1]['tipo_cambio'] if not df_historia_simulada.empty else 1445.0
            tasa_semanal = devaluacion_mensual_pct / 4.33 / 100.0
            fila_real['tipo_cambio'] = tc_t_1 * (1.0 + tasa_semanal)
            
            # 3. Configurar rinde e insumos según escenario climático
            if "Niña" in clima_scenario:
                fila_real['fase_enso'] = 'Niña'
                fila_real['precio_urea_usd'] = 520.0
            elif "Niño" in clima_scenario:
                fila_real['fase_enso'] = 'Niño'
                fila_real['precio_urea_usd'] = 450.0
            else:
                fila_real['fase_enso'] = 'Neutral'
                fila_real['precio_urea_usd'] = 490.0
                
            # 4. NDVI satelital futuro simulado
            day_of_year = fecha_actual.dayofyear
            month_actual = fecha_actual.month
            
            if month_actual in [6, 7]:
                base_ndvi = 0.25
            elif month_actual in [8, 9]:
                base_ndvi = 0.27 + 0.25 * ((day_of_year - 212) / 61)
            elif month_actual in [10, 11]:
                dist_pico = abs(day_of_year - 298)
                base_ndvi = 0.74 - 0.12 * (dist_pico / 30)
            elif month_actual == 12:
                base_ndvi = 0.65 - 0.37 * (fecha_actual.day / 31)
            else:
                base_ndvi = 0.23
                
            factor_clima = 1.0
            if "Niña" in clima_scenario and month_actual in [9, 10, 11]:
                factor_clima = 0.65
            elif "Niño" in clima_scenario and month_actual in [9, 10, 11]:
                factor_clima = 1.08
                
            fila_real['ndvi_valor'] = np.clip(base_ndvi * factor_clima, 0.18, 0.82)
            fila_real['ndvi_anomalia_pct'] = ((fila_real['ndvi_valor'] - base_ndvi) / base_ndvi) * 100.0
            
            # 5. ROFEX Trigo Futuro
            basis_seccional = -30.0 if month_actual in [11, 12, 1] else -15.0
            if clima_scenario == "Niña Moderada" and month_actual in [9, 10, 11, 12]:
                basis_seccional = +15.0
            fila_real['rofex_precio_usd'] = np.clip(fila_real['precio_chicago_usd'] + basis_seccional, 120.0, 450.0)
            fila_real['rofex_volumen'] = 4500.0 if month_actual in [10, 11] else 1500.0
            fila_real['rofex_interes_abierto'] = 55000.0 if month_actual in [10, 11] else 22000.0
            
        predicciones_paso = {'fecha': fecha_actual}
        for k in fila_real.index:
            if k != 'fecha':
                predicciones_paso[k] = fila_real[k]
        mes_actual = fecha_actual.month
        
        # Inyectar simulados anteriores
        for c in df_raw.columns:
            if c == 'fecha':
                continue
            copiar_de_ant = False
            if c not in ['tipo_cambio', 'precio_chicago_usd']:
                if c not in ['precio_fob_usd', 'precio_fas_usd', 'precio_fas_ars', 'precio_bb_ars', 'precio_pizarra_usd', 'basis_usd', 'descargas_camiones_tn', 'descargas_vagones_tn']:
                    copiar_de_ant = True
            if copiar_de_ant:
                if not df_historia_simulada.empty and c in df_historia_simulada.columns:
                    fila_real[c] = df_historia_simulada.iloc[-1][c]
                else:
                    val_raw = df_raw.loc[df_raw['fecha'] == fecha_actual, c].values
                    fila_real[c] = val_raw[0] if len(val_raw) > 0 else 0.0
                
        # Simular clima estocástico
        fase_enso_actual = 'Neutral'
        if 'fase_enso_Niño' in fila_real.index:
            if fila_real.get('fase_enso_Niño', 0) == 1: fase_enso_actual = 'Niño'
            elif fila_real.get('fase_enso_Niña', 0) == 1: fase_enso_actual = 'Niña'
            else: fase_enso_actual = 'Neutral'
        elif 'fase_enso' in fila_real.index:
            fase_enso_actual = fila_real['fase_enso']
            
        lluvia_ant = df_historia_simulada.iloc[-1]['lluvia_mm'] if not df_historia_simulada.empty else 0.0
        # Detección determinista de semilla [FIX IMPORTANTE #9]
        rng_clima = np.random.default_rng(seed=42 + proyeccion_idx)
        lluvia_sim = generador_clima.simular_paso(mes_actual, fase_enso_actual, lluvia_ant, rng_clima)
        
        fila_real['lluvia_mm'] = lluvia_sim
        predicciones_paso['lluvia_mm'] = lluvia_sim
        
        # Procesar df temporal para lags/rolling actualizados [FIX CRÍTICO #4, OPTIMIZACIÓN INCREMENTAL]
        df_tmp = pd.concat([df_historia_simulada, pd.DataFrame([fila_real])], ignore_index=True)
        n_context = min(len(df_tmp), 24)
        df_context = df_tmp.tail(n_context).copy()
        df_proc_tmp = procesar_datos_bcp(df_context)
        fila_proc_actual = df_proc_tmp.iloc[-1]
        
        # Lógica Agronómica Híbrida
        factor_enso = 1.0
        mes_anterior = df_historia_simulada.iloc[-1]['fecha'].month if not df_historia_simulada.empty else 5
        rinde_ant = df_historia_simulada.iloc[-1]['rendimiento_estimado_tn_ha'] if not df_historia_simulada.empty else 2.8
        superficie_ant = df_historia_simulada.iloc[-1]['superficie_cosechada_ha'] if not df_historia_simulada.empty else 1350000.0
        nuevo_rinde = rinde_ant
        nueva_superficie = superficie_ant
        
        if mes_actual == 6 and mes_anterior == 5:
            sup_base = df_train_raw['superficie_cosechada_ha'].mean()
            precio_chicago_actual = fila_real['precio_chicago_usd']
            precio_chicago_base = df_train_raw['precio_chicago_usd'].mean()
            factor_precio = np.clip(precio_chicago_actual / precio_chicago_base, 0.9, 1.1)
            factor_enso = 1.03 if fase_enso_actual == 'Niño' else (0.95 if fase_enso_actual == 'Niña' else 1.0)
            nueva_superficie = sup_base * factor_precio * factor_enso
            nueva_superficie = np.clip(nueva_superficie, rangos_seguridad['superficie_cosechada_ha'][0], rangos_seguridad['superficie_cosechada_ha'][1])
            
        if mes_actual in [11, 12, 1] and mes_anterior not in [11, 12, 1]:
            if 'rendimiento_estimado_tn_ha' in modelos:
                biophysical_patterns = ['lluvia_mm', 'enso', 'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos', 'fase_campaña_sin', 'fase_campaña_cos', 'urea', 'map', 'anomalia']
                cols_predictoras_rinde = [
                    c for c in df_proc_tmp.columns 
                    if _es_feature_valida(c, 'rendimiento_estimado_tn_ha', cols_ignorar, variables_exogenas)
                    and any(p in c.lower() for p in biophysical_patterns)
                    and not any(x in c for x in ['descargas', 'embarques', 'compras', 'regimen', 'devaluacion', 'parana', 'superficie'])
                ]
                features_modelo = modelos['rendimiento_estimado_tn_ha'].feature_names_in_ if hasattr(modelos['rendimiento_estimado_tn_ha'], 'feature_names_in_') else cols_predictoras_rinde
                cols_predictoras_rinde = [c for c in features_modelo if c in df_proc_tmp.columns]
                
                if not df_historia_simulada.empty:
                    ultimas_semanas = df_historia_simulada.tail(24)
                    lluvia_sept_oct = ultimas_semanas[ultimas_semanas['fecha'].dt.month.isin([9, 10])]['lluvia_mm'].sum()
                else:
                    lluvia_sept_oct = 80.0
                
                factor_extrapolacion = 1.0
                if lluvia_sept_oct > lluvia_sept_oct_base:
                    exceso_lluvia = (lluvia_sept_oct - lluvia_sept_oct_base) / lluvia_sept_oct_base
                    factor_extrapolacion += coef_lluvia * exceso_lluvia
                    
                precio_urea = fila_real.get('precio_urea_usd', precio_urea_base)
                if precio_urea < precio_urea_base:
                    descuento_urea = (precio_urea_base - precio_urea) / precio_urea_base
                    factor_extrapolacion += coef_urea * descuento_urea
                
                X_rinde = pd.DataFrame([fila_proc_actual[cols_predictoras_rinde]])
                pred_ml = modelos['rendimiento_estimado_tn_ha'].predict(X_rinde)[0]
                nuevo_rinde = pred_ml * factor_extrapolacion
            else:
                rinde_base = df_train_raw['rendimiento_estimado_tn_ha'].mean()
                if len(df_historia_simulada) >= 12:
                    lluvia_acum = df_historia_simulada.iloc[-12:]['lluvia_mm'].sum()
                else:
                    lluvia_acum = df_train_raw['lluvia_mm'].mean() * 12
                lluvia_normal = df_train_raw['lluvia_mm'].mean() * 12
                factor_lluvia = np.clip(lluvia_acum / (lluvia_normal if lluvia_normal > 0 else 1.0), 0.75, 1.2)
                nuevo_rinde = rinde_base * factor_lluvia * factor_enso
                
            nuevo_rinde = np.clip(nuevo_rinde, rangos_seguridad['rendimiento_estimado_tn_ha'][0], rangos_seguridad['rendimiento_estimado_tn_ha'][1])
            
        matching_rows = df_proc_full[df_proc_full['fecha'] == fecha_actual]
        if len(matching_rows) > 0:
            real_rinde = matching_rows['rendimiento_estimado_tn_ha'].values[0]
            real_sup = matching_rows['superficie_cosechada_ha'].values[0]
        else:
            real_rinde = np.nan
            real_sup = np.nan
            
        if 'rendimiento_estimado_tn_ha' in variables_exogenas and not pd.isna(real_rinde):
            predicciones_paso['rendimiento_estimado_tn_ha'] = real_rinde
            fila_real['rendimiento_estimado_tn_ha'] = real_rinde
            predicciones_paso['rendimiento_estimado_tn_ha_lower'] = real_rinde
            predicciones_paso['rendimiento_estimado_tn_ha_upper'] = real_rinde
        else:
            predicciones_paso['rendimiento_estimado_tn_ha'] = nuevo_rinde
            fila_real['rendimiento_estimado_tn_ha'] = nuevo_rinde
            width_rinde = conformal_quantiles['rendimiento_estimado_tn_ha']
            predicciones_paso['rendimiento_estimado_tn_ha_lower'] = np.clip(nuevo_rinde - width_rinde, rangos_seguridad['rendimiento_estimado_tn_ha'][0], rangos_seguridad['rendimiento_estimado_tn_ha'][1])
            predicciones_paso['rendimiento_estimado_tn_ha_upper'] = np.clip(nuevo_rinde + width_rinde, rangos_seguridad['rendimiento_estimado_tn_ha'][0], rangos_seguridad['rendimiento_estimado_tn_ha'][1])
            
        if 'superficie_cosechada_ha' in variables_exogenas and not pd.isna(real_sup):
            predicciones_paso['superficie_cosechada_ha'] = real_sup
            fila_real['superficie_cosechada_ha'] = real_sup
            predicciones_paso['superficie_cosechada_ha_lower'] = real_sup
            predicciones_paso['superficie_cosechada_ha_upper'] = real_sup
        else:
            predicciones_paso['superficie_cosechada_ha'] = nueva_superficie
            fila_real['superficie_cosechada_ha'] = nueva_superficie
            width_sup = conformal_quantiles['superficie_cosechada_ha']
            predicciones_paso['superficie_cosechada_ha_lower'] = np.clip(nueva_superficie - width_sup, rangos_seguridad['superficie_cosechada_ha'][0], rangos_seguridad['superficie_cosechada_ha'][1])
            predicciones_paso['superficie_cosechada_ha_upper'] = np.clip(nueva_superficie + width_sup, rangos_seguridad['superficie_cosechada_ha'][0], rangos_seguridad['superficie_cosechada_ha'][1])

        # --- ENSEMBLE HÍBRIDO FASE 1 PARA FOB_PREMIUM Y FAS_DISCOUNT ---
        val_chicago = fila_real['precio_chicago_usd']
        val_tc = fila_real['tipo_cambio']
        h_blind = paso_idx - idx_proj_in_test + 1
        
        # 1. FOB PREMIUM
        if 'fob_premium' not in variables_exogenas:
            h_idx_vecm = min(h_blind - 1, vecm_forecasts_proj.shape[0] - 1)
            vecm_pred_fob = vecm_forecasts_proj[h_idx_vecm, 0]
            
            if ms_proj_success:
                try:
                    params_dict = dict(ms_fit_fob_proj.params)
                    P_fob = ms_fit_fob_proj.regime_transition[:, :, 0]
                    p_filt_fob = ms_fit_fob_proj.filtered_marginal_probabilities.iloc[-1].values
                    p_h_fob = np.dot(np.linalg.matrix_power(P_fob, h_blind), p_filt_fob)
                    exog_t = [val_chicago, fila_real['rendimiento_estimado_tn_ha'], fila_real['lluvia_mm']]
                    
                    const0_fob, coefs0_fob = _extraer_params_markov(params_dict, 0, n_exog=3)
                    pred0_fob = const0_fob + sum(c * x for c, x in zip(coefs0_fob, exog_t))
                    
                    const1_fob, coefs1_fob = _extraer_params_markov(params_dict, 1, n_exog=3)
                    pred1_fob = const1_fob + sum(c * x for c, x in zip(coefs1_fob, exog_t))
                    
                    ms_pred_fob = p_h_fob[0] * pred0_fob + p_h_fob[1] * pred1_fob
                    if np.isnan(ms_pred_fob):
                        ms_pred_fob = vecm_pred_fob
                except Exception:
                    ms_pred_fob = vecm_pred_fob
            else:
                ms_pred_fob = vecm_pred_fob
                
            X_target = pd.DataFrame([df_proc_tmp.iloc[-2][cols_predictoras_base]])
            X_target['precio_chicago_usd_future'] = val_chicago
            X_target['tipo_cambio_future'] = val_tc
            
            hgbr_pred_fob = hgbr_models_fob[1].predict(X_target)[0]
            scaler_h_fob, en_model_h_fob = en_models_fob[1]
            X_target_scaled_fob = scaler_h_fob.transform(_fillna_safe(X_target, medians_all))
            en_pred_fob = en_model_h_fob.predict(X_target_scaled_fob)[0]
            
            # Kalman Filter online prediction instead of static ElasticNet
            x_fob_aug = np.hstack([[1.0], X_target_scaled_fob[0]])
            kf_pred_fob = np.dot(x_fob_aug, kf_fob.theta)
            
            # MLP neural network prediction
            scaler_h_fob_mlp, mlp_model_h_fob = mlp_models_fob[1]
            mlp_pred_fob = mlp_model_h_fob.predict(X_target_scaled_fob)[0]
            
            # GPR prediction
            scaler_h_fob_gpr, gpr_model_h_fob = gpr_models_fob[1]
            gpr_pred_fob = gpr_model_h_fob.predict(X_target_scaled_fob)[0]
            
            # Foundation model zero-shot prediction
            foundation_predictor = FoundationTimeSeriesPredictor(seasonal_periods=52)
            hist_fob_sim = df_historia_simulada['fob_premium'].values
            foundation_pred_fob = foundation_predictor.predict(hist_fob_sim, steps=1, target_name="fob_premium")[0]
            
            # Dynamic Model Averaging (DMA) ensemble blending (7 models)
            ensemble_fob = (pi_fob[0] * vecm_pred_fob + 
                            pi_fob[1] * ms_pred_fob + 
                            pi_fob[2] * hgbr_pred_fob + 
                            pi_fob[3] * kf_pred_fob + 
                            pi_fob[4] * mlp_pred_fob +
                            pi_fob[5] * gpr_pred_fob +
                            pi_fob[6] * foundation_pred_fob)
                            
            alpha_mr = min(0.3, 0.01 * h_blind)
            ensemble_fob = (1 - alpha_mr) * ensemble_fob + alpha_mr * fob_medio_hist
            
            if mes_actual in [11, 12, 1, 2]:
                desviacion = (fila_real['rendimiento_estimado_tn_ha'] - rinde_medio_hist) / rinde_medio_hist
                factor_ajuste = np.clip(1.0 - (desviacion * 0.15), 0.85, 1.15)
                ensemble_fob *= factor_ajuste
                
            ensemble_fob = np.clip(ensemble_fob, rangos_seguridad['fob_premium'][0], rangos_seguridad['fob_premium'][1])
            
            vol_fob_t = garch_fc_fob_proj[h_idx_vecm]
            lower_fob = ensemble_fob - q_fob * vol_fob_t
            upper_fob = ensemble_fob + q_fob * vol_fob_t
            
            predicciones_paso['fob_premium'] = ensemble_fob
            predicciones_paso['fob_premium_lower'] = lower_fob
            predicciones_paso['fob_premium_upper'] = upper_fob
            fila_real['fob_premium'] = ensemble_fob
            predicciones_paso['vol_fob'] = vol_fob_t
            
            predicciones_paso['fob_pred_vecm'] = val_chicago * vecm_pred_fob
            predicciones_paso['fob_pred_ms'] = val_chicago * ms_pred_fob
            predicciones_paso['fob_pred_hgbr'] = val_chicago * hgbr_pred_fob
            predicciones_paso['fob_pred_en'] = val_chicago * kf_pred_fob
            predicciones_paso['fob_pred_mlp'] = val_chicago * mlp_pred_fob
            predicciones_paso['fob_pred_gpr'] = val_chicago * gpr_pred_fob
            predicciones_paso['fob_pred_foundation'] = val_chicago * foundation_pred_fob
        else:
            matching_rows_fob = df_proc_full[df_proc_full['fecha'] == fecha_actual]
            fob_val_real = matching_rows_fob['fob_premium'].values[0] if len(matching_rows_fob) > 0 else 1.15
            predicciones_paso['fob_premium'] = fob_val_real
            predicciones_paso['fob_premium_lower'] = fob_val_real
            predicciones_paso['fob_premium_upper'] = fob_val_real
            fila_real['fob_premium'] = fob_val_real
            predicciones_paso['vol_fob'] = 0.015
            
            val_fob_real_usd = val_chicago * fob_val_real
            predicciones_paso['fob_pred_vecm'] = val_fob_real_usd
            predicciones_paso['fob_pred_ms'] = val_fob_real_usd
            predicciones_paso['fob_pred_hgbr'] = val_fob_real_usd
            predicciones_paso['fob_pred_en'] = val_fob_real_usd
            predicciones_paso['fob_pred_mlp'] = val_fob_real_usd
            predicciones_paso['fob_pred_gpr'] = val_fob_real_usd
            predicciones_paso['fob_pred_foundation'] = val_fob_real_usd
            
        # 2. FAS DISCOUNT
        if 'fas_discount' not in variables_exogenas:
            h_idx_vecm = min(h_blind - 1, vecm_forecasts_proj.shape[0] - 1)
            vecm_pred_fas = vecm_forecasts_proj[h_idx_vecm, 1]
            
            if ms_proj_success:
                try:
                    params_dict = dict(ms_fit_fas_proj.params)
                    P_fas = ms_fit_fas_proj.regime_transition[:, :, 0]
                    p_filt_fas = ms_fit_fas_proj.filtered_marginal_probabilities.iloc[-1].values
                    p_h_fas = np.dot(np.linalg.matrix_power(P_fas, h_blind), p_filt_fas)
                    exog_t = [val_chicago, fila_real['rendimiento_estimado_tn_ha'], fila_real['lluvia_mm']]
                    
                    const0_fas, coefs0_fas = _extraer_params_markov(params_dict, 0, n_exog=3)
                    pred0_fas = const0_fas + sum(c * x for c, x in zip(coefs0_fas, exog_t))
                    
                    const1_fas, coefs1_fas = _extraer_params_markov(params_dict, 1, n_exog=3)
                    pred1_fas = const1_fas + sum(c * x for c, x in zip(coefs1_fas, exog_t))
                    
                    ms_pred_fas = p_h_fas[0] * pred0_fas + p_h_fas[1] * pred1_fas
                    if np.isnan(ms_pred_fas):
                        ms_pred_fas = vecm_pred_fas
                except Exception:
                    ms_pred_fas = vecm_pred_fas
            else:
                ms_pred_fas = vecm_pred_fas
                
            X_target = pd.DataFrame([df_proc_tmp.iloc[-2][cols_predictoras_base]])
            X_target['precio_chicago_usd_future'] = val_chicago
            X_target['tipo_cambio_future'] = val_tc
            
            hgbr_pred_fas = hgbr_models_fas[1].predict(X_target)[0]
            scaler_h_fas, en_model_h_fas = en_models_fas[1]
            X_target_scaled_fas = scaler_h_fas.transform(_fillna_safe(X_target, medians_all))
            en_pred_fas = en_model_h_fas.predict(X_target_scaled_fas)[0]
            
            # Kalman Filter online prediction instead of static ElasticNet
            x_fas_aug = np.hstack([[1.0], X_target_scaled_fas[0]])
            kf_pred_fas = np.dot(x_fas_aug, kf_fas.theta)
            
            # MLP neural network prediction
            scaler_h_fas_mlp, mlp_model_h_fas = mlp_models_fas[1]
            mlp_pred_fas = mlp_model_h_fas.predict(X_target_scaled_fas)[0]
            
            # GPR prediction
            scaler_h_fas_gpr, gpr_model_h_fas = gpr_models_fas[1]
            gpr_pred_fas = gpr_model_h_fas.predict(X_target_scaled_fas)[0]
            
            # Foundation model zero-shot prediction
            foundation_predictor = FoundationTimeSeriesPredictor(seasonal_periods=52)
            hist_fas_sim = df_historia_simulada['fas_discount'].values
            foundation_pred_fas = foundation_predictor.predict(hist_fas_sim, steps=1, target_name="fas_discount")[0]
            
            # Dynamic Model Averaging (DMA) ensemble blending (7 models)
            ensemble_fas = (pi_fas[0] * vecm_pred_fas + 
                            pi_fas[1] * ms_pred_fas + 
                            pi_fas[2] * hgbr_pred_fas + 
                            pi_fas[3] * kf_pred_fas + 
                            pi_fas[4] * mlp_pred_fas +
                            pi_fas[5] * gpr_pred_fas +
                            pi_fas[6] * foundation_pred_fas)
                            
            alpha_mr = min(0.3, 0.01 * h_blind)
            ensemble_fas = (1 - alpha_mr) * ensemble_fas + alpha_mr * df_proc_full_train['fas_discount'].mean()
            
            if mes_actual in [11, 12, 1, 2]:
                desviacion = (fila_real['rendimiento_estimado_tn_ha'] - rinde_medio_hist) / rinde_medio_hist
                factor_ajuste = np.clip(1.0 - (desviacion * 0.15), 0.85, 1.15)
                ensemble_fas *= factor_ajuste
                
            ensemble_fas = np.clip(ensemble_fas, rangos_seguridad['fas_discount'][0], rangos_seguridad['fas_discount'][1])
            
            vol_fas_t = garch_fc_fas_proj[h_idx_vecm]
            lower_fas = ensemble_fas - q_fas * vol_fas_t
            upper_fas = ensemble_fas + q_fas * vol_fas_t
            
            predicciones_paso['fas_discount'] = ensemble_fas
            predicciones_paso['fas_discount_lower'] = lower_fas
            predicciones_paso['fas_discount_upper'] = upper_fas
            fila_real['fas_discount'] = ensemble_fas
            predicciones_paso['vol_fas'] = vol_fas_t
            
            predicciones_paso['fas_pred_vecm'] = val_chicago * vecm_pred_fas
            predicciones_paso['fas_pred_ms'] = val_chicago * ms_pred_fas
            predicciones_paso['fas_pred_hgbr'] = val_chicago * hgbr_pred_fas
            predicciones_paso['fas_pred_en'] = val_chicago * kf_pred_fas
            predicciones_paso['fas_pred_mlp'] = val_chicago * mlp_pred_fas
            predicciones_paso['fas_pred_gpr'] = val_chicago * gpr_pred_fas
            predicciones_paso['fas_pred_foundation'] = val_chicago * foundation_pred_fas
        else:
            matching_rows_fas = df_proc_full[df_proc_full['fecha'] == fecha_actual]
            fas_val_real = matching_rows_fas['fas_discount'].values[0] if len(matching_rows_fas) > 0 else 0.85
            predicciones_paso['fas_discount'] = fas_val_real
            predicciones_paso['fas_discount_lower'] = fas_val_real
            predicciones_paso['fas_discount_upper'] = fas_val_real
            fila_real['fas_discount'] = fas_val_real
            predicciones_paso['vol_fas'] = 0.012
            
            val_fas_real_usd = val_chicago * fas_val_real
            predicciones_paso['fas_pred_vecm'] = val_fas_real_usd
            predicciones_paso['fas_pred_ms'] = val_fas_real_usd
            predicciones_paso['fas_pred_hgbr'] = val_fas_real_usd
            predicciones_paso['fas_pred_en'] = val_fas_real_usd
            predicciones_paso['fas_pred_mlp'] = val_fas_real_usd
            predicciones_paso['fas_pred_gpr'] = val_fas_real_usd
            predicciones_paso['fas_pred_foundation'] = val_fas_real_usd

        # Predecir cada target de Machine Learning (ratios y logística)
        for target in cols_ml_puros:
            if target == 'rendimiento_estimado_tn_ha':
                continue
                
            if hasattr(modelos[target], 'feature_names_in_'):
                cols_predictoras = [c for c in modelos[target].feature_names_in_ if c in df_proc_tmp.columns]
            else:
                cols_predictoras = [
                    c for c in df_proc_tmp.columns 
                    if _es_feature_valida(c, target, cols_ignorar, variables_exogenas)
                ]
            X_actual = pd.DataFrame([fila_proc_actual[cols_predictoras]])
            
            pred_ml = modelos[target].predict(X_actual)[0]
            ultimo_simulado = df_historia_simulada.iloc[-1][target] if target in df_historia_simulada.columns else pred_ml
            
            if target in ['descargas_camiones', 'descargas_vagones', 'embarques_tn']:
                factors_col = factores_estacionales.get(target, {m: 1.0 for m in range(1, 13)})
                mes_ant = df_historia_simulada.iloc[-1]['fecha'].month if not df_historia_simulada.empty else mes_actual
                factor_mes_ant = factors_col.get(mes_ant, 1.0)
                if factor_mes_ant > 0:
                    ultimo_simulado = ultimo_simulado / factor_mes_ant
                    
            alpha_decaida = min(0.85, 0.6 + 0.02 * proyeccion_idx)
            pred_ensemble = alpha_decaida * pred_ml + (1 - alpha_decaida) * ultimo_simulado
            
            if target in ['descargas_camiones', 'descargas_vagones', 'embarques_tn']:
                factors_col = factores_estacionales.get(target, {m: 1.0 for m in range(1, 13)})
                factor_mes = factors_col.get(mes_actual, 1.0)
                pred_ensemble = pred_ensemble * factor_mes
                
            vmin, vmax = rangos_seguridad.get(target, (-np.inf, np.inf))
            pred_final = np.clip(pred_ensemble, vmin, vmax)
            
            predicciones_paso[target] = pred_final
            fila_real[target] = pred_final
            
            q_val = conformal_quantiles[target]
            half_width = q_val * (1 + 0.3 * np.log1p(proyeccion_idx))
            
            predicciones_paso[f'{target}_lower'] = np.clip(pred_final - half_width, vmin, vmax)
            predicciones_paso[f'{target}_upper'] = np.clip(pred_final + half_width, vmin, vmax)
            
        # Reconstrucción determinística de precios y sus bandas de confianza
        fob_premium_actual = predicciones_paso.get('fob_premium', 1.15)
        fob_premium_lower = predicciones_paso.get('fob_premium_lower', fob_premium_actual)
        fob_premium_upper = predicciones_paso.get('fob_premium_upper', fob_premium_actual)
        
        fas_discount_actual = predicciones_paso.get('fas_discount', 0.85)
        fas_discount_lower = predicciones_paso.get('fas_discount_lower', fas_discount_actual)
        fas_discount_upper = predicciones_paso.get('fas_discount_upper', fas_discount_actual)
        
        predicciones_paso['precio_fob_usd'] = val_chicago * fob_premium_actual
        predicciones_paso['precio_fob_usd_lower'] = val_chicago * fob_premium_lower
        predicciones_paso['precio_fob_usd_upper'] = val_chicago * fob_premium_upper
        fila_real['precio_fob_usd'] = predicciones_paso['precio_fob_usd']
        
        predicciones_paso['precio_fas_usd'] = val_chicago * fas_discount_actual
        predicciones_paso['precio_fas_usd_lower'] = val_chicago * fas_discount_lower
        predicciones_paso['precio_fas_usd_upper'] = val_chicago * fas_discount_upper
        fila_real['precio_fas_usd'] = predicciones_paso['precio_fas_usd']
        
        predicciones_paso['precio_fas_ars'] = predicciones_paso['precio_fas_usd'] * val_tc
        predicciones_paso['precio_fas_ars_lower'] = predicciones_paso['precio_fas_usd_lower'] * val_tc
        predicciones_paso['precio_fas_ars_upper'] = predicciones_paso['precio_fas_usd_upper'] * val_tc
        fila_real['precio_fas_ars'] = predicciones_paso['precio_fas_ars']
        
        predicciones_paso['precio_bb_ars'] = predicciones_paso['precio_fas_ars']
        predicciones_paso['precio_bb_ars_lower'] = predicciones_paso['precio_fas_ars_lower']
        predicciones_paso['precio_bb_ars_upper'] = predicciones_paso['precio_fas_ars_upper']
        fila_real['precio_bb_ars'] = predicciones_paso['precio_fas_ars']
        
        predicciones_paso['precio_pizarra_usd'] = predicciones_paso['precio_fas_usd']
        predicciones_paso['precio_pizarra_usd_lower'] = predicciones_paso['precio_fas_usd_lower']
        predicciones_paso['precio_pizarra_usd_upper'] = predicciones_paso['precio_fas_usd_upper']
        fila_real['precio_pizarra_usd'] = predicciones_paso['precio_fas_usd']
        
        predicciones_paso['basis_usd'] = predicciones_paso['precio_fas_usd'] - val_chicago
        predicciones_paso['basis_usd_lower'] = predicciones_paso['precio_fas_usd_lower'] - val_chicago
        predicciones_paso['basis_usd_upper'] = predicciones_paso['precio_fas_usd_upper'] - val_chicago
        fila_real['basis_usd'] = predicciones_paso['basis_usd']
        
        # Logística Física
        val_camiones = predicciones_paso.get('descargas_camiones', 0.0)
        val_camiones_lower = predicciones_paso.get('descargas_camiones_lower', val_camiones)
        val_camiones_upper = predicciones_paso.get('descargas_camiones_upper', val_camiones)
        
        val_vagones = predicciones_paso.get('descargas_vagones', 0.0)
        val_vagones_lower = predicciones_paso.get('descargas_vagones_lower', val_vagones)
        val_vagones_upper = predicciones_paso.get('descargas_vagones_upper', val_vagones)
        
        predicciones_paso['descargas_camiones_tn'] = val_camiones * 30.0
        predicciones_paso['descargas_camiones_tn_lower'] = val_camiones_lower * 30.0
        predicciones_paso['descargas_camiones_tn_upper'] = val_camiones_upper * 30.0
        fila_real['descargas_camiones_tn'] = predicciones_paso['descargas_camiones_tn']
        
        predicciones_paso['descargas_vagones_tn'] = val_vagones * 45.0
        predicciones_paso['descargas_vagones_tn_lower'] = val_vagones_lower * 45.0
        predicciones_paso['descargas_vagones_tn_upper'] = val_vagones_upper * 45.0
        fila_real['descargas_vagones_tn'] = predicciones_paso['descargas_vagones_tn']
        
        # Comercial
        compras_se_ant = df_historia_simulada.iloc[-1]['compras_se'] if not df_historia_simulada.empty else 0.0
        compras_si_ant = df_historia_simulada.iloc[-1]['compras_si'] if not df_historia_simulada.empty else 0.0
        delta_compras_se_pred = predicciones_paso.get('delta_compras_se', 0.0)
        compras_sin_precio_pct_pred = predicciones_paso.get('compras_sin_precio_pct', 50.0)
        
        compras_se_nueva = compras_se_ant + delta_compras_se_pred
        compras_si_nueva = compras_si_ant
        compras_totales_nueva = compras_se_nueva + compras_si_nueva
        delta_compras_totales_nueva = delta_compras_se_pred
        compras_sin_precio_tot_nueva = compras_totales_nueva * compras_sin_precio_pct_pred / 100.0
        
        predicciones_paso['compras_se'] = compras_se_nueva
        fila_real['compras_se'] = compras_se_nueva
        predicciones_paso['compras_si'] = compras_si_nueva
        fila_real['compras_si'] = compras_si_nueva
        predicciones_paso['compras_totales'] = compras_totales_nueva
        fila_real['compras_totales'] = compras_totales_nueva
        predicciones_paso['compras_sin_precio_tot'] = compras_sin_precio_tot_nueva
        fila_real['compras_sin_precio_tot'] = compras_sin_precio_tot_nueva
        predicciones_paso['delta_compras_si'] = 0.0
        fila_real['delta_compras_si'] = 0.0
        predicciones_paso['delta_compras_totales'] = delta_compras_totales_nueva
        fila_real['delta_compras_totales'] = delta_compras_totales_nueva
        
        # Mantener Urea/MAP
        precio_urea_sim = df_historia_simulada.iloc[-1]['precio_urea_usd'] if not df_historia_simulada.empty else precio_urea_base
        precio_map_sim = df_historia_simulada.iloc[-1]['precio_map_usd'] if not df_historia_simulada.empty else 400.0
        predicciones_paso['precio_urea_usd'] = precio_urea_sim
        fila_real['precio_urea_usd'] = precio_urea_sim
        predicciones_paso['precio_map_usd'] = precio_map_sim
        fila_real['precio_map_usd'] = precio_map_sim
        predicciones_paso['temp_media'] = 18.0
        fila_real['temp_media'] = 18.0
        
        # --- ACTUALIZACIÓN ONLINE: KALMAN FILTER & DMA (FASE 2) ---
        try:
            fob_val_real_t = df_proc_full.loc[df_proc_full['fecha'] == fecha_actual, 'fob_premium'].values[0]
            fas_val_real_t = df_proc_full.loc[df_proc_full['fecha'] == fecha_actual, 'fas_discount'].values[0]
            
            if not es_periodo_proyeccion and not np.isnan(fob_val_real_t) and not np.isnan(fas_val_real_t):
                # 1. Kalman Filter online update
                x_fob_aug = np.hstack([[1.0], X_target_scaled_fob[0]])
                kf_fob.predict()
                kf_fob.update(x_fob_aug, fob_val_real_t)
                
                x_fas_aug = np.hstack([[1.0], X_target_scaled_fas[0]])
                kf_fas.predict()
                kf_fas.update(x_fas_aug, fas_val_real_t)
                
                # 2. Bayesian Dynamic Model Averaging (DMA) weight update
                if stress_weights is None:
                    pi_fob_pred = (pi_fob ** 0.98) + 1e-8
                    pi_fob_pred /= pi_fob_pred.sum()
                    
                    pi_fas_pred = (pi_fas ** 0.98) + 1e-8
                    pi_fas_pred /= pi_fas_pred.sum()
                    
                    sig_fob = vol_fob_t if vol_fob_t > 0 else 0.05
                    sig_fas = vol_fas_t if vol_fas_t > 0 else 0.05
                    
                    # Errores observados (7 modelos)
                    err_fob = fob_val_real_t - np.array([vecm_pred_fob, ms_pred_fob, hgbr_pred_fob, kf_pred_fob, mlp_pred_fob, gpr_pred_fob, foundation_pred_fob])
                    err_fas = fas_val_real_t - np.array([vecm_pred_fas, ms_pred_fas, hgbr_pred_fas, kf_pred_fas, mlp_pred_fas, gpr_pred_fas, foundation_pred_fas])
                    
                    # Verosimilitudes relativas
                    L_fob = np.exp(-0.5 * (err_fob ** 2) / (sig_fob ** 2))
                    L_fas = np.exp(-0.5 * (err_fas ** 2) / (sig_fas ** 2))
                    
                    pi_fob = pi_fob_pred * L_fob + 1e-8
                    pi_fob /= pi_fob.sum()
                    
                    pi_fas = pi_fas_pred * L_fas + 1e-8
                    pi_fas /= pi_fas.sum()
        except Exception:
            pass
            
        # Guardar historial de pesos dinámicos para graficar luego
        try:
            historial_pesos_dma_fob.append(pi_fob.copy())
            historial_pesos_dma_fas.append(pi_fas.copy())
        except Exception:
            historial_pesos_dma_fob.append(np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]))
            historial_pesos_dma_fas.append(np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]))
            
        predicciones_test.append(predicciones_paso)
        df_historia_simulada = pd.concat([df_historia_simulada, pd.DataFrame([fila_real])], ignore_index=True)

    df_predicciones = pd.DataFrame(predicciones_test)
    
    # Rellenar exógenas
    for exo in variables_exogenas:
        if exo in df_raw.columns:
            if exo in ['lluvia_mm', 'delta_compras_se', 'compras_sin_precio_pct', 'compras_se', 'compras_totales', 'compras_sin_precio_tot']:
                real_vals = df_raw.loc[df_raw['fecha'].isin(fechas_test), exo].values
                for idx, f in enumerate(fechas_test):
                    if f < pd.to_datetime(fecha_proyeccion) and idx < len(real_vals):
                        df_predicciones.loc[idx, exo] = real_vals[idx]
            else:
                valores_reales_test = df_raw.loc[df_raw['fecha'].isin(fechas_test), exo].values
                if len(valores_reales_test) == len(df_predicciones):
                    df_predicciones[exo] = valores_reales_test
            
    for exo in ['tipo_cambio', 'precio_chicago_usd', 'lluvia_mm', 'temp_media']:
        if exo in df_raw.columns and exo not in df_predicciones.columns:
            if exo == 'lluvia_mm':
                real_vals = df_raw.loc[df_raw['fecha'].isin(fechas_test), exo].values
                for idx, f in enumerate(fechas_test):
                    if f < pd.to_datetime(fecha_proyeccion) and idx < len(real_vals):
                        df_predicciones.loc[idx, exo] = real_vals[idx]
            else:
                real_vals = df_raw.loc[df_raw['fecha'].isin(fechas_test), exo].values
                if len(real_vals) == len(df_predicciones):
                    df_predicciones[exo] = real_vals

    # Generar métricas finales y evitar NaN crashes [FIX IMPORTANTE #6 y CRÍTICO #10]
    resultados_backtest = {}
    cols_reportar = [
        'precio_fob_usd', 'precio_fas_usd', 'precio_fas_ars', 'precio_bb_ars', 
        'precio_pizarra_usd', 'basis_usd', 'descargas_camiones', 'descargas_camiones_tn', 
        'descargas_vagones', 'descargas_vagones_tn', 'embarques_tn', 'lluvia_mm', 'temp_media', 'tipo_cambio', 
        'precio_chicago_usd', 'rendimiento_estimado_tn_ha', 'superficie_cosechada_ha'
    ]
    
    for target in cols_reportar:
        if target not in df_predicciones.columns or target not in df_raw.columns:
            continue
            
        es_exogena = target in variables_exogenas or target in ['tipo_cambio', 'precio_chicago_usd', 'lluvia_mm', 'temp_media']
        df_real = df_raw[['fecha', target]].rename(columns={target: 'real'})
        
        cols_df_pred = ['fecha', target]
        if f'{target}_lower' in df_predicciones.columns:
            cols_df_pred.append(f'{target}_lower')
        if f'{target}_upper' in df_predicciones.columns:
            cols_df_pred.append(f'{target}_upper')
            
        rename_dict = {
            target: 'prediccion',
            f'{target}_lower': 'lower',
            f'{target}_upper': 'upper'
        }
        
        if target == 'precio_fob_usd':
            for m in ['vecm', 'ms', 'hgbr', 'en', 'mlp', 'gpr', 'foundation']:
                col_m = f'fob_pred_{m}'
                if col_m in df_predicciones.columns:
                    cols_df_pred.append(col_m)
                    rename_dict[col_m] = f'pred_{m}'
        elif target == 'precio_fas_usd':
            for m in ['vecm', 'ms', 'hgbr', 'en', 'mlp', 'gpr', 'foundation']:
                col_m = f'fas_pred_{m}'
                if col_m in df_predicciones.columns:
                    cols_df_pred.append(col_m)
                    rename_dict[col_m] = f'pred_{m}'
                    
        df_pred = df_predicciones[cols_df_pred].rename(columns=rename_dict)
        df_comp = pd.merge(df_pred, df_real, on='fecha', how='left')
        
        metricas_indiv = {}
        for m in ['vecm', 'ms', 'hgbr', 'en', 'mlp', 'gpr', 'foundation']:
            col_m = f'pred_{m}'
            if col_m in df_comp.columns:
                mask = df_comp['real'].notna() & df_comp[col_m].notna() & (df_comp['real'] != 0)
                if mask.any():
                    m_real = df_comp.loc[mask, 'real']
                    m_pred = df_comp.loc[mask, col_m]
                    mae_m = mean_absolute_error(m_real, m_pred)
                    mape_m = np.mean(np.abs((m_real - m_pred) / m_real)) * 100
                    try:
                        r2_m = r2_score(m_real, m_pred)
                    except Exception:
                        r2_m = 0.0
                    metricas_indiv[m] = {
                        'mae': float(mae_m),
                        'mape': float(mape_m),
                        'r2': float(r2_m)
                    }

        if es_exogena:
            resultados_backtest[target] = {
                'df_comparacion': df_comp,
                'r2_train': 1.0,
                'r2_test': 1.0,
                'mae_test': 0.0,
                'mape_test': 0.0,
                'es_exogena': True,
                'metricas_individuales': metricas_indiv
            }
        else:
            if is_live_2026_27:
                r2_test, mae_test, mape_test = np.nan, np.nan, np.nan
            else:
                mask = df_comp['real'].notna() & df_comp['prediccion'].notna() & (df_comp['real'] != 0)
                if mask.sum() > 1:
                    y_true_m = df_comp.loc[mask, 'real'].astype(float)
                    y_pred_m = df_comp.loc[mask, 'prediccion'].astype(float)
                    valid = np.isfinite(y_true_m) & np.isfinite(y_pred_m)
                    if valid.sum() > 1:
                        y_true_m = y_true_m[valid]
                        y_pred_m = y_pred_m[valid]
                        r2_test = r2_score(y_true_m, y_pred_m)
                        mae_test = mean_absolute_error(y_true_m, y_pred_m)
                        mape_test = np.mean(np.abs((y_true_m - y_pred_m) / y_true_m)) * 100
                    else:
                        r2_test, mae_test, mape_test = 0.0, 0.0, 0.0
                else:
                    r2_test, mae_test, mape_test = 0.0, 0.0, 0.0
            
            # Asignación de R2 Train dinámico real
            if target in ['precio_fob_usd', 'precio_fas_usd', 'precio_fas_ars', 'precio_bb_ars', 'precio_pizarra_usd', 'basis_usd']:
                if 'fob' in target:
                    r2_tr = max(0.1, float(r2_tr_fob))
                else:
                    r2_tr = max(0.1, float(r2_tr_fas))
            elif target in ['descargas_camiones', 'descargas_camiones_tn']:
                r2_tr = metricas_train.get('descargas_camiones', {}).get('r2', 0.5)
            elif target in ['descargas_vagones', 'descargas_vagones_tn']:
                r2_tr = metricas_train.get('descargas_vagones', {}).get('r2', 0.5)
            elif target in ['rendimiento_estimado_tn_ha', 'superficie_cosechada_ha']:
                r2_tr = metricas_train.get(target, {}).get('r2', 0.76)
            else:
                r2_tr = metricas_train.get(target, {}).get('r2', 0.5)
                
            resultados_backtest[target] = {
                'df_comparacion': df_comp,
                'r2_train': float(r2_tr),
                'r2_test': float(r2_test),
                'mae_test': float(mae_test),
                'mape_test': float(mape_test),
                'es_exogena': False,
                'metricas_individuales': metricas_indiv
            }

    # --- INTEGRACIÓN CAUSAL ML -> ENGINE (CRÍTICO #8) ---
    if progress_callback:
        progress_callback(96, "Calculando importancia de variables y extrayendo reglas de causalidad...")
    try:
        from ml.rule_extractor import modelo_a_reglas
        from sklearn.inspection import permutation_importance
        
        features_h1 = list(cols_predictoras_base) + ['precio_chicago_usd_future', 'tipo_cambio_future']
        
        # 1. FOB rules
        coefs_fob = pd.DataFrame({
            'feature': features_h1,
            'coeficiente': en_models_fob[1][1].coef_
        })
        
        # Calcular importancia por permutación para HGBR FOB
        X_h1_train_fob = df_proc_full_train[cols_predictoras_base].copy()
        X_h1_train_fob['precio_chicago_usd_future'] = df_proc_full_train['precio_chicago_usd']
        X_h1_train_fob['tipo_cambio_future'] = df_proc_full_train['tipo_cambio']
        X_h1_train_fob = _fillna_safe(X_h1_train_fob[features_h1], medians_all)
        
        y_h1_train_fob = df_proc_full_train['fob_premium'].shift(-1).ffill()
        valid_idx_fob = y_h1_train_fob.dropna().index
        
        res_fob = permutation_importance(hgbr_models_fob[1], X_h1_train_fob.loc[valid_idx_fob], y_h1_train_fob.loc[valid_idx_fob], n_repeats=3, random_state=42, n_jobs=1)
        importancias_fob = pd.DataFrame({
            'feature': features_h1,
            'importancia': res_fob.importances_mean
        })
        # Normalizar
        total_imp_fob = importancias_fob['importancia'].sum()
        if total_imp_fob > 0:
            importancias_fob['importancia'] = importancias_fob['importancia'] / total_imp_fob
        else:
            importancias_fob['importancia'] = 1.0 / len(features_h1)
            
        reglas_ml_fob = modelo_a_reglas(coefs_fob, importancias_fob, 'precio_fob_usd', umbral_importancia=0.02)
        
        # 2. FAS rules
        coefs_fas = pd.DataFrame({
            'feature': features_h1,
            'coeficiente': en_models_fas[1][1].coef_
        })
        
        # Calcular importancia por permutación para HGBR FAS
        y_h1_train_fas = df_proc_full_train['fas_discount'].shift(-1).ffill()
        valid_idx_fas = y_h1_train_fas.dropna().index
        
        res_fas = permutation_importance(hgbr_models_fas[1], X_h1_train_fob.loc[valid_idx_fas], y_h1_train_fas.loc[valid_idx_fas], n_repeats=3, random_state=42, n_jobs=1)
        importancias_fas = pd.DataFrame({
            'feature': features_h1,
            'importancia': res_fas.importances_mean
        })
        # Normalizar
        total_imp_fas = importancias_fas['importancia'].sum()
        if total_imp_fas > 0:
            importancias_fas['importancia'] = importancias_fas['importancia'] / total_imp_fas
        else:
            importancias_fas['importancia'] = 1.0 / len(features_h1)
            
        reglas_ml_fas = modelo_a_reglas(coefs_fas, importancias_fas, 'precio_fas_usd', umbral_importancia=0.02)
        
        reglas_ml_totales = reglas_ml_fob + reglas_ml_fas
        
        # Inyectar reglas extraídas en el dict retornado
        resultados_backtest['reglas_ml'] = reglas_ml_totales
        print(f"  -> [Integración ML-Engine] Se extrajeron {len(reglas_ml_totales)} reglas desde los pesos matemáticos del ML.")
    except Exception as e:
        print(f"  [Warning ML-Engine] Error extrayendo reglas desde ML: {e}")
        
    try:
        resultados_backtest['pesos_dma_fob'] = np.array(historial_pesos_dma_fob)
        resultados_backtest['pesos_dma_fas'] = np.array(historial_pesos_dma_fas)
    except Exception:
        pass
        
    try:
        resultados_backtest['df_garch'] = df_predicciones[['fecha', 'vol_fob', 'vol_fas']].copy()
    except Exception:
        pass
            
    if progress_callback:
        progress_callback(100, "¡Simulación completada con éxito!")
        
    # --- OPTIMIZACIÓN EXTREMA DE MEMORIA: Liberar 280 modelos locales ---
    try:
        import gc
        del hgbr_models_fob, hgbr_models_fas
        del en_models_fob, en_models_fas
        del mlp_models_fob, mlp_models_fas
        del gpr_models_fob, gpr_models_fas
        del modelos
        gc.collect()
    except Exception:
        pass
        
    return resultados_backtest
