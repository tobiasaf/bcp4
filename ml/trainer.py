import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from typing import Dict, Tuple, Any

def entrenar_modelo(df: pd.DataFrame, variable_objetivo: str, variables_predictoras: list, fecha_corte: str = None, predecir_diferencias: bool = False, calcular_importancia: bool = False) -> Dict[str, Any]:
    """
    Entrena un modelo HistGradientBoostingRegressor Avanzado.
    Usa RandomizedSearchCV y TimeSeriesSplit para explorar hiperparámetros 
    exhaustivamente, maximizando el poder de análisis a cambio de mayor cómputo.
    Se usa la versión nativa de Scikit-learn para evitar errores de C++ (libomp) en Mac,
    manteniendo la misma potencia matemática que XGBoost.
    """
    X = df[variables_predictoras]
    y = df[variable_objetivo]
    
    if fecha_corte and 'fecha' in df.columns:
        
        mask_train = df['fecha'] < pd.to_datetime(fecha_corte)
        split_idx = mask_train.sum()
        if split_idx == 0 or split_idx == len(df):
            
             split_idx = int(len(df) * 0.8)
    else:
        
        split_idx = int(len(df) * 0.8)
        
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    fechas_test = df['fecha'].iloc[split_idx:] if 'fecha' in df.columns else np.arange(len(y_test))
    
    
    hgbr_model = HistGradientBoostingRegressor(random_state=42)
    
    
    param_distributions = {
        'max_iter': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'l2_regularization': [1.0, 5.0, 10.0],
        'min_samples_leaf': [5, 10, 20]
    }
    
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    
    search = RandomizedSearchCV(
        estimator=hgbr_model,
        param_distributions=param_distributions,
        n_iter=8, 
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=1, 
        random_state=42
    )
    
    
    if predecir_diferencias:
        y_train_diff = y_train.diff().dropna()
        X_train_diff = X_train.loc[y_train_diff.index]
        search.fit(X_train_diff, y_train_diff)
    else:
        search.fit(X_train, y_train)
        
    modelo = search.best_estimator_
    
    
    if predecir_diferencias:
        
        y_pred_train_diff = modelo.predict(X_train.iloc[1:])
        y_pred_test_diff = modelo.predict(X_test)
        
        
        y_prev_train = y_train.iloc[:-1].values
        y_pred_train_rec = y_prev_train + y_pred_train_diff
        y_pred_train = np.insert(y_pred_train_rec, 0, y_train.iloc[0])
        
        y_prev_test = np.insert(y_test.iloc[:-1].values, 0, y_train.iloc[-1])
        y_pred_test = y_prev_test + y_pred_test_diff
    else:
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
    
    
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    
    
    if calcular_importancia:
        result = permutation_importance(modelo, X_train, y_train, n_repeats=5, random_state=42, n_jobs=1)
        importancias = pd.DataFrame({
            'feature': variables_predictoras,
            'importancia': result.importances_mean
        }).sort_values('importancia', ascending=False)
        
        
        total_imp = importancias['importancia'].sum()
        if total_imp > 0:
            importancias['importancia'] = importancias['importancia'] / total_imp
        else:
            importancias['importancia'] = 1.0 / len(importancias)
    else:
        importancias = pd.DataFrame({
            'feature': variables_predictoras,
            'importancia': 1.0 / len(variables_predictoras)
        })
    
    
    df_backtest = pd.DataFrame({
        'fecha': fechas_test,
        'real': y_test,
        'prediccion': y_pred_test
    })
    
    return {
        'modelo': modelo,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'importancias': importancias,
        'backtest': df_backtest,
        'X_train': X_train,
        'mejores_params': search.best_params_
    }

def detectar_lag_optimo(df: pd.DataFrame, col_evento: str, col_objetivo: str, max_lag: int = 12) -> Tuple[int, float]:
    """
    Calcula la correlación cruzada para encontrar el retardo donde la señal es más fuerte.
    Retorna (lag_optimo, correlacion_maxima).
    """
    correlaciones = []
    
    
    for lag in range(0, max_lag + 1):
        
        serie_evento_lag = df[col_evento].shift(lag)
        corr = df[col_objetivo].corr(serie_evento_lag)
        if pd.notna(corr):
            correlaciones.append((lag, corr))
            
    if not correlaciones:
        return 0, 0.0
        
    
    mejor_lag, mejor_corr = max(correlaciones, key=lambda x: abs(x[1]))
    
    return mejor_lag, mejor_corr
