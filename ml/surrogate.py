import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, Any

def extraer_modelo_sustituto(modelo_complejo, X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Entrena un modelo lineal interpretable usando las predicciones del modelo complejo.
    """
    # Usar las predicciones del modelo complejo como variable objetivo para el sustituto
    y_sintetico = modelo_complejo.predict(X_train)
    
    # Entrenar regresión lineal (modelo interpretable)
    sustituto = LinearRegression()
    sustituto.fit(X_train, y_sintetico)
    
    # Evaluar qué tan bien el sustituto imita al modelo complejo
    y_pred_sustituto = sustituto.predict(X_train)
    fidelidad_r2 = r2_score(y_sintetico, y_pred_sustituto)
    
    # Extraer coeficientes
    coeficientes = pd.DataFrame({
        'feature': X_train.columns,
        'coeficiente': sustituto.coef_
    })
    
    # Generar ecuación legible
    ecuacion_partes = [f"{sustituto.intercept_:.2f}"]
    for _, row in coeficientes.iterrows():
        coef = row['coeficiente']
        feat = row['feature']
        if abs(coef) > 1e-4: # ignorar coeficientes casi nulos
            signo = "+" if coef > 0 else "-"
            ecuacion_partes.append(f"{signo} {abs(coef):.2f} × {feat}")
            
    ecuacion_str = " ≈ " + " ".join(ecuacion_partes)
    
    return {
        'modelo_sustituto': sustituto,
        'fidelidad_r2': fidelidad_r2,
        'coeficientes': coeficientes,
        'ecuacion_str': ecuacion_str,
        'intercept': sustituto.intercept_
    }
