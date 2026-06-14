import pandas as pd
from typing import List, Dict
from engine.models import Regla, TipoCurva

def modelo_a_reglas(
    coeficientes: pd.DataFrame, 
    importancias: pd.DataFrame,
    variable_objetivo: str,
    umbral_importancia: float = 0.05
) -> List[Regla]:
    """
    Convierte los coeficientes del modelo sustituto en Reglas para el motor de simulación.
    Solo usa las variables que tienen una importancia mayor al umbral en el modelo original.
    """
    reglas = []
    
    
    df_merged = pd.merge(coeficientes, importancias, on='feature')
    
    
    features_relevantes = df_merged[df_merged['importancia'] >= umbral_importancia]
    
    for _, row in features_relevantes.iterrows():
        feature = row['feature']
        coef = row['coeficiente']
        importancia = row['importancia']
        
        
        if feature == 'intercept' or abs(coef) < 1e-6:
            continue
            
        
        
        retardo_dias = 0
        feature_base = feature
        if "_lag_" in feature:
            partes = feature.split("_lag_")
            feature_base = partes[0]
            try:
                retardo_semanas = int(partes[1])
                retardo_dias = retardo_semanas * 7
            except ValueError:
                pass
                
        
        
        
        
        
        
        
        
        def make_condicion(feat):
            
            
            return lambda estado: feat in estado
            
        def make_calcular_impacto(feat, coeficiente):
            
            
            
            
            return lambda estado: coeficiente * estado.get(feat, 0.0)
            
        nombre_regla = f"Efecto de {feature_base} en {variable_objetivo}"
        if retardo_dias > 0:
            nombre_regla += f" (Retardo: {retardo_dias} días)"
            
        
        tipo_curva = TipoCurva.SIGMOID
        if "lluvia" in feature_base or "temp" in feature_base:
            tipo_curva = TipoCurva.BELL
        elif "anomalia" in feature_base:
            tipo_curva = TipoCurva.RAMP
            
        regla = Regla(
            nombre=nombre_regla,
            condicion=make_condicion(feature_base),
            variable_afectada=variable_objetivo,
            calcular_impacto_total=make_calcular_impacto(feature_base, coef),
            retardo_inicio_dias=retardo_dias,
            duracion_efecto_dias=30, 
            tipo_curva=tipo_curva,
            origen="ML",
            feature_base=feature_base,
            coeficiente=coef
        )
        
        reglas.append(regla)
        
    return reglas
