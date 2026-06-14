import numpy as np
from typing import List
from scipy.stats import norm

def generar_curva_impacto(impacto_total: float, duracion_dias: int, tipo_curva: str) -> List[float]:
    """
    Distribuye el impacto_total a lo largo de duracion_dias según el tipo_curva.
    Retorna una lista de floats de longitud duracion_dias que suman exactamente impacto_total.
    """
    if duracion_dias <= 1 or tipo_curva == "Instantáneo":
        return [impacto_total]
        
    dias = np.arange(duracion_dias)
    
    if tipo_curva == "Sigmoidal":
        # S-curve: logística centrada a la mitad
        # CDF de la distribución normal da forma de S
        midpoint = duracion_dias / 2.0
        std_dev = duracion_dias / 6.0 # 99% de la curva cae dentro del rango
        cdf = norm.cdf(dias, loc=midpoint, scale=std_dev)
        # Queremos los deltas (el aporte de cada día)
        # Añadimos un 0 al principio para el cálculo de diferencias
        cdf_shifted = np.concatenate(([0], cdf[:-1]))
        deltas = cdf - cdf_shifted
        
    elif tipo_curva == "Campana (Normal)":
        # PDF de distribución normal
        midpoint = duracion_dias / 2.0
        std_dev = duracion_dias / 6.0
        pdf = norm.pdf(dias, loc=midpoint, scale=std_dev)
        deltas = pdf
        
    elif tipo_curva == "Rampa con Meseta":
        # Distribución uniforme
        deltas = np.ones(duracion_dias)
        
    elif tipo_curva == "Decaimiento Exponencial":
        # El efecto es fuerte al principio y decae
        decay_rate = 5.0 / duracion_dias
        deltas = np.exp(-decay_rate * dias)
        
    else: # Fallback a instantáneo
        return [impacto_total] + [0] * (duracion_dias - 1)
        
    # Normalizar para que la suma sea exactamente impacto_total
    suma_deltas = np.sum(deltas)
    if suma_deltas > 0:
        deltas = (deltas / suma_deltas) * impacto_total
    else:
        # Prevención de error por división por cero (no debería ocurrir)
        deltas = np.zeros(duracion_dias)
        deltas[0] = impacto_total
        
    return deltas.tolist()
