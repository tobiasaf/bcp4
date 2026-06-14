import numpy as np
import pandas as pd
import copy
from typing import Dict, List, Tuple
from .simulation import MotorSimulacion
from .models import SnapshotEstado

def correr_montecarlo(
    motor_base: MotorSimulacion, 
    variables_variacion: Dict[str, Tuple[float, float]], # ej: {'lluvia_mm': (media, std_dev)}
    dias: int,
    n_simulaciones: int = 100,
    seed: int = 42 # [FIX IMPORTANTE #9 - Semilla fija para reproducibilidad]
) -> Dict[str, pd.DataFrame]:
    """
    Corre N simulaciones variando aleatoriamente las condiciones iniciales.
    Utiliza np.random.default_rng(seed) para reproducibilidad total.
    """
    todas_trayectorias = []
    rng = np.random.default_rng(seed=seed)
    
    for i in range(n_simulaciones):
        # 1. Perturbar estado inicial de forma determinista y acotada
        estado_perturbado = motor_base.estado_inicial.copy()
        for var, (media, std) in variables_variacion.items():
            if var in estado_perturbado:
                estado_perturbado[var] = rng.normal(media, std)
                # Evitar precios o valores logísticos negativos absurdos
                if estado_perturbado[var] < 0 and media > 0:
                    estado_perturbado[var] = max(0.0, estado_perturbado[var])
        
        # 2. Copia profunda de las reglas (ya que tienen estado mutable de disparo)
        reglas_copia = copy.deepcopy(motor_base.reglas)
        motor = MotorSimulacion(
            estado_inicial=estado_perturbado,
            reglas=reglas_copia,
            variables_truncamiento=motor_base.variables_truncamiento
        )
        
        # 3. Correr
        historial = motor.correr(dias=dias)
        
        # 4. Extraer serie de tiempo
        df_trayectoria = pd.DataFrame([s.valores for s in historial])
        todas_trayectorias.append(df_trayectoria)
        
    # Calcular percentiles
    resultados_percentiles = {}
    variables = todas_trayectorias[0].columns
    
    for var in variables:
        matriz_var = np.column_stack([t[var].values for t in todas_trayectorias])
        
        df_percentiles = pd.DataFrame({
            'p5': np.percentile(matriz_var, 5, axis=1),
            'p25': np.percentile(matriz_var, 25, axis=1),
            'p50': np.percentile(matriz_var, 50, axis=1),
            'p75': np.percentile(matriz_var, 75, axis=1),
            'p95': np.percentile(matriz_var, 95, axis=1),
            'mean': np.mean(matriz_var, axis=1)
        })
        resultados_percentiles[var] = df_percentiles
        
    return resultados_percentiles
