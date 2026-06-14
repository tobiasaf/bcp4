import pandas as pd
import time
from ml.multi_predictor import entrenar_y_predecir_todo

print("Cargando datos...")
df_raw = pd.read_csv("data/historico_trigo_real.csv")
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

def progress(pct, msg):
    print(f"[{pct}%] {msg}")

start_time = time.time()
print("Iniciando simulación...")
res = entrenar_y_predecir_todo(
    df_raw,
    "2025-06-01",
    variables_exogenas=['tipo_cambio', 'precio_chicago_usd', 'lluvia_mm', 'temp_media'],
    predecir_diferencias=False,
    fecha_proyeccion="2025-06-01",
    progress_callback=progress
)
end_time = time.time()
print(f"Simulación completada en {end_time - start_time:.2f} segundos.")
