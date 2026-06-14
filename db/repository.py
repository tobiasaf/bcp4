import sqlite3
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from engine.models import Regla

DB_PATH = "data/bcp_models.db"

def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Inicializa la tabla de modelos si no existe."""
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS modelos_desplegados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_despliegue TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            estado_inicial_json TEXT,
            reglas_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def guardar_modelo_desplegado(estado_inicial: Dict[str, float], reglas: List[Regla]):
    """Guarda el estado base y el set de reglas activo como el último modelo de producción."""
    init_db()
    conn = _get_connection()
    c = conn.cursor()
    
    estado_json = json.dumps(estado_inicial)
    reglas_json = json.dumps([r.to_dict() for r in reglas])
    
    c.execute('''
        INSERT INTO modelos_desplegados (estado_inicial_json, reglas_json)
        VALUES (?, ?)
    ''', (estado_json, reglas_json))
    
    conn.commit()
    conn.close()

def cargar_ultimo_modelo() -> Tuple[Optional[Dict[str, float]], Optional[List[Regla]]]:
    """Carga el último modelo desplegado para usar en el simulador."""
    init_db()
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT estado_inicial_json, reglas_json 
        FROM modelos_desplegados 
        ORDER BY fecha_despliegue DESC LIMIT 1
    ''')
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None, None
        
    estado_inicial = json.loads(row[0])
    reglas_dicts = json.loads(row[1])
    reglas = [Regla.from_dict(d) for d in reglas_dicts]
    
    return estado_inicial, reglas
