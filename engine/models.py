from dataclasses import dataclass, field
from typing import Dict, Callable, List, Optional, Any
from enum import Enum
import re

class TipoCurva(Enum):
    SIGMOID = "Sigmoidal"
    BELL = "Campana (Normal)"
    RAMP = "Rampa con Meseta"
    EXPONENTIAL = "Decaimiento Exponencial"
    INSTANT = "Instantáneo"

@dataclass
class EventoProgramado:
    dia_ejecucion: int
    variable: str
    impacto: float
    origen: str  


def safe_eval_expression(expr_str: str) -> Callable[[dict], Any]:
    """
    Evalúa de forma segura una expresión simple mapeando strings a funciones de Python.
    Valida rigurosamente la cadena usando expresiones regulares para prevenir
    Code Injection e inyección de payloads peligrosos en el backend del Engine.
    """
    
    suspect_patterns = [r"__", r"import", r"eval", r"exec", r"globals", r"locals", r"class", r"base", r"subclasses", r"system", r"os"]
    for pattern in suspect_patterns:
        if re.search(pattern, expr_str):
            raise ValueError(f"Expresión sospechosa bloqueada por seguridad: '{expr_str}'")
            
    
    safe_chars_pattern = r"^[a-zA-Z0-9_\s\+\-\*\/\<\>\=\!\&\(\)\.\,\'\"\[\]\:]+$"
    if not re.match(safe_chars_pattern, expr_str):
        raise ValueError(f"Expresión contiene caracteres prohibidos: '{expr_str}'")
        
    
    clean_globals = {"__builtins__": {"float": float, "int": int, "abs": abs, "min": min, "max": max}}
    
    
    lambda_str = f"lambda estado: {expr_str}"
    return eval(lambda_str, clean_globals, {})


@dataclass
class Regla:
    nombre: str
    
    condicion: Callable[[Dict[str, float]], bool]
    variable_afectada: str
    
    calcular_impacto_total: Callable[[Dict[str, float]], float]
    retardo_inicio_dias: int
    duracion_efecto_dias: int
    tipo_curva: TipoCurva
    origen: str = "ML" 
    activa: bool = True
    
    
    condicion_str: Optional[str] = None
    impacto_str: Optional[str] = None
    feature_base: Optional[str] = None
    coeficiente: Optional[float] = None
    
    
    _fue_disparada_recientemente: bool = field(default=False, repr=False)
    
    def resetear_estado(self):
        self._fue_disparada_recientemente = False

    def to_dict(self):
        return {
            'nombre': self.nombre,
            'variable_afectada': self.variable_afectada,
            'retardo_inicio_dias': self.retardo_inicio_dias,
            'duracion_efecto_dias': self.duracion_efecto_dias,
            'tipo_curva': self.tipo_curva.name,
            'origen': self.origen,
            'activa': self.activa,
            'condicion_str': self.condicion_str,
            'impacto_str': self.impacto_str,
            'feature_base': self.feature_base,
            'coeficiente': self.coeficiente
        }

    @classmethod
    def from_dict(cls, data):
        tipo_curva = TipoCurva[data['tipo_curva']]
        
        
        condicion = None
        calcular_impacto_total = None
        
        if data.get('origen') == 'USUARIO' or data.get('condicion_str'):
            condicion_str = data.get('condicion_str', "True")
            impacto_str = data.get('impacto_str', "0.0")
            
            
            try:
                condicion = safe_eval_expression(condicion_str)
                calcular_impacto_total = safe_eval_expression(impacto_str)
            except Exception as e:
                
                print(f"[Engine Security Warning] Expresión bloqueada o inválida: {e}")
                condicion = lambda estado: True
                calcular_impacto_total = lambda estado: 0.0
        else:
            feat = data.get('feature_base')
            coef = data.get('coeficiente', 0.0)
            condicion = lambda estado, f=feat: f in estado
            calcular_impacto_total = lambda estado, f=feat, c=coef: c * estado.get(f, 0.0)
            
        return cls(
            nombre=data['nombre'],
            condicion=condicion,
            variable_afectada=data['variable_afectada'],
            calcular_impacto_total=calcular_impacto_total,
            retardo_inicio_dias=data['retardo_inicio_dias'],
            duracion_efecto_dias=data['duracion_efecto_dias'],
            tipo_curva=tipo_curva,
            origen=data.get('origen', 'ML'),
            activa=data.get('activa', True),
            condicion_str=data.get('condicion_str'),
            impacto_str=data.get('impacto_str'),
            feature_base=data.get('feature_base'),
            coeficiente=data.get('coeficiente')
        )

@dataclass
class SnapshotEstado:
    dia: int
    valores: Dict[str, float]
    reglas_disparadas: List[str]
    eventos_ejecutados: List[str]
    eventos_pendientes: int
