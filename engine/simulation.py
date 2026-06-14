from typing import Dict, List, Optional
from .models import Regla, EventoProgramado, SnapshotEstado
from .impact_curves import generar_curva_impacto

class MotorSimulacion:
    def __init__(self, estado_inicial: Dict[str, float], reglas: List[Regla], variables_truncamiento: Optional[Dict[str, tuple]] = None):
        """
        estado_inicial: dict con las variables y sus valores iniciales
        reglas: lista de objetos Regla
        variables_truncamiento: dict donde key es nombre variable y value es tupla (min, max)
        """
        self.estado_inicial = estado_inicial.copy()
        self.reglas = reglas
        self.variables_truncamiento = variables_truncamiento or {}
        
    def _aplicar_truncamiento(self, estado: Dict[str, float]):
        for var, (min_val, max_val) in self.variables_truncamiento.items():
            if var in estado:
                if min_val is not None:
                    estado[var] = max(min_val, estado[var])
                if max_val is not None:
                    estado[var] = min(max_val, estado[var])
                    
    def correr(self, dias: int, eventos_manuales: Optional[List[EventoProgramado]] = None, datos_futuros_conocidos: Optional[Dict[str, List[float]]] = None) -> List[SnapshotEstado]:
        estado = self.estado_inicial.copy()
        cola_eventos: List[EventoProgramado] = eventos_manuales.copy() if eventos_manuales else []
        datos_futuros = datos_futuros_conocidos or {}
        historial: List[SnapshotEstado] = []
        
        # Resetear estado de las reglas antes de empezar
        for regla in self.reglas:
            regla.resetear_estado()
            
        for dia in range(dias):
            reglas_disparadas_hoy = []
            eventos_ejecutados_hoy = []
            
            # 0. Inyectar valores de variables exógenas (datos futuros conocidos)
            for var, curva in datos_futuros.items():
                if dia < len(curva):
                    estado[var] = curva[dia]
                    
            # 1. Ejecutar eventos programados para HOY
            eventos_hoy = [e for e in cola_eventos if e.dia_ejecucion == dia]
            for evento in eventos_hoy:
                if evento.variable in estado:
                    estado[evento.variable] += evento.impacto
                    eventos_ejecutados_hoy.append(evento.origen)
                cola_eventos.remove(evento)
                
            # 2. Aplicar truncamientos (ej: precios no pueden ser negativos)
            self._aplicar_truncamiento(estado)
            
            # 3. Evaluar reglas para generar eventos futuros
            for regla in self.reglas:
                if not regla.activa:
                    continue
                    
                condicion_cumplida = regla.condicion(estado)
                
                if condicion_cumplida and not regla._fue_disparada_recientemente:
                    # La regla se dispara!
                    impacto_total = regla.calcular_impacto_total(estado)
                    deltas = generar_curva_impacto(impacto_total, regla.duracion_efecto_dias, regla.tipo_curva.value)
                    
                    # Generar los eventos programados a lo largo del tiempo
                    for i, delta in enumerate(deltas):
                        if abs(delta) > 1e-6: # ignorar impactos ínfimos
                            cola_eventos.append(EventoProgramado(
                                dia_ejecucion=dia + regla.retardo_inicio_dias + i,
                                variable=regla.variable_afectada,
                                impacto=delta,
                                origen=regla.nombre
                            ))
                            
                    regla._fue_disparada_recientemente = True
                    reglas_disparadas_hoy.append(regla.nombre)
                    
                elif not condicion_cumplida:
                    # Si la condición dejó de cumplirse, reseteamos el flag para que pueda volver a dispararse en el futuro
                    regla._fue_disparada_recientemente = False
                    
            # 4. Ejecutar eventos que fueron recién programados para HOY (retardo=0)
            #    Esto es necesario porque el paso 1 ya procesó los eventos previos del día,
            #    pero las reglas del paso 3 pueden haber creado nuevos eventos para el día actual.
            eventos_nuevos_hoy = [e for e in cola_eventos if e.dia_ejecucion == dia]
            for evento in eventos_nuevos_hoy:
                if evento.variable in estado:
                    estado[evento.variable] += evento.impacto
                    eventos_ejecutados_hoy.append(evento.origen)
                cola_eventos.remove(evento)
            
            # 5. Aplicar truncamientos de nuevo después de los eventos inmediatos
            if eventos_nuevos_hoy:
                self._aplicar_truncamiento(estado)
            
            # 6. Guardar snapshot
            historial.append(SnapshotEstado(
                dia=dia,
                valores=estado.copy(),
                reglas_disparadas=reglas_disparadas_hoy,
                eventos_ejecutados=eventos_ejecutados_hoy,
                eventos_pendientes=len(cola_eventos)
            ))
            
        return historial
