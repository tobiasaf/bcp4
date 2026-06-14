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
                    
    def correr(self, dias: int, escenario: str = "Ninguno (Mercado Neutral)", eventos_manuales: Optional[List[EventoProgramado]] = None, datos_futuros_conocidos: Optional[Dict[str, List[float]]] = None) -> List[SnapshotEstado]:
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
                    
            # --- CÁLCULO DE VARIABLES ENDÓGENAS ---
            
            # 1. Rendimiento Estimado (tn/ha)
            lluvia = estado.get('lluvia_mm', 10.0)
            temp = estado.get('temp_media', 18.0)
            rinde = estado.get('rendimiento_estimado_tn_ha', 2.8)
            
            # Si el escenario es Sequía Severa, reducimos progresivamente el rendimiento
            if "sequía" in escenario.lower() or "niña" in escenario.lower():
                rinde -= 0.012  # Causa una pérdida progresiva de rendimiento acumulada
            else:
                if lluvia < 8.0:
                    rinde -= 0.005 * (8.0 - lluvia)
                elif lluvia > 18.0:
                    rinde += 0.001 * (lluvia - 18.0)
                
            if temp > 30.0:
                rinde -= 0.004 * (temp - 30.0)
                
            # Truncar rinde a límites lógicos
            rinde = max(0.5, min(4.8, rinde))
            estado['rendimiento_estimado_tn_ha'] = rinde
            
            # 2. Camiones Diarios (Logística)
            # Estacionalidad de la cosecha de trigo (pico entre día 35 y 75 de la simulación)
            base_camiones = self.estado_inicial.get('descargas_camiones', 675.0)
            factor_cosecha = 0.0
            if 35 <= dia <= 75:
                import math
                factor_cosecha = math.sin(math.pi * (dia - 35) / 40.0)
                
            camiones = base_camiones + 450.0 * (rinde / 2.8) * factor_cosecha
            
            # Shocks aplicados a camiones
            if "paro" in escenario.lower() or "camioneros" in escenario.lower():
                # Paro de camioneros: dura 7 días desde el día 15
                if 15 <= dia < 22:
                    camiones = camiones * 0.20  # Cae 80%
            
            if "bajante" in escenario.lower() or "paraná" in escenario.lower():
                camiones = camiones * 1.25  # Aumenta 25% por desvío de puertos
                
            if lluvia > 25.0:
                camiones = camiones * 0.70  # Lluvia excesiva frena la logística
                
            camiones = max(50.0, camiones)
            estado['descargas_camiones'] = camiones
            estado['descargas_camiones_tn'] = camiones * 30.0  # 30 toneladas promedio por camión
            
            # 3. Precios en Dólares (FAS, Futuros, Chicago, FOB)
            precio_fas = estado.get('precio_fas_usd', 175.0)
            precio_futuro = estado.get('precio_futuro_usd', 215.0)
            precio_chicago = estado.get('precio_chicago_usd', 188.0)
            
            # Convergencia de base: al final de la campaña, el FAS físico y el Futuro convergen
            dias_restantes = max(1, dias - dia)
            paso_convergencia = (precio_futuro - precio_fas) / (dias_restantes * 1.2)
            precio_fas += paso_convergencia
            
            # Shocks a precios
            if "sequía" in escenario.lower() or "niña" in escenario.lower():
                # Sequía aumenta el futuro y Chicago por menor oferta global
                precio_futuro += 0.20
                precio_chicago += 0.22
                
            if "bajante" in escenario.lower() or "paraná" in escenario.lower():
                # Aumenta el FAS local en Bahía Blanca por demanda inmediata de completamiento
                precio_fas += 0.25
                
            if "paro" in escenario.lower() or "camioneros" in escenario.lower():
                if 15 <= dia < 22:
                    # El FAS baja temporalmente por falta de entrega de mercadería física
                    precio_fas -= 0.40
            
            # Límites lógicos para precios
            precio_fas = max(50.0, min(500.0, precio_fas))
            precio_futuro = max(50.0, min(500.0, precio_futuro))
            precio_chicago = max(50.0, min(500.0, precio_chicago))
            
            estado['precio_fas_usd'] = precio_fas
            estado['precio_futuro_usd'] = precio_futuro
            estado['precio_chicago_usd'] = precio_chicago
            estado['precio_fob_usd'] = precio_fas + 34.4  # FOB sigue al FAS con margen de exportación
            
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
