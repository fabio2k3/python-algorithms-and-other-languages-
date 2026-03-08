"""
Modelo de un Servidor Simple
==============================
Sistema de servicio elemental: una cola + un servidor (M/G/1 genérico).

Dinámica:
  - Clientes arriban según una distribución de interarribo.
  - Si el servidor está vacío → atendido de inmediato.
  - Si está ocupado → entra en la cola (FIFO).
  - El sistema opera hasta tiempo T; tras T no entran más clientes,
    pero se termina de atender a los que ya están en cola.

Variables del sistema (según el resumen de conferencias):
  t    : tiempo general de simulación
  tA   : tiempo programado para el próximo arribo
  tD   : tiempo programado para la próxima salida (∞ si servidor vacío)
  NA   : total de arribos ocurridos
  ND   : total de partidas ocurridas
  n    : número actual de clientes en el sistema (cola + servidor)
  A    : diccionario  {arribo_id: tiempo_de_arribo}
  D    : diccionario  {partida_id: tiempo_de_partida}

Métricas recogidas:
  - Tiempo medio de espera en cola (W_q)
  - Tiempo medio en sistema (W)
  - Longitud media de la cola (L_q)  vía integración por tramos
  - Utilización del servidor (ρ)
  - Total de clientes atendidos (ND)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ── Tipos ─────────────────────────────────────────────────────────────────────

GenTime = Callable[[random.Random], float]   # generador de tiempos


@dataclass
class ResultadoServidorSimple:
    """Resultados de una corrida del modelo de servidor simple."""
    NA: int                      # total de arribos
    ND: int                      # total de partidas (clientes atendidos)
    tiempo_sim: float            # tiempo final de simulación

    A: Dict[int, float]          # {id_arribo: t_arribo}
    D: Dict[int, float]          # {id_partida: t_partida}

    tiempos_espera_cola: List[float]   # W_q por cliente
    tiempos_en_sistema: List[float]    # W   por cliente

    area_n: float                # ∫ n(t) dt  (para Lq y L)
    tiempo_servidor_ocupado: float

    @property
    def W_q(self) -> float:
        """Tiempo medio de espera en cola."""
        return sum(self.tiempos_espera_cola) / max(self.ND, 1)

    @property
    def W(self) -> float:
        """Tiempo medio en sistema."""
        return sum(self.tiempos_en_sistema) / max(self.ND, 1)

    @property
    def L(self) -> float:
        """Número medio de clientes en sistema (Ley de Little: L = λ·W)."""
        return self.area_n / max(self.tiempo_sim, 1e-12)

    @property
    def rho(self) -> float:
        """Utilización del servidor."""
        return self.tiempo_servidor_ocupado / max(self.tiempo_sim, 1e-12)

    @property
    def throughput(self) -> float:
        """Tasa efectiva de salidas."""
        return self.ND / max(self.tiempo_sim, 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def simular_servidor_simple(
    gen_interarribo: GenTime,
    gen_servicio: GenTime,
    T: float,
    seed: int = None,
) -> ResultadoServidorSimple:
    """
    Simula el modelo de un servidor simple hasta el tiempo T.

    Args:
        gen_interarribo: función rng → tiempo entre llegadas (> 0)
        gen_servicio:    función rng → tiempo de servicio   (> 0)
        T:               horizonte de simulación
        seed:            semilla aleatoria

    Returns:
        ResultadoServidorSimple con todas las métricas.
    """
    rng = random.Random(seed)

    # ── Inicialización (paso 1 del algoritmo del resumen) ──────────────────
    t  = 0.0
    NA = 0
    ND = 0
    n  = 0

    A: Dict[int, float] = {}
    D: Dict[int, float] = {}

    tA = gen_interarribo(rng)       # primer arribo
    tD = math.inf                   # sin salida programada

    # Métricas acumuladas
    tiempos_arribo_sistema: Dict[int, float] = {}   # id → t de entrada al sistema
    tiempos_inicio_servicio: Dict[int, float] = {}  # id → t de inicio de servicio
    tiempos_espera_cola: List[float] = []
    tiempos_en_sistema:  List[float] = []
    area_n = 0.0                   # ∫ n(t) dt
    t_last = 0.0                   # último tiempo en que n cambió
    t_servidor_ocupado = 0.0
    t_inicio_servicio_actual = 0.0

    # Cola explícita para rastrear el orden de servicio
    cola: List[int] = []           # ids de clientes esperando

    # ── Bucle principal de eventos ─────────────────────────────────────────
    while True:
        prox = min(tA, tD)

        # ── Evento de arribo: tA ≤ tD y tA ≤ T ────────────────────────────
        if tA <= tD and tA <= T:
            # Acumular área antes de cambiar n
            area_n += n * (tA - t_last)
            t_last  = tA

            t   = tA
            NA += 1
            n  += 1

            A[NA] = t
            tiempos_arribo_sistema[NA] = t

            # Programar próximo arribo
            tA = t + gen_interarribo(rng)

            if n == 1:
                # Servidor vacío → comenzar servicio de inmediato
                tiempos_inicio_servicio[NA] = t
                tiempos_espera_cola.append(0.0)
                t_inicio_servicio_actual = t
                tD = t + gen_servicio(rng)
            else:
                # Servidor ocupado → entrar en cola
                cola.append(NA)

        # ── Evento de salida: tD < tA y tD ≤ T ────────────────────────────
        elif tD < tA and tD <= T:
            area_n += n * (tD - t_last)
            t_last  = tD

            t   = tD
            ND += 1
            n  -= 1

            # Registrar tiempo en sistema del cliente que sale
            # (el cliente que está siendo atendido es el primero que llegó)
            # Buscamos el cliente con menor tA que aún no ha partido
            cliente_sale = _buscar_cliente_en_servicio(
                tiempos_arribo_sistema, tiempos_inicio_servicio, ND
            )
            D[ND] = t
            if cliente_sale in tiempos_arribo_sistema:
                w = t - tiempos_arribo_sistema.pop(cliente_sale)
                tiempos_en_sistema.append(w)
            if cliente_sale in tiempos_inicio_servicio:
                del tiempos_inicio_servicio[cliente_sale]

            t_servidor_ocupado += t - t_inicio_servicio_actual

            if n == 0:
                tD = math.inf
            else:
                # Atender al siguiente en cola
                siguiente = cola.pop(0)
                tiempos_inicio_servicio[siguiente] = t
                espera = t - tiempos_arribo_sistema[siguiente]
                tiempos_espera_cola.append(max(espera, 0.0))
                t_inicio_servicio_actual = t
                tD = t + gen_servicio(rng)

        # ── Cierre: arribo fuera de tiempo ────────────────────────────────
        elif tA > T and tD == math.inf:
            break   # servidor vacío y no hay más clientes → fin

        # ── Cierre: salida tras tiempo T ───────────────────────────────────
        elif tA > T and tD <= math.inf and n > 0:
            area_n += n * (tD - t_last)
            t_last  = tD

            t   = tD
            ND += 1
            n  -= 1

            cliente_sale = _buscar_cliente_en_servicio(
                tiempos_arribo_sistema, tiempos_inicio_servicio, ND
            )
            D[ND] = t
            if cliente_sale in tiempos_arribo_sistema:
                w = t - tiempos_arribo_sistema.pop(cliente_sale)
                tiempos_en_sistema.append(w)
            if cliente_sale in tiempos_inicio_servicio:
                del tiempos_inicio_servicio[cliente_sale]

            t_servidor_ocupado += t - t_inicio_servicio_actual

            if n == 0:
                tD = math.inf
                break
            else:
                siguiente = cola.pop(0)
                tiempos_inicio_servicio[siguiente] = t
                espera = t - tiempos_arribo_sistema[siguiente]
                tiempos_espera_cola.append(max(espera, 0.0))
                t_inicio_servicio_actual = t
                tD = t + gen_servicio(rng)
        else:
            break

    return ResultadoServidorSimple(
        NA=NA,
        ND=ND,
        tiempo_sim=t,
        A=A,
        D=D,
        tiempos_espera_cola=tiempos_espera_cola,
        tiempos_en_sistema=tiempos_en_sistema,
        area_n=area_n,
        tiempo_servidor_ocupado=t_servidor_ocupado,
    )


def _buscar_cliente_en_servicio(
    tiempos_arribo: Dict[int, float],
    tiempos_inicio: Dict[int, float],
    nd: int,
) -> int:
    """Devuelve el id del cliente actualmente en servicio (el que llegó primero y tiene inicio de servicio)."""
    if tiempos_inicio:
        return min(tiempos_inicio.keys())
    if tiempos_arribo:
        return min(tiempos_arribo.keys())
    return nd


# ── Generadores de tiempo exponencial ─────────────────────────────────────────

def gen_exp(lam: float) -> GenTime:
    """Devuelve un generador de tiempos ~ Exp(λ)."""
    def _gen(rng: random.Random) -> float:
        u = rng.random()
        while u == 0:
            u = rng.random()
        return -math.log(u) / lam
    return _gen


# ── Replicaciones ─────────────────────────────────────────────────────────────

def replicar_servidor_simple(
    gen_interarribo: GenTime,
    gen_servicio: GenTime,
    T: float,
    r: int,
    seed: int = None,
) -> List[ResultadoServidorSimple]:
    """Ejecuta r replicaciones independientes."""
    base = random.Random(seed)
    return [
        simular_servidor_simple(
            gen_interarribo, gen_servicio, T,
            seed=base.randint(0, 2**31)
        )
        for _ in range(r)
    ]


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LAM_A = 3.0   # tasa de arribo  λ_a
    LAM_S = 4.0   # tasa de servicio λ_s
    T     = 100.0
    R     = 1_000

    # Para M/M/1: ρ = λ/μ, W = 1/(μ-λ), W_q = ρ/(μ-λ)
    rho_teo = LAM_A / LAM_S
    W_teo   = 1 / (LAM_S - LAM_A)
    Wq_teo  = rho_teo / (LAM_S - LAM_A)

    resultados = replicar_servidor_simple(
        gen_exp(LAM_A), gen_exp(LAM_S), T, R, seed=42
    )

    W_emp  = sum(r.W   for r in resultados) / R
    Wq_emp = sum(r.W_q for r in resultados) / R
    rho_emp= sum(r.rho for r in resultados) / R

    print("=== MODELO DE UN SERVIDOR SIMPLE (M/M/1) ===")
    print(f"λ={LAM_A}, μ={LAM_S}, T={T}, {R} réplicas\n")
    print(f"{'Métrica':<25} {'Teórico':>12} {'Empírico':>12}")
    print("-" * 52)
    print(f"{'ρ (utilización)':<25} {rho_teo:>12.4f} {rho_emp:>12.4f}")
    print(f"{'W (tiempo sistema)':<25} {W_teo:>12.4f} {W_emp:>12.4f}")
    print(f"{'W_q (espera cola)':<25} {Wq_teo:>12.4f} {Wq_emp:>12.4f}")
