"""
Modelo de dos Servidores en Serie
====================================
Sistema con dos etapas consecutivas (S1 → S2).

Dinámica:
  - Clientes llegan y forman cola para S1.
  - Al terminar en S1, el cliente pasa directamente a S2
    (o espera si S2 está ocupado).
  - El cliente abandona el sistema al terminar en S2.
  - El sistema opera hasta T; tras T no entran más clientes
    pero se termina de atender a todos los presentes.

Variables del sistema (según el resumen de conferencias):
  tA   : tiempo del próximo arribo
  tD1  : tiempo de próxima salida de S1  (∞ si vacío)
  tD2  : tiempo de próxima salida de S2  (∞ si vacío)
  n1   : clientes en S1 (servidor + cola 1)
  n2   : clientes en S2 (servidor + cola 2)
  NA   : total de arribos
  ND   : total de partidas definitivas (salida de S2)
  A1   : {id: t_arribo_a_S1}
  A2   : {id: t_arribo_a_S2}

Métricas:
  - W1  : tiempo medio en S1 (espera + servicio)
  - W2  : tiempo medio en S2 (espera + servicio)
  - W   : tiempo total en sistema = W1 + W2
  - ρ1, ρ2 : utilización de cada servidor
  - throughput: clientes atendidos / tiempo
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


GenTime = Callable[[random.Random], float]


@dataclass
class ResultadoSerieDoble:
    NA: int
    ND: int
    tiempo_sim: float

    tiempos_en_S1: List[float]    # tiempo desde arribo a S1 hasta salida de S1
    tiempos_en_S2: List[float]    # tiempo desde arribo a S2 hasta salida de S2
    tiempos_totales: List[float]  # tiempo total en sistema

    t_ocupado_S1: float
    t_ocupado_S2: float

    @property
    def W1(self) -> float:
        return sum(self.tiempos_en_S1) / max(len(self.tiempos_en_S1), 1)

    @property
    def W2(self) -> float:
        return sum(self.tiempos_en_S2) / max(len(self.tiempos_en_S2), 1)

    @property
    def W(self) -> float:
        return sum(self.tiempos_totales) / max(self.ND, 1)

    @property
    def rho1(self) -> float:
        return self.t_ocupado_S1 / max(self.tiempo_sim, 1e-12)

    @property
    def rho2(self) -> float:
        return self.t_ocupado_S2 / max(self.tiempo_sim, 1e-12)

    @property
    def throughput(self) -> float:
        return self.ND / max(self.tiempo_sim, 1e-12)


def _gen_exp_sample(lam: float, rng: random.Random) -> float:
    u = rng.random()
    while u == 0:
        u = rng.random()
    return -math.log(u) / lam


def simular_serie_doble(
    gen_interarribo: GenTime,
    gen_servicio_S1: GenTime,
    gen_servicio_S2: GenTime,
    T: float,
    seed: int = None,
) -> ResultadoSerieDoble:
    """
    Simula el modelo de dos servidores en serie.

    Args:
        gen_interarribo : generador de tiempos entre llegadas
        gen_servicio_S1 : generador de tiempos de servicio en S1
        gen_servicio_S2 : generador de tiempos de servicio en S2
        T               : horizonte de simulación
        seed            : semilla aleatoria
    """
    rng = random.Random(seed)

    # ── Estado inicial ────────────────────────────────────────────────────
    t   = 0.0
    NA  = 0
    ND  = 0
    n1  = 0    # clientes en S1 (incluye el que está siendo atendido)
    n2  = 0    # clientes en S2

    tA  = gen_interarribo(rng)
    tD1 = math.inf
    tD2 = math.inf

    # Registros de tiempo
    A1: Dict[int, float] = {}   # id → t arribo a S1
    A2: Dict[int, float] = {}   # id → t arribo a S2

    cola1: List[int] = []       # cola explícita S1
    cola2: List[int] = []       # cola explícita S2
    en_servicio_S1: Optional[int] = None
    en_servicio_S2: Optional[int] = None
    t_inicio_s1: float = 0.0
    t_inicio_s2: float = 0.0

    tiempos_en_S1:    List[float] = []
    tiempos_en_S2:    List[float] = []
    tiempos_totales:  List[float] = []
    salidas_S1_t:     Dict[int, float] = {}  # id → t salida de S1

    t_ocupado_S1 = 0.0
    t_ocupado_S2 = 0.0

    def siguiente_evento():
        return min(tA, tD1, tD2)

    while True:
        ev = siguiente_evento()

        # ── EVENTO DE ARRIBO ──────────────────────────────────────────────
        if tA <= tD1 and tA <= tD2 and tA <= T:
            t   = tA
            NA += 1
            n1 += 1
            A1[NA] = t
            tA = t + gen_interarribo(rng)

            if n1 == 1:   # S1 vacío → servicio inmediato
                en_servicio_S1 = NA
                t_inicio_s1    = t
                tD1 = t + gen_servicio_S1(rng)
            else:
                cola1.append(NA)

        # ── EVENTO SALIDA S1 ──────────────────────────────────────────────
        elif tD1 <= tA and tD1 <= tD2 and (tD1 <= T or (tA > T and n1 > 0)):
            t   = tD1
            n1 -= 1
            n2 += 1

            cliente = en_servicio_S1
            t_ocupado_S1 += t - t_inicio_s1
            tiempos_en_S1.append(t - A1[cliente])
            salidas_S1_t[cliente] = t
            A2[cliente] = t   # llega a S2 en tiempo t

            if n1 > 0:
                siguiente_s1 = cola1.pop(0)
                en_servicio_S1 = siguiente_s1
                t_inicio_s1    = t
                tD1 = t + gen_servicio_S1(rng)
            else:
                en_servicio_S1 = None
                tD1 = math.inf

            if n2 == 1:   # S2 vacío → servicio inmediato
                en_servicio_S2 = cliente
                t_inicio_s2    = t
                tD2 = t + gen_servicio_S2(rng)
            else:
                cola2.append(cliente)

        # ── EVENTO SALIDA S2 ──────────────────────────────────────────────
        elif tD2 < tA and tD2 <= tD1 and (tD2 <= T or (tA > T and n2 > 0)):
            t   = tD2
            ND += 1
            n2 -= 1

            cliente = en_servicio_S2
            t_ocupado_S2 += t - t_inicio_s2
            tiempos_en_S2.append(t - A2[cliente])
            tiempos_totales.append(t - A1[cliente])

            if n2 > 0:
                siguiente_s2 = cola2.pop(0)
                en_servicio_S2 = siguiente_s2
                t_inicio_s2    = t
                tD2 = t + gen_servicio_S2(rng)
            else:
                en_servicio_S2 = None
                tD2 = math.inf

            if n1 == 0 and n2 == 0 and tA > T:
                break

        # ── Sin más eventos relevantes ────────────────────────────────────
        else:
            if tA > T and n1 == 0 and n2 == 0:
                break
            # avanzar tiempo al próximo evento de cierre
            if tA > T:
                if tD1 == math.inf and tD2 == math.inf:
                    break
            break

    return ResultadoSerieDoble(
        NA=NA,
        ND=ND,
        tiempo_sim=t,
        tiempos_en_S1=tiempos_en_S1,
        tiempos_en_S2=tiempos_en_S2,
        tiempos_totales=tiempos_totales,
        t_ocupado_S1=t_ocupado_S1,
        t_ocupado_S2=t_ocupado_S2,
    )


def gen_exp(lam: float) -> GenTime:
    def _g(rng: random.Random) -> float:
        u = rng.random()
        while u == 0:
            u = rng.random()
        return -math.log(u) / lam
    return _g


def replicar_serie_doble(
    gen_interarribo: GenTime,
    gen_s1: GenTime,
    gen_s2: GenTime,
    T: float,
    r: int,
    seed: int = None,
) -> List[ResultadoSerieDoble]:
    base = random.Random(seed)
    return [
        simular_serie_doble(gen_interarribo, gen_s1, gen_s2, T,
                            seed=base.randint(0, 2**31))
        for _ in range(r)
    ]


# ── Ejemplo de uso ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LAM_A  = 2.0
    MU_S1  = 3.0
    MU_S2  = 4.0
    T      = 200.0
    R      = 1_000

    resultados = replicar_serie_doble(
        gen_exp(LAM_A), gen_exp(MU_S1), gen_exp(MU_S2), T, R, seed=42
    )

    W1  = sum(r.W1  for r in resultados) / R
    W2  = sum(r.W2  for r in resultados) / R
    W   = sum(r.W   for r in resultados) / R
    rho1 = sum(r.rho1 for r in resultados) / R
    rho2 = sum(r.rho2 for r in resultados) / R

    # Teórico M/M/1 en cada etapa
    rho1_teo = LAM_A / MU_S1
    rho2_teo = LAM_A / MU_S2
    W1_teo   = 1 / (MU_S1 - LAM_A)
    W2_teo   = 1 / (MU_S2 - LAM_A)

    print("=== MODELO DE DOS SERVIDORES EN SERIE ===")
    print(f"λ={LAM_A}, μ1={MU_S1}, μ2={MU_S2}, T={T}, {R} réplicas\n")
    print(f"{'Métrica':<25} {'Teórico':>12} {'Empírico':>12}")
    print("-" * 52)
    print(f"{'ρ1':<25} {rho1_teo:>12.4f} {rho1:>12.4f}")
    print(f"{'ρ2':<25} {rho2_teo:>12.4f} {rho2:>12.4f}")
    print(f"{'W1':<25} {W1_teo:>12.4f} {W1:>12.4f}")
    print(f"{'W2':<25} {W2_teo:>12.4f} {W2:>12.4f}")
    print(f"{'W total':<25} {W1_teo+W2_teo:>12.4f} {W:>12.4f}")
