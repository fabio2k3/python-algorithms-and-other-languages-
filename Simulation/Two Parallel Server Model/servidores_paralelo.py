"""
Modelo de dos Servidores en Paralelo
========================================
Dos servidores trabajan simultáneamente con una cola común.

Dinámica:
  - Clientes llegan; si hay servidor libre → atendido inmediatamente.
  - Si ambos ocupados → entra en cola común FIFO.
  - Al terminar un servidor → atiende al primer cliente de la cola.
  - Opera hasta T; tras T no entran más clientes pero se atiende
    a todos los que ya están en el sistema.

Estado (según el resumen de conferencias):
  SS = (n, i1, i2, i3, ..., in)
  n = total en sistema, i1 = cliente en S1, i2 = cliente en S2,
  i3..in = cola.

Variables clave:
  tA  : próximo arribo
  tD1 : próxima salida S1  (inf si vacío)
  tD2 : próxima salida S2  (inf si vacío)
  n   : total en sistema
  i1  : id cliente en S1  (0 = vacío)
  i2  : id cliente en S2  (0 = vacío)

Invariantes garantizados:
  len(tiempos_espera_cola) == len(tiempos_en_sistema) == ND
  Para todo i: tiempos_espera_cola[i] >= 0
  Para todo i: tiempos_en_sistema[i]  >= tiempos_espera_cola[i]
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List


GenTime = Callable[[random.Random], float]


@dataclass
class ResultadoParalelo:
    NA: int
    ND: int
    tiempo_sim: float

    tiempos_espera_cola: List[float]   # len == ND
    tiempos_en_sistema:  List[float]   # len == ND

    area_n: float
    t_ocupado_S1: float
    t_ocupado_S2: float

    @property
    def W_q(self) -> float:
        return sum(self.tiempos_espera_cola) / max(self.ND, 1)

    @property
    def W(self) -> float:
        return sum(self.tiempos_en_sistema) / max(self.ND, 1)

    @property
    def L(self) -> float:
        return self.area_n / max(self.tiempo_sim, 1e-12)

    @property
    def rho1(self) -> float:
        return self.t_ocupado_S1 / max(self.tiempo_sim, 1e-12)

    @property
    def rho2(self) -> float:
        return self.t_ocupado_S2 / max(self.tiempo_sim, 1e-12)

    @property
    def rho(self) -> float:
        return (self.rho1 + self.rho2) / 2

    @property
    def throughput(self) -> float:
        return self.ND / max(self.tiempo_sim, 1e-12)


def simular_paralelo(
    gen_interarribo: GenTime,
    gen_servicio: GenTime,
    T: float,
    seed: int = None,
) -> ResultadoParalelo:
    """
    Simulación plana (sin funciones anidadas) para evitar problemas
    de captura de variables por referencia en closures Python.
    """
    rng = random.Random(seed)

    t   = 0.0
    NA  = 0
    ND  = 0
    n   = 0

    # id del cliente en cada servidor (0 = servidor vacío)
    i1: int = 0
    i2: int = 0

    tA  = gen_interarribo(rng)
    tD1 = math.inf
    tD2 = math.inf

    cola: List[int] = []

    # tiempos indexados por id de cliente
    t_arr: Dict[int, float] = {}   # t de llegada al sistema
    t_ini: Dict[int, float] = {}   # t de inicio de servicio

    # Para medir tiempo de ocupación de cada servidor
    t_ini_s1 = 0.0
    t_ini_s2 = 0.0

    wq_list: List[float] = []
    w_list:  List[float] = []

    area_n   = 0.0
    t_last   = 0.0
    ocu_S1   = 0.0
    ocu_S2   = 0.0

    while True:

        # ── EVENTO DE ARRIBO ────────────────────────────────────────────────
        if tA <= tD1 and tA <= tD2 and tA <= T:
            area_n += n * (tA - t_last)
            t_last  = tA
            t       = tA

            NA += 1
            n  += 1
            t_arr[NA] = t
            tA = t + gen_interarribo(rng)

            if i1 == 0:                          # S1 libre
                i1       = NA
                t_ini[NA] = t
                t_ini_s1  = t
                tD1 = t + gen_servicio(rng)

            elif i2 == 0:                        # S2 libre
                i2       = NA
                t_ini[NA] = t
                t_ini_s2  = t
                tD2 = t + gen_servicio(rng)

            else:                                # ambos ocupados
                cola.append(NA)

        # ── EVENTO SALIDA S1 ────────────────────────────────────────────────
        elif tD1 <= tD2 and tD1 != math.inf and (tD1 <= T or (tA > T and n > 0)):
            area_n += n * (tD1 - t_last)
            t_last  = tD1
            t       = tD1

            cliente = i1
            ocu_S1 += t - t_ini_s1

            # Registrar métricas del cliente que sale
            arr = t_arr.pop(cliente)
            ini = t_ini.pop(cliente)
            wq_list.append(max(ini - arr, 0.0))
            w_list.append(t - arr)

            ND += 1
            n  -= 1
            i1  = 0
            tD1 = math.inf

            # Atender siguiente en cola con S1
            if cola:
                sig      = cola.pop(0)
                i1       = sig
                t_ini[sig] = t
                t_ini_s1   = t
                tD1 = t + gen_servicio(rng)

            if n == 0 and tA > T:
                break

        # ── EVENTO SALIDA S2 ────────────────────────────────────────────────
        elif tD2 < tD1 and tD2 != math.inf and (tD2 <= T or (tA > T and n > 0)):
            area_n += n * (tD2 - t_last)
            t_last  = tD2
            t       = tD2

            cliente = i2
            ocu_S2 += t - t_ini_s2

            arr = t_arr.pop(cliente)
            ini = t_ini.pop(cliente)
            wq_list.append(max(ini - arr, 0.0))
            w_list.append(t - arr)

            ND += 1
            n  -= 1
            i2  = 0
            tD2 = math.inf

            if cola:
                sig      = cola.pop(0)
                i2       = sig
                t_ini[sig] = t
                t_ini_s2   = t
                tD2 = t + gen_servicio(rng)

            if n == 0 and tA > T:
                break

        else:
            break

    return ResultadoParalelo(
        NA=NA,
        ND=ND,
        tiempo_sim=t,
        tiempos_espera_cola=wq_list,
        tiempos_en_sistema=w_list,
        area_n=area_n,
        t_ocupado_S1=ocu_S1,
        t_ocupado_S2=ocu_S2,
    )


def gen_exp(lam: float) -> GenTime:
    def _g(rng: random.Random) -> float:
        u = rng.random()
        while u == 0:
            u = rng.random()
        return -math.log(u) / lam
    return _g


def replicar_paralelo(
    gen_interarribo: GenTime,
    gen_servicio: GenTime,
    T: float,
    r: int,
    seed: int = None,
) -> List[ResultadoParalelo]:
    base = random.Random(seed)
    return [
        simular_paralelo(gen_interarribo, gen_servicio, T,
                         seed=base.randint(0, 2**31))
        for _ in range(r)
    ]


# ── Ejemplo de uso ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    LAM_A = 3.0
    MU_S  = 2.0
    T     = 200.0
    R     = 1_000

    rho_teo = LAM_A / (2 * MU_S)
    resultados = replicar_paralelo(gen_exp(LAM_A), gen_exp(MU_S), T, R, seed=42)

    def media(lst): return sum(lst) / len(lst)
    W_emp   = media([r.W   for r in resultados])
    Wq_emp  = media([r.W_q for r in resultados])
    rho_emp = media([r.rho for r in resultados])

    print("=== MODELO DE DOS SERVIDORES EN PARALELO (M/M/2) ===")
    print(f"λ={LAM_A}, μ={MU_S}/servidor, ρ_teo={rho_teo:.3f}, T={T}, {R} réplicas\n")
    print(f"  ρ (utilización media) = {rho_emp:.4f}  (teórico ≈ {rho_teo:.4f})")
    print(f"  W (tiempo en sistema) = {W_emp:.4f}")
    print(f"  W_q (espera en cola)  = {Wq_emp:.4f}")
