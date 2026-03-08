"""
Modelo de Inventario con Política (s, S)
==========================================
Gestión de existencias en un almacén con política de umbral (s, S).

Política:
  - Si el nivel de inventario x cae por debajo del mínimo s:
    se realiza un pedido de reposición para alcanzar el nivel máximo S.
    Cantidad pedida: y = S - x
  - El pedido llega tras un tiempo de entrega L (lead time).
  - Solo se puede tener un pedido pendiente a la vez.

Variables del sistema (según el resumen de conferencias):
  t    : tiempo de simulación
  x    : nivel actual de inventario
  y    : cantidad pedida pendiente de arribo (0 si no hay pedido)
  tA   : tiempo del próximo evento de demanda
  tM   : tiempo de llegada del próximo pedido (∞ si no hay pedido)

Variables de costo (según el resumen):
  C    : costo acumulado por órdenes de reposición
  H    : costo acumulado por almacenaje
  R    : ingresos acumulados por ventas
  P    : pérdidas por demandas no satisfechas

Costos:
  c(y) : costo de pedir y unidades  (fijo + variable: c_f + c_v·y)
  h    : costo de almacenaje por unidad por unidad de tiempo
  r    : ingreso por unidad vendida
  p    : pérdida por unidad de demanda no satisfecha

Métricas:
  ganancia_neta = R - C - H - P
  nivel_medio_inventario = ∫x(t)dt / T
  fraccion_demanda_satisfecha = unidades_vendidas / demanda_total
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple


GenTime  = Callable[[random.Random], float]
GenDemand = Callable[[random.Random], float]


@dataclass
class ResultadoInventario:
    tiempo_sim: float
    n_pedidos: int          # número de órdenes realizadas
    n_demandas: int         # número de eventos de demanda
    unidades_vendidas: float
    demanda_total: float
    unidades_faltantes: float

    C: float    # costo total de pedidos
    H: float    # costo total de almacenaje
    R: float    # ingresos por ventas
    P: float    # pérdidas por faltantes

    historial_inventario: List[Tuple[float, float]]  # (t, nivel)

    @property
    def ganancia_neta(self) -> float:
        return self.R - self.C - self.H - self.P

    @property
    def nivel_medio_inventario(self) -> float:
        """Nivel medio de inventario (área bajo la curva / T)."""
        if not self.historial_inventario or self.tiempo_sim == 0:
            return 0.0
        area = 0.0
        for i in range(1, len(self.historial_inventario)):
            t0, x0 = self.historial_inventario[i-1]
            t1, _  = self.historial_inventario[i]
            area  += x0 * (t1 - t0)
        return area / self.tiempo_sim

    @property
    def fraccion_satisfecha(self) -> float:
        if self.demanda_total == 0:
            return 1.0
        return self.unidades_vendidas / self.demanda_total


def _costo_pedido(y: float, c_fijo: float, c_var: float) -> float:
    """c(y) = c_fijo + c_var·y  si y > 0, else 0."""
    return c_fijo + c_var * y if y > 0 else 0.0


def simular_inventario(
    s: float,
    S: float,
    T: float,
    x0: float,
    gen_interdemanda: GenTime,
    gen_demanda: GenDemand,
    L: float,
    c_fijo: float,
    c_var: float,
    h: float,
    r_unit: float,
    p_unit: float,
    seed: int = None,
) -> ResultadoInventario:
    """
    Simula el modelo de inventario con política (s, S).

    Args:
        s              : nivel mínimo de reorden
        S              : nivel máximo de inventario
        T              : horizonte de simulación
        x0             : nivel inicial de inventario
        gen_interdemanda: generador de tiempos entre demandas
        gen_demanda    : generador de tamaño de demanda
        L              : tiempo de entrega (lead time, constante)
        c_fijo         : costo fijo por orden
        c_var          : costo variable por unidad pedida
        h              : costo de almacenaje por unidad por unidad tiempo
        r_unit         : ingreso por unidad vendida
        p_unit         : penalización por unidad no satisfecha
        seed           : semilla aleatoria
    """
    assert s < S,   "s debe ser menor que S"
    assert x0 >= 0, "inventario inicial no negativo"

    rng = random.Random(seed)

    # ── Inicialización ────────────────────────────────────────────────────
    t = 0.0
    x = x0
    y = 0.0                         # pedido pendiente
    tA = gen_interdemanda(rng)      # próxima demanda
    tM = math.inf                   # próxima llegada de pedido

    C = H = R = P = 0.0
    n_pedidos   = 0
    n_demandas  = 0
    unidades_vendidas  = 0.0
    demanda_total      = 0.0
    unidades_faltantes = 0.0

    historial: List[Tuple[float, float]] = [(0.0, x)]

    while True:
        prox = min(tA, tM)

        # ── EVENTO DE DEMANDA (tA ≤ tM, tA ≤ T) ─────────────────────────
        if tA <= tM and tA <= T:
            # Actualizar costo almacenaje hasta tA
            H += (tA - t) * x * h
            t  = tA

            n_demandas += 1
            X = gen_demanda(rng)        # demanda aleatoria
            demanda_total += X

            w = min(x, X)               # unidades vendidas
            R += w * r_unit
            P += (X - w) * p_unit
            unidades_vendidas  += w
            unidades_faltantes += X - w
            x -= w

            historial.append((t, x))

            # Verificar umbral de reorden
            if x < s and y == 0:
                y  = S - x
                tM = t + L
                C += _costo_pedido(y, c_fijo, c_var)
                n_pedidos += 1

            tA = t + gen_interdemanda(rng)

        # ── EVENTO DE REPOSICIÓN (tM < tA, tM ≤ T) ──────────────────────
        elif tM < tA and tM <= T:
            # Actualizar costo almacenaje hasta tM
            H += (tM - t) * x * h
            t  = tM

            x += y
            historial.append((t, x))
            y  = 0.0
            tM = math.inf

        # ── FIN: ambos eventos superan T ─────────────────────────────────
        else:
            # Procesar el último tramo hasta T
            H += (T - t) * x * h
            t  = T
            historial.append((t, x))
            break

    return ResultadoInventario(
        tiempo_sim=t,
        n_pedidos=n_pedidos,
        n_demandas=n_demandas,
        unidades_vendidas=unidades_vendidas,
        demanda_total=demanda_total,
        unidades_faltantes=unidades_faltantes,
        C=C,
        H=H,
        R=R,
        P=P,
        historial_inventario=historial,
    )


def gen_exp(lam: float) -> GenTime:
    def _g(rng: random.Random) -> float:
        u = rng.random()
        while u == 0:
            u = rng.random()
        return -math.log(u) / lam
    return _g


def gen_uniforme_int(lo: int, hi: int) -> GenDemand:
    """Demanda entera uniforme en [lo, hi]."""
    def _g(rng: random.Random) -> float:
        return float(rng.randint(lo, hi))
    return _g


def gen_constante(valor: float) -> GenDemand:
    """Demanda constante."""
    def _g(rng: random.Random) -> float:
        return valor
    return _g


def replicar_inventario(
    s: float, S: float, T: float, x0: float,
    gen_interdemanda: GenTime, gen_demanda: GenDemand,
    L: float, c_fijo: float, c_var: float,
    h: float, r_unit: float, p_unit: float,
    r: int, seed: int = None,
) -> List[ResultadoInventario]:
    base = random.Random(seed)
    return [
        simular_inventario(
            s, S, T, x0, gen_interdemanda, gen_demanda,
            L, c_fijo, c_var, h, r_unit, p_unit,
            seed=base.randint(0, 2**31)
        )
        for _ in range(r)
    ]


# ── Ejemplo de uso ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s, S   = 20, 100
    T      = 365.0      # un año de simulación
    x0     = 60.0
    L      = 5.0        # lead time 5 días
    c_fijo = 50.0       # costo fijo por pedido
    c_var  = 2.0        # costo variable por unidad
    h      = 0.1        # almacenaje por unidad-día
    r_unit = 10.0       # ingreso por unidad
    p_unit = 5.0        # penalización por unidad faltante
    R      = 500

    resultados = replicar_inventario(
        s, S, T, x0,
        gen_exp(1.0),          # demanda cada ~1 día
        gen_uniforme_int(1, 10),  # demanda entre 1 y 10 unidades
        L, c_fijo, c_var, h, r_unit, p_unit,
        R, seed=42
    )

    media = lambda lst: sum(lst) / len(lst)

    print("=== MODELO DE INVENTARIO (s, S) ===")
    print(f"s={s}, S={S}, T={T}, L={L}, {R} réplicas\n")
    print(f"  Ganancia neta media    : {media([r.ganancia_neta for r in resultados]):.2f}")
    print(f"  Costo pedidos medio    : {media([r.C for r in resultados]):.2f}")
    print(f"  Costo almacenaje medio : {media([r.H for r in resultados]):.2f}")
    print(f"  Ingresos medios        : {media([r.R for r in resultados]):.2f}")
    print(f"  Frec. satisfecha media : {media([r.fraccion_satisfecha for r in resultados]):.4f}")
    print(f"  Nivel medio inv. media : {media([r.nivel_medio_inventario for r in resultados]):.2f}")
