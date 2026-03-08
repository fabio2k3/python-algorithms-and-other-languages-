"""
Simulación de Procesos de Poisson
====================================
Dos variantes según si la tasa λ es constante o varía con el tiempo.

═══════════════════════════════════════════════════════════
1. PROCESO HOMOGÉNEO (tasa constante λ)
───────────────────────────────────────
Los tiempos entre eventos son i.i.d. ~ Exp(λ).

Algoritmo:
  t = 0;  eventos = []
  Mientras t ≤ T:
    U ~ Uniforme(0,1)
    delta = -(1/λ) · ln(U)
    t = t + delta
    Si t ≤ T:  registrar evento en t

═══════════════════════════════════════════════════════════
2. PROCESO NO HOMOGÉNEO (tasa variable λ(t)) — Thinning
────────────────────────────────────────────────────────
Se usa una tasa máxima λ_max ≥ λ(t) para todo t en [0,T].
Cada candidato se acepta con probabilidad λ(t) / λ_max.

Algoritmo (Lewis & Shedler, 1979):
  t = 0
  Mientras t ≤ T:
    U1 ~ Uniforme(0,1)
    delta = -(1/λ_max) · ln(U1)
    t = t + delta
    Si t > T: terminar
    U2 ~ Uniforme(0,1)
    Si U2 ≤ λ(t) / λ_max: aceptar evento en t

Propiedad clave: el proceso resultante es efectivamente
un proceso de Poisson con tasa λ(t).
═══════════════════════════════════════════════════════════
"""

import math
import random
from typing import List, Callable


# ── Generador base ─────────────────────────────────────────────────────────────

def _uniforme(rng: random.Random) -> float:
    """U ~ Uniforme(0,1) sin ceros exactos."""
    u = 0.0
    while u == 0.0:
        u = rng.random()
    return u


# ══════════════════════════════════════════════════════════════════════════════
#  1. PROCESO HOMOGÉNEO
# ══════════════════════════════════════════════════════════════════════════════

def proceso_poisson_homogeneo(
    lam: float,
    T: float,
    seed: int = None,
) -> List[float]:
    """
    Simula un proceso de Poisson homogéneo en el intervalo [0, T].

    Args:
        lam:  tasa constante λ > 0
        T:    tiempo de horizonte de simulación
        seed: semilla para reproducibilidad

    Returns:
        Lista de tiempos de ocurrencia de los eventos en [0, T].
    """
    assert lam > 0, "λ debe ser positivo"
    assert T > 0,   "T debe ser positivo"

    rng = random.Random(seed)
    t = 0.0
    eventos: List[float] = []

    while True:
        u     = _uniforme(rng)
        delta = -math.log(u) / lam      # tiempo hasta el próximo evento ~ Exp(λ)
        t    += delta
        if t > T:
            break
        eventos.append(t)

    return eventos


def replicaciones_homogeneo(
    lam: float,
    T: float,
    r: int,
    seed: int = None,
) -> List[List[float]]:
    """
    Ejecuta r replicaciones del proceso homogéneo.

    Returns:
        Lista de r listas de tiempos de eventos.
    """
    base_rng = random.Random(seed)
    return [
        proceso_poisson_homogeneo(lam, T, seed=base_rng.randint(0, 2**31))
        for _ in range(r)
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  2. PROCESO NO HOMOGÉNEO (Thinning)
# ══════════════════════════════════════════════════════════════════════════════

def proceso_poisson_no_homogeneo(
    lam_t: Callable[[float], float],
    lam_max: float,
    T: float,
    seed: int = None,
) -> List[float]:
    """
    Simula un proceso de Poisson no homogéneo en [0, T] via Thinning.

    Args:
        lam_t:   función λ(t) ≥ 0 para todo t en [0, T]
        lam_max: cota superior: lam_max ≥ lam_t(t) para todo t
        T:       horizonte de simulación
        seed:    semilla para reproducibilidad

    Returns:
        Lista de tiempos de eventos aceptados en [0, T].
    """
    assert lam_max > 0, "λ_max debe ser positivo"
    assert T > 0,       "T debe ser positivo"

    rng = random.Random(seed)
    t = 0.0
    eventos: List[float] = []

    while True:
        # Generar candidato con proceso homogéneo a tasa λ_max
        u1    = _uniforme(rng)
        delta = -math.log(u1) / lam_max
        t    += delta

        if t > T:
            break

        # Aceptar con probabilidad λ(t) / λ_max  (Thinning)
        u2 = rng.random()
        if u2 <= lam_t(t) / lam_max:
            eventos.append(t)

    return eventos


def replicaciones_no_homogeneo(
    lam_t: Callable[[float], float],
    lam_max: float,
    T: float,
    r: int,
    seed: int = None,
) -> List[List[float]]:
    """
    Ejecuta r replicaciones del proceso no homogéneo.
    """
    base_rng = random.Random(seed)
    return [
        proceso_poisson_no_homogeneo(lam_t, lam_max, T, seed=base_rng.randint(0, 2**31))
        for _ in range(r)
    ]


# ── Métricas del proceso ──────────────────────────────────────────────────────

def conteo_esperado_homogeneo(lam: float, T: float) -> float:
    """E[N(T)] = λ·T para proceso homogéneo."""
    return lam * T

def conteo_esperado_no_homogeneo(lam_t: Callable[[float], float], T: float,
                                  pasos: int = 10_000) -> float:
    """
    E[N(T)] = ∫₀ᵀ λ(t) dt  (aproximación numérica por regla del trapecio).
    """
    dt = T / pasos
    integral = sum(lam_t(i * dt) * dt for i in range(pasos))
    return integral

def media_conteos(replicas: List[List[float]]) -> float:
    """Media del número de eventos sobre todas las replicaciones."""
    return sum(len(r) for r in replicas) / len(replicas)

def varianza_conteos(replicas: List[List[float]]) -> float:
    """Varianza del número de eventos sobre todas las replicaciones."""
    conteos = [len(r) for r in replicas]
    m = sum(conteos) / len(conteos)
    return sum((c - m) ** 2 for c in conteos) / len(conteos)


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SIMULACIÓN DE PROCESOS DE POISSON ===\n")

    # ── Proceso homogéneo ──────────────────────────────────────────────────
    LAM = 3.0
    T   = 10.0
    R   = 5_000

    replicas_hom = replicaciones_homogeneo(LAM, T, R, seed=42)

    print(f"PROCESO HOMOGÉNEO  λ={LAM}, T={T}, {R:,} replicaciones")
    print(f"  E[N(T)] teórico  : {conteo_esperado_homogeneo(LAM, T):.2f}")
    print(f"  E[N(T)] muestral : {media_conteos(replicas_hom):.2f}")
    print(f"  Var[N(T)] teórica: {conteo_esperado_homogeneo(LAM, T):.2f}  (=λT)")
    print(f"  Var[N(T)] muestral:{varianza_conteos(replicas_hom):.2f}")

    # Muestra de la primera réplica
    print(f"\n  Primera réplica ({len(replicas_hom[0])} eventos):")
    print(f"  Tiempos: {[round(x, 3) for x in replicas_hom[0][:8]]} ...")

    # ── Proceso no homogéneo ──────────────────────────────────────────────
    # λ(t) = 2 + sin(t)   →   λ_max = 3.0
    def lam_senoidal(t: float) -> float:
        return 2.0 + math.sin(t)

    LAM_MAX = 3.0
    T_NH    = 2 * math.pi   # un período completo

    replicas_nh = replicaciones_no_homogeneo(lam_senoidal, LAM_MAX, T_NH, R, seed=42)

    e_teo = conteo_esperado_no_homogeneo(lam_senoidal, T_NH)
    print(f"\nPROCESO NO HOMOGÉNEO  λ(t)=2+sin(t), λ_max={LAM_MAX}, T={T_NH:.3f}")
    print(f"  E[N(T)] teórico  : {e_teo:.2f}")
    print(f"  E[N(T)] muestral : {media_conteos(replicas_nh):.2f}")
    print(f"  Var[N(T)] muestral:{varianza_conteos(replicas_nh):.2f}")
    print(f"\n  Primera réplica ({len(replicas_nh[0])} eventos):")
    print(f"  Tiempos: {[round(x, 3) for x in replicas_nh[0][:8]]} ...")
