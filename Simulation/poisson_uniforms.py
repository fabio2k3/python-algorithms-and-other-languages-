"""
Generación de la Distribución de Poisson
  — Método del Producto de Uniformes —
=========================================
Basado en la relación entre procesos de Poisson y tiempos exponenciales.

Fundamento matemático:
  Sea N el número de eventos Poisson(λ) en [0,1].
  Los tiempos entre eventos son Exp(λ), por tanto los tiempos acumulados
  T_k = E_1 + E_2 + ... + E_k siguen una distribución Gamma(k, λ).

  Usando la propiedad  T_k = -(1/λ)·ln(U_1·U_2·...·U_k):

      N = mín{ n : U_1·U_2·...·U_n < e^(-λ) } - 1

Algoritmo (Pseudocódigo del resumen):
  L = exp(-λ)
  p = 1
  k = 0
  Repetir:
    k = k + 1
    Generar U ~ Uniforme(0,1)
    p = p * U
  Hasta que p < L
  Retornar k - 1

Distribución de Poisson:
  P(N = k) = e^(-λ) · λ^k / k!
  Media = λ,  Varianza = λ
"""

import math
import random
from typing import List, Dict


# ── Generador base ─────────────────────────────────────────────────────────────

def _uniforme(rng: random.Random) -> float:
    """U ~ Uniforme(0,1) sin ceros exactos."""
    u = 0.0
    while u == 0.0:
        u = rng.random()
    return u


# ── Método del Producto de Uniformes ─────────────────────────────────────────

def poisson_producto_uniformes(lam: float, rng: random.Random = None) -> int:
    """
    Genera X ~ Poisson(λ) usando el método del producto de uniformes.

    N = mín{n : U_1·U_2·...·U_n < e^(-λ)} - 1

    Args:
        lam: parámetro λ > 0
        rng: generador aleatorio (opcional)

    Returns:
        Muestra entera k ≥ 0 de la distribución Poisson(λ)
    """
    assert lam > 0, "λ debe ser positivo"
    if rng is None:
        rng = random.Random()

    L = math.exp(-lam)   # umbral e^(-λ)
    p = 1.0
    k = 0

    while True:
        k += 1
        u = _uniforme(rng)
        p *= u
        if p < L:
            break

    return k - 1


def muestras_poisson(lam: float, n: int, seed: int = None) -> List[int]:
    """
    Genera n muestras ~ Poisson(λ).

    Args:
        lam:  parámetro λ > 0
        n:    número de muestras
        seed: semilla para reproducibilidad

    Returns:
        Lista de n enteros no negativos.
    """
    rng = random.Random(seed)
    return [poisson_producto_uniformes(lam, rng) for _ in range(n)]


# ── Distribución teórica ──────────────────────────────────────────────────────

def pmf_poisson(k: int, lam: float) -> float:
    """P(N = k) = e^(-λ) · λ^k / k!"""
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ── Estadísticas y frecuencias ────────────────────────────────────────────────

def media(muestras: List[float]) -> float:
    return sum(muestras) / len(muestras)

def varianza(muestras: List[float]) -> float:
    m = media(muestras)
    return sum((x - m) ** 2 for x in muestras) / len(muestras)

def frecuencias(muestras: List[int]) -> Dict[int, int]:
    """Devuelve un diccionario {valor: conteo}."""
    freq: Dict[int, int] = {}
    for v in muestras:
        freq[v] = freq.get(v, 0) + 1
    return dict(sorted(freq.items()))


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N   = 100_000
    LAM = 3.5

    print("=== POISSON — MÉTODO DEL PRODUCTO DE UNIFORMES ===\n")
    muestras = muestras_poisson(LAM, N, seed=42)

    print(f"Poisson(λ={LAM})  —  {N:,} muestras")
    print(f"  Media teórica    : {LAM:.4f}")
    print(f"  Media muestral   : {media(muestras):.4f}")
    print(f"  Varianza teórica : {LAM:.4f}")
    print(f"  Varianza muestral: {varianza(muestras):.4f}")

    print(f"\n{'k':>4} | {'P(N=k) teórica':>16} | {'Frec. relativa':>16}")
    print("-" * 42)
    freq = frecuencias(muestras)
    for k in range(min(12, max(freq.keys()) + 1)):
        p_teo  = pmf_poisson(k, LAM)
        p_emp  = freq.get(k, 0) / N
        print(f"  {k:>2} | {p_teo:>16.5f} | {p_emp:>16.5f}")
