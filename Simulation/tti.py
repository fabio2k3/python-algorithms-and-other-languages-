"""
Método de la Transformada Inversa (TTI)
========================================
Fundamento teórico:
  Sea U ~ Uniforme(0,1) y F una FDA continua e invertible. Entonces:
      X = F⁻¹(U)  tiene distribución F.

Lógica:
  1. Generar U ~ Uniforme(0,1)
  2. Calcular X = F⁻¹(U)
  3. Retornar X

Distribuciones implementadas:
  - Exponencial: F(x) = 1 - e^(-λx)  →  X = -(1/λ) * ln(U)
  - Uniforme(a,b): F(x) = (x-a)/(b-a)  →  X = a + (b-a)*U
  - Weibull(λ,k): F(x) = 1 - e^(-(λx)^k)  →  X = (1/λ)*(-ln(U))^(1/k)
  - Cauchy(x0,γ): F(x) = 0.5 + arctan((x-x0)/γ)/π  →  X = x0 + γ*tan(π*(U-0.5))
"""

import math
import random
from typing import List


# ── Generador base ─────────────────────────────────────────────────────────────

def uniforme(rng: random.Random = None) -> float:
    """Genera U ~ Uniforme(0,1)."""
    if rng is None:
        rng = random.Random()
    u = 0.0
    while u == 0.0:           # evitar ln(0)
        u = rng.random()
    return u


# ── TTI: Distribución Exponencial ─────────────────────────────────────────────

def tti_exponencial(lam: float, rng: random.Random = None) -> float:
    """
    Genera X ~ Exponencial(λ) usando TTI.

    FDA:     F(x) = 1 - e^(-λx),   x ≥ 0
    Inversa: X = -(1/λ) * ln(U)

    Args:
        lam: tasa λ > 0
        rng: generador de números aleatorios (opcional)

    Returns:
        Muestra X ~ Exp(λ)
    """
    assert lam > 0, "λ debe ser positivo"
    u = uniforme(rng)
    return -math.log(u) / lam


def muestras_exponencial(lam: float, n: int, seed: int = None) -> List[float]:
    """Genera n muestras ~ Exponencial(λ)."""
    rng = random.Random(seed)
    return [tti_exponencial(lam, rng) for _ in range(n)]


# ── TTI: Distribución Uniforme(a, b) ──────────────────────────────────────────

def tti_uniforme_ab(a: float, b: float, rng: random.Random = None) -> float:
    """
    Genera X ~ Uniforme(a, b) usando TTI.

    FDA:     F(x) = (x - a) / (b - a)
    Inversa: X = a + (b - a) * U

    Args:
        a: límite inferior
        b: límite superior (b > a)
        rng: generador de números aleatorios (opcional)
    """
    assert b > a, "Debe cumplirse b > a"
    if rng is None:
        rng = random.Random()
    u = rng.random()
    return a + (b - a) * u


def muestras_uniforme_ab(a: float, b: float, n: int, seed: int = None) -> List[float]:
    """Genera n muestras ~ Uniforme(a, b)."""
    rng = random.Random(seed)
    return [tti_uniforme_ab(a, b, rng) for _ in range(n)]


# ── TTI: Distribución Weibull ─────────────────────────────────────────────────

def tti_weibull(lam: float, k: float, rng: random.Random = None) -> float:
    """
    Genera X ~ Weibull(λ, k) usando TTI.

    FDA:     F(x) = 1 - e^(-(λx)^k)
    Inversa: X = (1/λ) * (-ln(U))^(1/k)

    Args:
        lam: parámetro de escala λ > 0
        k:   parámetro de forma  k > 0
        rng: generador (opcional)
    """
    assert lam > 0 and k > 0, "λ y k deben ser positivos"
    u = uniforme(rng)
    return (1.0 / lam) * ((-math.log(u)) ** (1.0 / k))


def muestras_weibull(lam: float, k: float, n: int, seed: int = None) -> List[float]:
    """Genera n muestras ~ Weibull(λ, k)."""
    rng = random.Random(seed)
    return [tti_weibull(lam, k, rng) for _ in range(n)]


# ── TTI: Distribución Cauchy ──────────────────────────────────────────────────

def tti_cauchy(x0: float, gamma: float, rng: random.Random = None) -> float:
    """
    Genera X ~ Cauchy(x0, γ) usando TTI.

    FDA:     F(x) = 0.5 + arctan((x - x0)/γ) / π
    Inversa: X = x0 + γ * tan(π * (U - 0.5))

    Args:
        x0:    parámetro de localización
        gamma: parámetro de escala γ > 0
        rng:   generador (opcional)
    """
    assert gamma > 0, "γ debe ser positivo"
    if rng is None:
        rng = random.Random()
    u = rng.random()
    while u == 0.5:            # tan(0) es válido pero evitamos bordes
        u = rng.random()
    return x0 + gamma * math.tan(math.pi * (u - 0.5))


def muestras_cauchy(x0: float, gamma: float, n: int, seed: int = None) -> List[float]:
    """Genera n muestras ~ Cauchy(x0, γ)."""
    rng = random.Random(seed)
    return [tti_cauchy(x0, gamma, rng) for _ in range(n)]


# ── Estadísticas descriptivas básicas ─────────────────────────────────────────

def media(muestras: List[float]) -> float:
    return sum(muestras) / len(muestras)

def varianza(muestras: List[float]) -> float:
    m = media(muestras)
    return sum((x - m) ** 2 for x in muestras) / len(muestras)


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 100_000
    LAM = 2.0

    print("=== MÉTODO DE LA TRANSFORMADA INVERSA (TTI) ===\n")

    # Exponencial
    exp_muestras = muestras_exponencial(LAM, N, seed=42)
    print(f"Exponencial(λ={LAM})")
    print(f"  Media teórica   : {1/LAM:.4f}")
    print(f"  Media muestral  : {media(exp_muestras):.4f}")
    print(f"  Varianza teórica: {1/LAM**2:.4f}")
    print(f"  Varianza muestral:{varianza(exp_muestras):.4f}")

    # Uniforme(a,b)
    a, b = 2.0, 5.0
    uni_muestras = muestras_uniforme_ab(a, b, N, seed=42)
    print(f"\nUniforme(a={a}, b={b})")
    print(f"  Media teórica   : {(a+b)/2:.4f}")
    print(f"  Media muestral  : {media(uni_muestras):.4f}")

    # Weibull
    k = 2.0
    wei_muestras = muestras_weibull(LAM, k, N, seed=42)
    import math
    media_teo = (1/LAM) * math.gamma(1 + 1/k)
    print(f"\nWeibull(λ={LAM}, k={k})")
    print(f"  Media teórica   : {media_teo:.4f}")
    print(f"  Media muestral  : {media(wei_muestras):.4f}")

    print("\nGeneración completada correctamente.")
