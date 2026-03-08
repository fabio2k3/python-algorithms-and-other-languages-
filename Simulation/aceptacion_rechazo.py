"""
Método de Aceptación y Rechazo
================================
Se utiliza cuando la inversa de la FDA no es analíticamente manejable.

Fundamento:
  Si existe g(x) tal que  f(x) ≤ c·g(x)  para todo x,
  se puede generar un candidato desde g(x) y aceptarlo con
  probabilidad  f(X) / (c·g(X)).

Algoritmo:
  Repetir:
    1. Generar X ~ g(x)
    2. Generar U ~ Uniforme(0,1)
    3. Si U ≤ f(X) / (c·g(X)):  aceptar X  →  salir
    4. Si no:  rechazar y repetir

La eficiencia del método es 1/c  (mayor c → más rechazos → menos eficiente).

Distribuciones implementadas:
  - Beta(α, β) usando Uniforme como g(x)
  - Normal(0,1) usando Exponencial como g(x)  [método de Box-Muller alternativo]
  - Distribución personalizada (f arbitraria)
"""

import math
import random
from typing import Callable, List, Tuple


# ── Función auxiliar ──────────────────────────────────────────────────────────

def _safe_uniform(rng: random.Random) -> float:
    """U ~ Uniforme(0,1) sin ceros."""
    u = 0.0
    while u == 0.0:
        u = rng.random()
    return u


# ── Motor genérico de Aceptación y Rechazo ───────────────────────────────────

def aceptacion_rechazo(
    f: Callable[[float], float],
    g_sampler: Callable[[random.Random], float],
    g_pdf: Callable[[float], float],
    c: float,
    rng: random.Random = None,
    max_iter: int = 100_000,
) -> Tuple[float, int]:
    """
    Motor genérico del método de Aceptación y Rechazo.

    Args:
        f:          densidad objetivo f(x)
        g_sampler:  función que genera X ~ g(x)
        g_pdf:      densidad de propuesta g(x)
        c:          constante tal que f(x) ≤ c·g(x) para todo x
        rng:        generador aleatorio (opcional)
        max_iter:   límite de iteraciones de seguridad

    Returns:
        (muestra_aceptada, número_de_intentos)
    """
    if rng is None:
        rng = random.Random()

    for intentos in range(1, max_iter + 1):
        x = g_sampler(rng)
        u = _safe_uniform(rng)
        g_val = g_pdf(x)
        if g_val <= 0:
            continue
        if u <= f(x) / (c * g_val):
            return x, intentos

    raise RuntimeError(f"No se aceptó ninguna muestra tras {max_iter} intentos.")


# ── Distribución Beta(α, β) ───────────────────────────────────────────────────
#   g(x) = Uniforme(0,1),  c = modo de f(x) sobre [0,1]

def _beta_pdf(x: float, alpha: float, beta: float) -> float:
    """Densidad Beta(α,β) sin normalizar (hasta la constante B(α,β))."""
    if x <= 0 or x >= 1:
        return 0.0
    return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))


def _beta_normconst(alpha: float, beta: float) -> float:
    """Valor máximo de x^(α-1)*(1-x)^(β-1) en (0,1) — usado como c."""
    if alpha > 1 and beta > 1:
        # Moda interior
        moda = (alpha - 1) / (alpha + beta - 2)
        c = _beta_pdf(moda, alpha, beta)
    elif alpha == 1 and beta == 1:
        c = 1.0   # Uniforme(0,1): f(x)=1 constante
    elif alpha <= 1 or beta <= 1:
        # Densidad diverge en los bordes; muestrear en (ε, 1-ε) y tomar el max
        xs = [i / 2000 for i in range(1, 2000)]
        c = max(_beta_pdf(x, alpha, beta) for x in xs)
    else:
        xs = [i / 1000 for i in range(1, 1000)]
        c = max(_beta_pdf(x, alpha, beta) for x in xs)
    return max(c, 1e-12)  # nunca devolver 0


def muestra_beta(alpha: float, beta: float, rng: random.Random = None) -> Tuple[float, int]:
    """
    Genera X ~ Beta(α, β) via Aceptación-Rechazo.
    g(x) = Uniforme(0,1)  →  c = max f(x) en (0,1).
    """
    assert alpha > 0 and beta > 0
    c = _beta_normconst(alpha, beta)
    f = lambda x: _beta_pdf(x, alpha, beta)
    g_sampler = lambda rng: rng.random()
    g_pdf     = lambda x: 1.0  # Uniforme(0,1)
    return aceptacion_rechazo(f, g_sampler, g_pdf, c, rng)


def muestras_beta(alpha: float, beta: float, n: int, seed: int = None) -> List[float]:
    """Genera n muestras ~ Beta(α, β)."""
    rng = random.Random(seed)
    return [muestra_beta(alpha, beta, rng)[0] for _ in range(n)]


# ── Distribución Normal(0,1) ──────────────────────────────────────────────────
#   g(x) = Exp(1) sobre x≥0 + simetría
#   c = sqrt(2e/π)

def _normal_pdf(x: float) -> float:
    """Densidad N(0,1)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _exp1_pdf(x: float) -> float:
    """Densidad Exponencial(1) sobre x≥0."""
    return math.exp(-x) if x >= 0 else 0.0


def _exp1_sampler(rng: random.Random) -> float:
    """Genera X ~ Exp(1)."""
    u = _safe_uniform(rng)
    return -math.log(u)


# c = sqrt(2e/π) ≈ 1.3155
_C_NORMAL = math.sqrt(2 * math.e / math.pi)


def muestra_normal(rng: random.Random = None) -> Tuple[float, int]:
    """
    Genera X ~ Normal(0,1) via Aceptación-Rechazo usando Exp(1) como propuesta.
    La mitad de las veces se aplica signo aleatorio (simetría).

    g(x) = Exp(1),  f*(x) = Normal(0,1) restringida a x≥0,
    c = sqrt(2e/π).
    """
    if rng is None:
        rng = random.Random()

    # Trabajamos en x≥0 y luego aplicamos signo aleatorio
    f_half = lambda x: 2 * _normal_pdf(x)   # densidad de |Z|
    x, intentos = aceptacion_rechazo(f_half, _exp1_sampler, _exp1_pdf, _C_NORMAL, rng)
    signo = 1 if rng.random() < 0.5 else -1
    return signo * x, intentos


def muestras_normal(n: int, seed: int = None) -> List[float]:
    """Genera n muestras ~ Normal(0,1)."""
    rng = random.Random(seed)
    return [muestra_normal(rng)[0] for _ in range(n)]


# ── Distribución personalizada (f arbitraria) ─────────────────────────────────

def muestras_custom(
    f: Callable[[float], float],
    g_sampler: Callable[[random.Random], float],
    g_pdf: Callable[[float], float],
    c: float,
    n: int,
    seed: int = None,
) -> List[float]:
    """
    Genera n muestras de una distribución arbitraria f(x)
    usando la propuesta g(x) con constante c.
    """
    rng = random.Random(seed)
    return [aceptacion_rechazo(f, g_sampler, g_pdf, c, rng)[0] for _ in range(n)]


# ── Estadísticas ──────────────────────────────────────────────────────────────

def media(muestras: List[float]) -> float:
    return sum(muestras) / len(muestras)

def varianza(muestras: List[float]) -> float:
    m = media(muestras)
    return sum((x - m) ** 2 for x in muestras) / len(muestras)


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 50_000
    print("=== MÉTODO DE ACEPTACIÓN Y RECHAZO ===\n")

    # Beta(2, 5)
    a, b = 2.0, 5.0
    beta_muestras = muestras_beta(a, b, N, seed=42)
    media_teo = a / (a + b)
    var_teo   = (a * b) / ((a + b) ** 2 * (a + b + 1))
    print(f"Beta(α={a}, β={b})")
    print(f"  Media teórica   : {media_teo:.4f}")
    print(f"  Media muestral  : {media(beta_muestras):.4f}")
    print(f"  Varianza teórica: {var_teo:.5f}")
    print(f"  Varianza muestral:{varianza(beta_muestras):.5f}")

    # Normal(0,1)
    norm_muestras = muestras_normal(N, seed=42)
    print(f"\nNormal(0,1)")
    print(f"  Media teórica   :  0.0000")
    print(f"  Media muestral  : {media(norm_muestras):.4f}")
    print(f"  Varianza teórica:  1.0000")
    print(f"  Varianza muestral: {varianza(norm_muestras):.4f}")

    print("\nGeneración completada correctamente.")
