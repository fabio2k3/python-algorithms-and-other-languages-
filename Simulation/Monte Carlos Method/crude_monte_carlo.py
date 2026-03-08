"""
Crude Monte Carlo (Monte Carlo Básico)
========================================
Método para estimar integrales definidas mediante muestreo aleatorio.

Fundamento teórico:
  Para estimar  I = ∫[a,b] f(x) dx:

  Si X ~ Uniforme(a, b), entonces:
      E[f(X)] = (1/(b-a)) · ∫[a,b] f(x) dx  =  I / (b-a)

  Por tanto:
      I = (b-a) · E[f(X)]

  Estimador de Monte Carlo con n muestras:
      Î = (b-a) · (1/n) · Σ f(Xᵢ)   donde Xᵢ ~ U(a,b)

  Por la Ley de los Grandes Números, Î → I cuando n → ∞.

Error estadístico:
  - Varianza del estimador:   Var(Î) = (b-a)² · Var(f(X)) / n
  - Desviación estándar:      σ(Î)   = (b-a) · σ(f(X)) / √n
  - Intervalo de confianza:   Î ± t_{α/2} · S / √n

  La convergencia es O(1/√n), independiente de la dimensión → ventaja
  clave en integrales de alta dimensión.

Pseudocódigo (del resumen de conferencias):
  S = 0
  Para i = 1 hasta n:
      Generar Xᵢ ~ Uniforme(a, b)
      S = S + f(Xᵢ)
  Î = (b-a) · (S / n)
"""

import math
import random
from typing import Callable, List, Tuple


# ── Tipos ─────────────────────────────────────────────────────────────────────

Func1D   = Callable[[float], float]
FuncND   = Callable[[List[float]], float]


# ══════════════════════════════════════════════════════════════════════════════
#  CRUDE MONTE CARLO — 1 DIMENSIÓN
# ══════════════════════════════════════════════════════════════════════════════

def crude_mc_1d(
    f: Func1D,
    a: float,
    b: float,
    n: int,
    seed: int = None,
) -> Tuple[float, float, float]:
    """
    Estima  I = ∫[a,b] f(x) dx  mediante Crude Monte Carlo.

    Algoritmo:
        S = 0
        Para i = 1..n:
            Xᵢ ~ U(a,b)
            S += f(Xᵢ)
        Î = (b-a) · S / n

    Args:
        f:    función a integrar
        a:    límite inferior
        b:    límite superior  (b > a)
        n:    número de muestras
        seed: semilla para reproducibilidad

    Returns:
        (estimacion, error_estandar, semivarianza_muestral)
        - estimacion:       Î ≈ ∫[a,b] f(x) dx
        - error_estandar:   σ̂(Î) = (b-a) · S_f / √n
        - varianza_muestral: S²_f  de los f(Xᵢ)
    """
    assert b > a,   "Debe cumplirse b > a"
    assert n >= 2,  "Se necesitan al menos 2 muestras"

    rng   = random.Random(seed)
    ancho = b - a

    # ── Paso 1: acumular suma y suma de cuadrados ────────────────────────────
    S  = 0.0   # Σ f(Xᵢ)
    S2 = 0.0   # Σ f(Xᵢ)²

    for _ in range(n):
        x  = a + ancho * rng.random()   # Xᵢ ~ U(a,b)
        fx = f(x)
        S  += fx
        S2 += fx * fx

    # ── Paso 2: estimación ──────────────────────────────────────────────────
    media_f   = S / n
    estimacion = ancho * media_f

    # ── Paso 3: error estadístico ────────────────────────────────────────────
    # Varianza muestral de f(X):  S² = [Σf²/n - (Σf/n)²] · n/(n-1)
    var_f         = (S2 / n - media_f ** 2) * n / (n - 1)
    error_estandar = ancho * math.sqrt(max(var_f, 0.0) / n)

    return estimacion, error_estandar, var_f


def intervalo_confianza_1d(
    f: Func1D,
    a: float,
    b: float,
    n: int,
    alpha: float = 0.05,
    seed: int = None,
) -> Tuple[float, float, float]:
    """
    Î ± z_{α/2} · σ̂(Î)   (aproximación normal para n grande)

    Args:
        f, a, b, n, seed: ídem crude_mc_1d
        alpha: nivel de significancia (0.05 → IC 95%)

    Returns:
        (estimacion, limite_inferior, limite_superior)
    """
    # Cuantil z_{α/2}  (aproximación numérica para casos comunes)
    z_table = {0.10: 1.645, 0.05: 1.960, 0.025: 2.241, 0.01: 2.576}
    z = z_table.get(alpha, 1.960)

    est, err, _ = crude_mc_1d(f, a, b, n, seed=seed)
    return est, est - z * err, est + z * err


# ══════════════════════════════════════════════════════════════════════════════
#  CRUDE MONTE CARLO — MULTIDIMENSIONAL
# ══════════════════════════════════════════════════════════════════════════════

def crude_mc_nd(
    f: FuncND,
    limites: List[Tuple[float, float]],
    n: int,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Estima  I = ∫...∫ f(x₁,...,xₖ) dx₁...dxₖ  sobre el hipercubo
    definido por 'limites' = [(a₁,b₁), (a₂,b₂), ..., (aₖ,bₖ)].

    Î = Vol · (1/n) · Σ f(Xᵢ)   donde Xᵢ ~ U(hipercubo)

    Args:
        f:       función k-dimensional
        limites: lista de (aᵢ, bᵢ) por dimensión
        n:       número de muestras
        seed:    semilla

    Returns:
        (estimacion, error_estandar)
    """
    assert all(b > a for a, b in limites), "bᵢ > aᵢ para todo i"
    assert n >= 2

    rng    = random.Random(seed)
    anchos = [b - a for a, b in limites]
    vol    = 1.0
    for w in anchos:
        vol *= w

    S  = 0.0
    S2 = 0.0

    for _ in range(n):
        x  = [limites[i][0] + anchos[i] * rng.random() for i in range(len(limites))]
        fx = f(x)
        S  += fx
        S2 += fx * fx

    media_f    = S / n
    estimacion = vol * media_f
    var_f      = (S2 / n - media_f ** 2) * n / (n - 1)
    error      = vol * math.sqrt(max(var_f, 0.0) / n)

    return estimacion, error


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERGENCIA: múltiples replicaciones independientes
# ══════════════════════════════════════════════════════════════════════════════

def replicaciones_mc(
    f: Func1D,
    a: float,
    b: float,
    n: int,
    r: int,
    seed: int = None,
) -> List[float]:
    """
    Ejecuta r replicaciones independientes y devuelve las r estimaciones.
    Útil para estudiar la distribución del estimador.

    Returns:
        Lista de r valores Î_j
    """
    base_rng = random.Random(seed)
    return [
        crude_mc_1d(f, a, b, n, seed=base_rng.randint(0, 2**31))[0]
        for _ in range(r)
    ]


def ic_replicaciones(
    estimaciones: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Intervalo de confianza empírico a partir de r replicaciones.
    X̄ ± t_{α/2, r-1} · S / √r  (fórmula del resumen de conferencias)

    Returns:
        (media, limite_inferior, limite_superior)
    """
    r   = len(estimaciones)
    xbar = sum(estimaciones) / r
    s2   = sum((x - xbar) ** 2 for x in estimaciones) / (r - 1)
    s    = math.sqrt(s2)

    # t_{α/2, r-1}: aproximación para r ≥ 5 (usa normal si r≥30)
    t_table = {
        (0.05, 4): 2.776, (0.05, 9): 2.262, (0.05, 19): 2.093,
        (0.05, 29): 2.045, (0.05, 49): 2.010, (0.05, 99): 1.984,
    }
    # Buscar el valor de tabla más cercano
    if r >= 100:
        t = 1.960
    elif r >= 50:
        t = t_table.get((alpha, 49), 2.010)
    elif r >= 30:
        t = t_table.get((alpha, 29), 2.045)
    elif r >= 20:
        t = t_table.get((alpha, 19), 2.093)
    elif r >= 10:
        t = t_table.get((alpha, 9), 2.262)
    else:
        t = t_table.get((alpha, max(r-1, 4)), 2.776)

    margen = t * s / math.sqrt(r)
    return xbar, xbar - margen, xbar + margen


# ══════════════════════════════════════════════════════════════════════════════
#  ESTIMACIÓN DE π (ejemplo clásico de Monte Carlo geométrico)
# ══════════════════════════════════════════════════════════════════════════════

def estimar_pi(n: int, seed: int = None) -> Tuple[float, float]:
    """
    Estima π usando el método del círculo inscrito en el cuadrado [-1,1]².

    Punto (x,y) cae dentro del círculo unitario si x²+y² ≤ 1.
    P(dentro) = π·r²/(2r)² = π/4   →   π ≈ 4 · (conteo_dentro / n)

    Args:
        n:    número de puntos aleatorios
        seed: semilla

    Returns:
        (estimacion_pi, error_estandar)
    """
    rng    = random.Random(seed)
    dentro = 0

    for _ in range(n):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            dentro += 1

    p_hat  = dentro / n
    pi_est = 4 * p_hat
    # Error: Bernoulli con p=π/4
    error  = 4 * math.sqrt(p_hat * (1 - p_hat) / n)
    return pi_est, error


# ── Estadísticas auxiliares ───────────────────────────────────────────────────

def media(vals: List[float]) -> float:
    return sum(vals) / len(vals)

def varianza(vals: List[float]) -> float:
    m = media(vals)
    return sum((x - m) ** 2 for x in vals) / len(vals)


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== CRUDE MONTE CARLO ===\n")

    # ── ∫[0,1] x² dx = 1/3 ────────────────────────────────────────────────
    f1     = lambda x: x ** 2
    est1, err1, _ = crude_mc_1d(f1, 0, 1, n=100_000, seed=42)
    print(f"∫[0,1] x² dx")
    print(f"  Valor exacto : {1/3:.6f}")
    print(f"  Estimación   : {est1:.6f}  ±  {err1:.6f}")

    # ── ∫[0,π] sin(x) dx = 2 ──────────────────────────────────────────────
    f2     = math.sin
    est2, err2, _ = crude_mc_1d(f2, 0, math.pi, n=100_000, seed=42)
    print(f"\n∫[0,π] sin(x) dx")
    print(f"  Valor exacto : 2.000000")
    print(f"  Estimación   : {est2:.6f}  ±  {err2:.6f}")

    # ── ∫[0,1]∫[0,1] (x²+y²) dxdy = 2/3 ──────────────────────────────────
    f3     = lambda xy: xy[0]**2 + xy[1]**2
    est3, err3 = crude_mc_nd(f3, [(0,1),(0,1)], n=100_000, seed=42)
    print(f"\n∫∫[0,1]² (x²+y²) dxdy")
    print(f"  Valor exacto : {2/3:.6f}")
    print(f"  Estimación   : {est3:.6f}  ±  {err3:.6f}")

    # ── Estimación de π ───────────────────────────────────────────────────
    pi_est, pi_err = estimar_pi(1_000_000, seed=42)
    print(f"\nEstimación de π (método geométrico)")
    print(f"  π exacto     : {math.pi:.6f}")
    print(f"  Estimación   : {pi_est:.6f}  ±  {pi_err:.6f}")

    # ── Intervalo de confianza con replicaciones ──────────────────────────
    reps  = replicaciones_mc(f1, 0, 1, n=10_000, r=50, seed=42)
    xbar, lo, hi = ic_replicaciones(reps)
    print(f"\nIC 95% para ∫[0,1] x² dx  (50 replicaciones, n=10000 c/u)")
    print(f"  Î = {xbar:.6f}  →  [{lo:.6f},  {hi:.6f}]")
    print(f"  Valor exacto contenido en IC: {lo <= 1/3 <= hi}")
