"""
test_crude_monte_carlo.py
==========================
Suite de tests completa para el método de Crude Monte Carlo.

Bloques de tests:
  1. Integrales 1D con solución analítica conocida
  2. Propiedades estadísticas del estimador (sesgo, varianza, convergencia)
  3. Integrales multidimensionales (2D y 3D)
  4. Estimación de π (método geométrico)
  5. Intervalos de confianza (cobertura y corrección)
  6. Casos borde y robustez
  7. Convergencia O(1/√n)

Estrategia de validación:
  - Tolerancias relativas del 1-3% para N grande.
  - Verificación de propiedades teóricas: E[Î]=I, Var(Î)→0 con n, cobertura IC.
  - Tests deterministas con semilla fija.

Ejecutar:
    python test_crude_monte_carlo.py
"""

import sys
import math
import random
from typing import List

from crude_monte_carlo import (
    crude_mc_1d, crude_mc_nd, intervalo_confianza_1d,
    replicaciones_mc, ic_replicaciones, estimar_pi,
    media, varianza,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades de test
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
results_summary: List[bool] = []


def check_approx(label: str, got: float, expected: float, tol_pct: float = 2.0):
    """Valida |got - expected| / |expected| ≤ tol_pct/100."""
    if abs(expected) < 1e-10:
        ok = abs(got - expected) < tol_pct / 100
    else:
        ok = abs(got - expected) / abs(expected) <= tol_pct / 100
    status = PASS if ok else FAIL
    results_summary.append(ok)
    delta_pct = abs(got - expected) / max(abs(expected), 1e-10) * 100
    print(f"  [{status}] {label:<62} | got={got:.6f}  exp={expected:.6f}  Δ={delta_pct:.2f}%")


def check_bool(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results_summary.append(condition)
    print(f"  [{status}] {label:<62} | {detail}")


def check_in_range(label: str, got: float, lo: float, hi: float):
    ok = lo <= got <= hi
    status = PASS if ok else FAIL
    results_summary.append(ok)
    print(f"  [{status}] {label:<62} | got={got:.6f}  rango=[{lo:.6f}, {hi:.6f}]")


def section(title: str):
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print('═'*80)


# ══════════════════════════════════════════════════════════════════════════════
#  Funciones de prueba con integrales conocidas
# ══════════════════════════════════════════════════════════════════════════════

# ∫[0,1] x dx = 1/2
f_lineal   = lambda x: x
I_lineal   = 0.5

# ∫[0,1] x² dx = 1/3
f_cuadrado = lambda x: x ** 2
I_cuadrado = 1 / 3

# ∫[0,1] x³ dx = 1/4
f_cubico   = lambda x: x ** 3
I_cubico   = 1 / 4

# ∫[0,π] sin(x) dx = 2
f_seno     = math.sin
I_seno     = 2.0

# ∫[0,1] e^x dx = e - 1
f_exp      = math.exp
I_exp      = math.e - 1

# ∫[1,2] ln(x) dx = 2ln(2) - 1
f_log      = math.log
I_log      = 2 * math.log(2) - 1

# ∫[0,1] 1/√(1-x²) dx = π/2  (con cuidado en x→1)
def f_arcsin(x):
    return 1.0 / math.sqrt(max(1 - x * x, 1e-12))
I_arcsin = math.pi / 2

# ∫[0,2] 4 - x² dx = 16/3
def f_parabola(x):
    return 4 - x ** 2
I_parabola = 16 / 3

# ∫[-1,1] |x| dx = 1
f_abs = abs
I_abs = 1.0

# Constante: ∫[0,5] 3 dx = 15
f_const = lambda x: 3.0
I_const = 15.0


# ══════════════════════════════════════════════════════════════════════════════
#  1. INTEGRALES 1D CON SOLUCIÓN ANALÍTICA
# ══════════════════════════════════════════════════════════════════════════════

def test_integrales_1d():
    section("1 · INTEGRALES 1D CON SOLUCIÓN ANALÍTICA")
    N = 500_000

    casos = [
        ("∫[0,1] x dx = 1/2",           f_lineal,   0,         1,          I_lineal,   1.5),
        ("∫[0,1] x² dx = 1/3",          f_cuadrado, 0,         1,          I_cuadrado, 1.5),
        ("∫[0,1] x³ dx = 1/4",          f_cubico,   0,         1,          I_cubico,   1.5),
        ("∫[0,π] sin(x) dx = 2",        f_seno,     0,         math.pi,    I_seno,     1.5),
        ("∫[0,1] e^x dx = e-1",         f_exp,      0,         1,          I_exp,      1.5),
        ("∫[1,2] ln(x) dx = 2ln2-1",    f_log,      1,         2,          I_log,      1.5),
        ("∫[0,2] (4-x²) dx = 16/3",     f_parabola, 0,         2,          I_parabola, 1.5),
        ("∫[-1,1] |x| dx = 1",          f_abs,      -1,        1,          I_abs,      1.5),
        ("∫[0,5] 3 dx = 15 (constante)",f_const,    0,         5,          I_const,    0.1),
    ]

    for label, f, a, b, I_exact, tol in casos:
        est, err, _ = crude_mc_1d(f, a, b, N, seed=42)
        check_approx(label, est, I_exact, tol_pct=tol)

    # Verificar que el error estándar reportado es coherente
    est, err, var_f = crude_mc_1d(f_cuadrado, 0, 1, N, seed=42)
    check_bool(
        "Error estándar > 0",
        err > 0,
        f"err={err:.2e}"
    )
    check_bool(
        "Error estándar decrece con n (N vs N/10)",
        crude_mc_1d(f_cuadrado, 0, 1, N,    seed=1)[1] <
        crude_mc_1d(f_cuadrado, 0, 1, N//10, seed=1)[1],
        "err(N) < err(N/10) ✓"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  2. PROPIEDADES ESTADÍSTICAS DEL ESTIMADOR
# ══════════════════════════════════════════════════════════════════════════════

def test_propiedades_estimador():
    section("2 · PROPIEDADES ESTADÍSTICAS DEL ESTIMADOR")

    # ── Sin sesgo: E[Î] = I ───────────────────────────────────────────────
    R   = 500
    N   = 10_000
    reps = replicaciones_mc(f_cuadrado, 0, 1, N, R, seed=42)
    media_reps = media(reps)
    check_approx(
        "Sin sesgo: E[Î] ≈ I = 1/3  (500 réplicas, n=10k)",
        media_reps, I_cuadrado, tol_pct=1.0
    )

    # ── Varianza decrece con n ────────────────────────────────────────────
    var_100   = varianza(replicaciones_mc(f_cuadrado, 0, 1, 100,    50, seed=1))
    var_1000  = varianza(replicaciones_mc(f_cuadrado, 0, 1, 1_000,  50, seed=1))
    var_10000 = varianza(replicaciones_mc(f_cuadrado, 0, 1, 10_000, 50, seed=1))
    check_bool(
        "Var(Î) decrece: n=100 > n=1000 > n=10000",
        var_100 > var_1000 > var_10000,
        f"Var100={var_100:.2e}  Var1k={var_1000:.2e}  Var10k={var_10000:.2e}"
    )

    # ── Varianza teórica ≈ (b-a)²·Var(f)/n ──────────────────────────────
    # Para f(x)=x² en [0,1]:  Var(f) = E[x⁴] - (E[x²])² = 1/5 - 1/9 = 4/45
    var_f_teo  = 1/5 - (1/3)**2   # = 4/45 ≈ 0.08889
    var_est_teo = (1-0)**2 * var_f_teo / N
    var_est_emp = varianza(replicaciones_mc(f_cuadrado, 0, 1, N, 1000, seed=42))
    check_approx(
        "Var(Î) ≈ (b-a)²·Var(f)/n  teórico vs empírico",
        var_est_emp, var_est_teo, tol_pct=10.0
    )

    # ── Error estándar ~ 1/√n ────────────────────────────────────────────
    _, err_1k, _  = crude_mc_1d(f_cuadrado, 0, 1, 1_000,   seed=42)
    _, err_4k, _  = crude_mc_1d(f_cuadrado, 0, 1, 4_000,   seed=42)
    _, err_16k, _ = crude_mc_1d(f_cuadrado, 0, 1, 16_000,  seed=42)
    _, err_64k, _ = crude_mc_1d(f_cuadrado, 0, 1, 64_000,  seed=42)
    # Al cuadruplicar n, el error debería reducirse ~2x
    ratio_1 = err_1k  / err_4k
    ratio_2 = err_4k  / err_16k
    ratio_3 = err_16k / err_64k
    check_in_range(
        "Ratio err(n) / err(4n) ≈ 2  (1k→4k)",
        ratio_1, 1.5, 2.8
    )
    check_in_range(
        "Ratio err(n) / err(4n) ≈ 2  (4k→16k)",
        ratio_2, 1.5, 2.8
    )
    check_in_range(
        "Ratio err(n) / err(4n) ≈ 2  (16k→64k)",
        ratio_3, 1.5, 2.8
    )

    # ── Reproducibilidad ─────────────────────────────────────────────────
    est_a, _, _ = crude_mc_1d(f_seno, 0, math.pi, 10_000, seed=99)
    est_b, _, _ = crude_mc_1d(f_seno, 0, math.pi, 10_000, seed=99)
    check_bool("Reproducibilidad: misma semilla → mismo resultado",
               est_a == est_b, f"est={est_a:.8f} ✓")

    est_c, _, _ = crude_mc_1d(f_seno, 0, math.pi, 10_000, seed=100)
    check_bool("Diferente semilla → diferente resultado",
               est_a != est_c, "✓")


# ══════════════════════════════════════════════════════════════════════════════
#  3. INTEGRALES MULTIDIMENSIONALES
# ══════════════════════════════════════════════════════════════════════════════

def test_integrales_nd():
    section("3 · INTEGRALES MULTIDIMENSIONALES (2D y 3D)")
    N = 500_000

    # ── 2D: ∫∫[0,1]² (x+y) dxdy = 1 ─────────────────────────────────────
    f2d_1   = lambda xy: xy[0] + xy[1]
    est, _  = crude_mc_nd(f2d_1, [(0,1),(0,1)], N, seed=42)
    check_approx("∫∫[0,1]² (x+y) dxdy = 1.0",
                 est, 1.0, tol_pct=1.5)

    # ── 2D: ∫∫[0,1]² (x²+y²) dxdy = 2/3 ─────────────────────────────────
    f2d_2   = lambda xy: xy[0]**2 + xy[1]**2
    est, _  = crude_mc_nd(f2d_2, [(0,1),(0,1)], N, seed=42)
    check_approx("∫∫[0,1]² (x²+y²) dxdy = 2/3",
                 est, 2/3, tol_pct=1.5)

    # ── 2D: ∫∫[0,2]×[0,3] 1 dxdy = 6  (volumen del rectángulo) ──────────
    f2d_3   = lambda xy: 1.0
    est, _  = crude_mc_nd(f2d_3, [(0,2),(0,3)], N, seed=42)
    check_approx("∫∫[0,2]×[0,3] 1 dxdy = 6 (área)",
                 est, 6.0, tol_pct=0.5)

    # ── 2D: ∫∫[0,1]² x·y dxdy = 1/4 ─────────────────────────────────────
    f2d_4   = lambda xy: xy[0] * xy[1]
    est, _  = crude_mc_nd(f2d_4, [(0,1),(0,1)], N, seed=42)
    check_approx("∫∫[0,1]² x·y dxdy = 1/4",
                 est, 1/4, tol_pct=1.5)

    # ── 3D: ∫∫∫[0,1]³ (x+y+z) dxdydz = 3/2 ─────────────────────────────
    f3d_1   = lambda xyz: xyz[0] + xyz[1] + xyz[2]
    est, _  = crude_mc_nd(f3d_1, [(0,1),(0,1),(0,1)], N, seed=42)
    check_approx("∫∫∫[0,1]³ (x+y+z) dxdydz = 3/2",
                 est, 3/2, tol_pct=1.5)

    # ── 3D: ∫∫∫[0,1]³ x·y·z dxdydz = 1/8 ───────────────────────────────
    f3d_2   = lambda xyz: xyz[0] * xyz[1] * xyz[2]
    est, _  = crude_mc_nd(f3d_2, [(0,1),(0,1),(0,1)], N, seed=42)
    check_approx("∫∫∫[0,1]³ x·y·z dxdydz = 1/8",
                 est, 1/8, tol_pct=2.0)

    # ── Error disminuye con n en 2D ───────────────────────────────────────
    _, err_small = crude_mc_nd(f2d_2, [(0,1),(0,1)], 1_000, seed=42)
    _, err_large = crude_mc_nd(f2d_2, [(0,1),(0,1)], 100_000, seed=42)
    check_bool("Error 2D decrece con n",
               err_large < err_small,
               f"err(1k)={err_small:.4f}  err(100k)={err_large:.4f} ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  4. ESTIMACIÓN DE π
# ══════════════════════════════════════════════════════════════════════════════

def test_estimacion_pi():
    section("4 · ESTIMACIÓN DE π (MÉTODO GEOMÉTRICO)")

    for n in [10_000, 100_000, 1_000_000]:
        pi_est, pi_err = estimar_pi(n, seed=42)
        tol = 3.0 if n < 100_000 else 1.0
        check_approx(
            f"π estimado con n={n:>9,}",
            pi_est, math.pi, tol_pct=tol
        )
        check_bool(
            f"Error estándar > 0  (n={n:>9,})",
            pi_err > 0,
            f"err={pi_err:.6f}"
        )
        check_bool(
            f"π ∈ [est - 3·err, est + 3·err]  (n={n:>9,})",
            abs(pi_est - math.pi) <= 3 * pi_err,
            f"|Δ|={abs(pi_est-math.pi):.6f}  3σ={3*pi_err:.6f}"
        )

    # Convergencia: más puntos → más preciso en promedio
    errores = [estimar_pi(50_000, seed=i)[1] for i in range(20)]
    err_promedio = media(errores)
    check_in_range(
        "Error promedio π con n=50k ∈ [0.001, 0.01]",
        err_promedio, 0.001, 0.01
    )

    # Reproducibilidad
    pi1, _ = estimar_pi(10_000, seed=7)
    pi2, _ = estimar_pi(10_000, seed=7)
    check_bool("Reproducibilidad: misma semilla → mismo π",
               pi1 == pi2, "✓")


# ══════════════════════════════════════════════════════════════════════════════
#  5. INTERVALOS DE CONFIANZA
# ══════════════════════════════════════════════════════════════════════════════

def test_intervalos_confianza():
    section("5 · INTERVALOS DE CONFIANZA")

    # ── IC de muestra única (intervalo_confianza_1d) ──────────────────────
    for f, a, b, I_exact, label in [
        (f_cuadrado, 0, 1,       I_cuadrado, "x²  [0,1]"),
        (f_seno,     0, math.pi, I_seno,     "sin [0,π]"),
        (f_exp,      0, 1,       I_exp,      "e^x [0,1]"),
    ]:
        est, lo, hi = intervalo_confianza_1d(f, a, b, n=500_000, alpha=0.05, seed=42)
        check_bool(
            f"IC 95% contiene I_exact  ({label})",
            lo <= I_exact <= hi,
            f"I={I_exact:.5f}  IC=[{lo:.5f}, {hi:.5f}]"
        )
        check_bool(
            f"IC bien formado: lo < est < hi  ({label})",
            lo < est < hi,
            f"[{lo:.5f}, {est:.5f}, {hi:.5f}] ✓"
        )

    # ── IC de replicaciones (ic_replicaciones) ────────────────────────────
    # Cobertura empírica: con 200 réplicas, el IC 95% debe contener I_exact ~95%
    cobertura = 0
    TRIALS    = 200
    for seed in range(TRIALS):
        reps = replicaciones_mc(f_cuadrado, 0, 1, n=5_000, r=30, seed=seed)
        _, lo, hi = ic_replicaciones(reps, alpha=0.05)
        if lo <= I_cuadrado <= hi:
            cobertura += 1
    cob_pct = cobertura / TRIALS
    check_in_range(
        "Cobertura empírica IC 95% ≈ 95%  (200 experimentos)",
        cob_pct, 0.88, 1.00
    )

    # ── IC se estrecha con más réplicas ───────────────────────────────────
    reps_10  = replicaciones_mc(f_cuadrado, 0, 1, n=5_000, r=10,  seed=42)
    reps_100 = replicaciones_mc(f_cuadrado, 0, 1, n=5_000, r=100, seed=42)
    _, lo10,  hi10  = ic_replicaciones(reps_10)
    _, lo100, hi100 = ic_replicaciones(reps_100)
    width_10  = hi10  - lo10
    width_100 = hi100 - lo100
    check_bool(
        "IC más estrecho con más réplicas: ancho(r=10) > ancho(r=100)",
        width_10 > width_100,
        f"ancho(10)={width_10:.5f}  ancho(100)={width_100:.5f}"
    )

    # ── Media de replicaciones es buen estimador ──────────────────────────
    reps_large = replicaciones_mc(f_seno, 0, math.pi, n=50_000, r=100, seed=42)
    xbar, _, _ = ic_replicaciones(reps_large)
    check_approx(
        "Media de 100 réplicas ≈ I_exact (sin(x))",
        xbar, I_seno, tol_pct=0.5
    )


# ══════════════════════════════════════════════════════════════════════════════
#  6. CASOS BORDE Y ROBUSTEZ
# ══════════════════════════════════════════════════════════════════════════════

def test_casos_borde():
    section("6 · CASOS BORDE Y ROBUSTEZ")

    # ── Función cero ──────────────────────────────────────────────────────
    f_cero  = lambda x: 0.0
    est, err, var_f = crude_mc_1d(f_cero, 0, 1, 10_000, seed=42)
    check_bool("∫[0,1] 0 dx = 0  (exactamente)",
               est == 0.0,  f"est={est}")
    check_bool("Varianza f=0 es exactamente 0",
               var_f == 0.0, f"var_f={var_f}")

    # ── Función constante ─────────────────────────────────────────────────
    f_5     = lambda x: 5.0
    est, _, _ = crude_mc_1d(f_5, 2, 7, 10_000, seed=42)
    check_approx("∫[2,7] 5 dx = 25  (exacto siempre)", est, 25.0, tol_pct=0.0)

    # ── n=2 (mínimo) ──────────────────────────────────────────────────────
    try:
        est, err, _ = crude_mc_1d(f_cuadrado, 0, 1, 2, seed=42)
        check_bool("n=2 no lanza excepción", True,
                   f"est={est:.4f}  err={err:.4f}")
    except Exception as e:
        check_bool("n=2 no lanza excepción", False, str(e))

    # ── Intervalo negativo (a < 0) ────────────────────────────────────────
    f_neg   = lambda x: x
    est, _, _ = crude_mc_1d(f_neg, -1, 1, 500_000, seed=42)
    check_approx("∫[-1,1] x dx = 0  (función impar)", est, 0.0, tol_pct=1.0)

    # ── Intervalo grande ──────────────────────────────────────────────────
    f_gauss = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    # ∫[-10,10] Normal(0,1) dx ≈ 1 (virtualmente toda la masa)
    est, _, _ = crude_mc_1d(f_gauss, -10, 10, 500_000, seed=42)
    check_approx("∫[-10,10] φ(x) dx ≈ 1.0", est, 1.0, tol_pct=0.5)

    # ── Función con salto (indicadora) ────────────────────────────────────
    # ∫[0,1] I(x > 0.5) dx = 0.5
    f_ind   = lambda x: 1.0 if x > 0.5 else 0.0
    est, _, _ = crude_mc_1d(f_ind, 0, 1, 500_000, seed=42)
    check_approx("∫[0,1] I(x>0.5) dx = 0.5", est, 0.5, tol_pct=1.0)

    # ── ND con una sola dimensión ≡ 1D ────────────────────────────────────
    est_nd, _ = crude_mc_nd(lambda x: x[0]**2, [(0,1)], 500_000, seed=42)
    est_1d, _, _ = crude_mc_1d(f_cuadrado, 0, 1, 500_000, seed=42)
    check_approx("ND 1-dim ≈ 1D  (x²)",
                 est_nd, est_1d, tol_pct=0.5)

    # ── Error estándar es no negativo siempre ─────────────────────────────
    for f_test, a, b in [(f_cuadrado, 0, 1), (f_seno, 0, math.pi), (f_cero, 0, 1)]:
        _, err, _ = crude_mc_1d(f_test, a, b, 1_000, seed=42)
        check_bool(f"Error estándar ≥ 0 para f={f_test.__name__ if hasattr(f_test,'__name__') else 'λ'}",
                   err >= 0, f"err={err:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. CONVERGENCIA O(1/√n)
# ══════════════════════════════════════════════════════════════════════════════

def test_convergencia():
    section("7 · CONVERGENCIA O(1/√n)")

    # Usamos la media de R réplicas para medir el error absoluto medio
    R = 200
    I_exact = I_cuadrado

    ns = [100, 400, 1_600, 6_400, 25_600]
    errores_absolutos = []
    for n in ns:
        reps = replicaciones_mc(f_cuadrado, 0, 1, n, R, seed=42)
        err_abs = sum(abs(r - I_exact) for r in reps) / R
        errores_absolutos.append(err_abs)

    # Al cuadruplicar n, el error medio debe reducirse ≈ 2x
    for i in range(len(ns) - 1):
        ratio = errores_absolutos[i] / errores_absolutos[i + 1]
        check_in_range(
            f"Ratio error(n={ns[i]}) / error(n={ns[i+1]}) ≈ 2",
            ratio, 1.3, 3.5
        )

    # El error absoluto debe decrecer monótonamente (en promedio)
    monotono = all(errores_absolutos[i] > errores_absolutos[i+1]
                   for i in range(len(errores_absolutos)-1))
    check_bool(
        "Error absoluto medio decrece monótonamente con n",
        monotono,
        "  ".join(f"n={ns[i]}:{errores_absolutos[i]:.5f}" for i in range(len(ns)))
    )

    # Con n suficientemente grande, el error debe ser < 0.001
    est_grande, err_grande, _ = crude_mc_1d(f_cuadrado, 0, 1, 10_000_000, seed=42)
    check_approx(
        "n=10M: estimación muy precisa (tol 0.1%)",
        est_grande, I_exact, tol_pct=0.1
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    total  = len(results_summary)
    passed = sum(results_summary)
    failed = total - passed
    pct    = 100 * passed / total if total else 0

    print(f"\n{'═'*80}")
    print(f"  RESUMEN FINAL")
    print(f"{'═'*80}")
    print(f"  Total tests : {total}")
    print(f"  Pasados     : \033[92m{passed}\033[0m")
    print(f"  Fallados    : \033[91m{failed}\033[0m")
    print(f"  Resultado   : {pct:.1f}%")
    print('═'*80)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("  SUITE DE TESTS — CRUDE MONTE CARLO")
    print("  Integrales 1D/ND · Propiedades · π · ICs · Convergencia")
    print("█"*80)

    test_integrales_1d()
    test_propiedades_estimador()
    test_integrales_nd()
    test_estimacion_pi()
    test_intervalos_confianza()
    test_casos_borde()
    test_convergencia()

    print_summary()
