"""
test_simulacion.py
==================
Suite de tests completa para todos los métodos de simulación estocástica:

  1. TTI — Método de la Transformada Inversa
     · Exponencial, Uniforme(a,b), Weibull, Cauchy
  2. Aceptación y Rechazo
     · Beta(α,β), Normal(0,1), distribución personalizada
  3. Poisson — Producto de Uniformes
     · Correctitud de la PMF, media, varianza, no negatividad
  4. Proceso de Poisson Homogéneo
     · Media, varianza, tiempos crecientes, eventos en [0,T]
  5. Proceso de Poisson No Homogéneo (Thinning)
     · Media, varianza, eventos en [0,T]

Estrategia de validación:
  - Se generan muestras grandes (N ≥ 50,000) para reducir el error estadístico.
  - Se comparan media/varianza muestrales con los valores teóricos
    usando tolerancias relativas porcentuales.
  - Tests de propiedades estructurales (positividad, orden, rango).

Ejecutar:
    python test_simulacion.py
"""

import sys
import math
import random
import traceback
from typing import List

from tti                import (muestras_exponencial, muestras_uniforme_ab,
                                 muestras_weibull, muestras_cauchy,
                                 media as tti_media, varianza as tti_var)

from aceptacion_rechazo import (muestras_beta, muestras_normal,
                                 muestras_custom, _exp1_sampler, _exp1_pdf,
                                 media as ar_media, varianza as ar_var,
                                 aceptacion_rechazo)

from poisson_uniforms   import (muestras_poisson, pmf_poisson, frecuencias,
                                 media as poi_media, varianza as poi_var)

from poisson_process    import (proceso_poisson_homogeneo,
                                 proceso_poisson_no_homogeneo,
                                 replicaciones_homogeneo,
                                 replicaciones_no_homogeneo,
                                 conteo_esperado_homogeneo,
                                 conteo_esperado_no_homogeneo,
                                 media_conteos, varianza_conteos)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades de test
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
results_summary: List[bool] = []


def check_approx(label: str, got: float, expected: float, tol_pct: float = 3.0):
    """
    Valida que |got - expected| / |expected| ≤ tol_pct/100.
    Para expected≈0 usa tolerancia absoluta tol_pct/100.
    """
    if abs(expected) < 1e-10:
        ok = abs(got - expected) < tol_pct / 100
    else:
        ok = abs(got - expected) / abs(expected) <= tol_pct / 100
    status = PASS if ok else FAIL
    results_summary.append(ok)
    diff = f"{abs(got - expected):.5f} ({abs(got-expected)/max(abs(expected),1e-10)*100:.1f}%)"
    print(f"  [{status}] {label:<58} | got={got:.5f}  exp={expected:.5f}  Δ={diff}")


def check_bool(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results_summary.append(condition)
    print(f"  [{status}] {label:<58} | {detail}")


def section(title: str):
    print(f"\n{'═'*78}")
    print(f"  {title}")
    print('═'*78)


# ══════════════════════════════════════════════════════════════════════════════
#  1. TESTS TTI
# ══════════════════════════════════════════════════════════════════════════════

def test_tti():
    section("1 · TTI — MÉTODO DE LA TRANSFORMADA INVERSA")
    N = 100_000

    # ── Exponencial ───────────────────────────────────────────────────────
    for lam in [0.5, 1.0, 3.0, 10.0]:
        m = muestras_exponencial(lam, N, seed=42)
        check_approx(f"  Exp(λ={lam}) media", tti_media(m), 1/lam, tol_pct=2.0)
        check_approx(f"  Exp(λ={lam}) varianza", tti_var(m), 1/lam**2, tol_pct=5.0)
        check_bool(f"  Exp(λ={lam}) todos positivos",
                   all(x > 0 for x in m), "x > 0 ✓")

    # ── Uniforme(a,b) ─────────────────────────────────────────────────────
    for a, b in [(0, 1), (2, 5), (-3, 3)]:
        m = muestras_uniforme_ab(a, b, N, seed=42)
        check_approx(f"  Uniforme({a},{b}) media", tti_media(m), (a+b)/2, tol_pct=1.0)
        check_approx(f"  Uniforme({a},{b}) varianza",
                     tti_var(m), (b-a)**2/12, tol_pct=3.0)
        check_bool(f"  Uniforme({a},{b}) rango correcto",
                   all(a <= x <= b for x in m), f"[{a},{b}] ✓")

    # ── Weibull ───────────────────────────────────────────────────────────
    for lam, k in [(1.0, 1.0), (2.0, 2.0), (1.0, 3.0)]:
        m = muestras_weibull(lam, k, N, seed=42)
        media_teo = (1/lam) * math.gamma(1 + 1/k)
        check_approx(f"  Weibull(λ={lam},k={k}) media",
                     tti_media(m), media_teo, tol_pct=2.0)
        check_bool(f"  Weibull(λ={lam},k={k}) todos positivos",
                   all(x > 0 for x in m), "x > 0 ✓")

    # ── Cauchy ────────────────────────────────────────────────────────────
    # Cauchy no tiene media finita; verificamos mediana ≈ x0
    x0, gamma = 0.0, 1.0
    m = muestras_cauchy(x0, gamma, N, seed=42)
    sorted_m = sorted(m)
    mediana = sorted_m[N // 2]
    check_approx("  Cauchy(x0=0,γ=1) mediana ≈ 0",
                 mediana, x0, tol_pct=5.0)


# ══════════════════════════════════════════════════════════════════════════════
#  2. TESTS ACEPTACIÓN Y RECHAZO
# ══════════════════════════════════════════════════════════════════════════════

def test_aceptacion_rechazo():
    section("2 · ACEPTACIÓN Y RECHAZO")
    N = 60_000

    # ── Beta(α, β) ────────────────────────────────────────────────────────
    for a, b in [(2, 5), (0.5, 0.5), (3, 1), (1, 1)]:
        m = muestras_beta(a, b, N, seed=42)
        media_teo = a / (a + b)
        var_teo   = (a * b) / ((a+b)**2 * (a+b+1))
        check_approx(f"  Beta(α={a},β={b}) media",
                     ar_media(m), media_teo, tol_pct=3.0)
        check_approx(f"  Beta(α={a},β={b}) varianza",
                     ar_var(m),   var_teo,   tol_pct=8.0)
        check_bool(f"  Beta(α={a},β={b}) rango [0,1]",
                   all(0 < x < 1 for x in m), "(0,1) ✓")

    # ── Normal(0,1) ───────────────────────────────────────────────────────
    m_norm = muestras_normal(N, seed=42)
    check_approx("  Normal(0,1) media ≈ 0",     ar_media(m_norm), 0.0, tol_pct=5.0)
    check_approx("  Normal(0,1) varianza ≈ 1",  ar_var(m_norm),   1.0, tol_pct=4.0)

    # Verificar simetría: P(X<0) ≈ 0.5
    p_neg = sum(1 for x in m_norm if x < 0) / N
    check_approx("  Normal(0,1) P(X<0) ≈ 0.5", p_neg, 0.5, tol_pct=3.0)

    # ── Distribución personalizada: triangular en [0,1] ──────────────────
    # f(x) = 2x  en [0,1],  g(x) = Uniforme(0,1),  c = 2
    def f_triangular(x):
        return 2 * x if 0 <= x <= 1 else 0.0

    rng42 = random.Random(42)
    m_tri = muestras_custom(
        f_triangular,
        lambda rng: rng.random(),   # g_sampler: Uniforme(0,1)
        lambda x: 1.0,              # g_pdf
        c=2.0,
        n=N,
        seed=42,
    )
    # Triangular(0,1,1): media = 2/3, varianza = 1/18
    check_approx("  Triangular (custom) media ≈ 2/3",
                 ar_media(m_tri), 2/3,  tol_pct=2.0)
    check_approx("  Triangular (custom) varianza ≈ 1/18",
                 ar_var(m_tri),  1/18,  tol_pct=6.0)
    check_bool("  Triangular (custom) rango [0,1]",
               all(0 <= x <= 1 for x in m_tri), "[0,1] ✓")

    # ── Propiedad: los intentos deben ser enteros positivos ───────────────
    rng = random.Random(0)
    import aceptacion_rechazo as AR
    _, intentos = AR.muestra_beta(2, 5, rng)
    check_bool("  Beta: intentos ≥ 1",
               isinstance(intentos, int) and intentos >= 1,
               f"intentos={intentos}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. TESTS POISSON — PRODUCTO DE UNIFORMES
# ══════════════════════════════════════════════════════════════════════════════

def test_poisson_uniformes():
    section("3 · POISSON — MÉTODO DEL PRODUCTO DE UNIFORMES")
    N = 100_000

    # ── Media y varianza ──────────────────────────────────────────────────
    for lam in [0.5, 1.0, 3.0, 7.0, 15.0]:
        m = muestras_poisson(lam, N, seed=42)
        check_approx(f"  Poisson(λ={lam:4.1f}) media",
                     poi_media(m), lam, tol_pct=2.0)
        check_approx(f"  Poisson(λ={lam:4.1f}) varianza",
                     poi_var(m),   lam, tol_pct=4.0)
        check_bool(f"  Poisson(λ={lam:4.1f}) todos ≥ 0",
                   all(k >= 0 for k in m), "k ≥ 0 ✓")
        check_bool(f"  Poisson(λ={lam:4.1f}) son enteros",
                   all(isinstance(k, int) for k in m), "int ✓")

    # ── PMF empírica vs teórica ───────────────────────────────────────────
    lam = 4.0
    m = muestras_poisson(lam, N, seed=42)
    freq = frecuencias(m)
    for k in range(8):
        p_emp = freq.get(k, 0) / N
        p_teo = pmf_poisson(k, lam)
        check_approx(f"  PMF k={k}: P(N={k}|λ={lam})",
                     p_emp, p_teo, tol_pct=8.0)

    # ── Reproducibilidad ─────────────────────────────────────────────────
    m1 = muestras_poisson(3.0, 1000, seed=99)
    m2 = muestras_poisson(3.0, 1000, seed=99)
    check_bool("  Reproducibilidad: misma semilla → mismas muestras",
               m1 == m2, "✓")

    m3 = muestras_poisson(3.0, 1000, seed=100)
    check_bool("  Diferente semilla → diferentes muestras",
               m1 != m3, "✓")


# ══════════════════════════════════════════════════════════════════════════════
#  4. TESTS PROCESO HOMOGÉNEO
# ══════════════════════════════════════════════════════════════════════════════

def test_proceso_homogeneo():
    section("4 · PROCESO DE POISSON HOMOGÉNEO")
    R   = 5_000
    T   = 10.0

    for lam in [0.5, 1.0, 3.0, 8.0]:
        replicas = replicaciones_homogeneo(lam, T, R, seed=42)

        # Media del número de eventos
        e_teo = conteo_esperado_homogeneo(lam, T)
        check_approx(f"  Hom. λ={lam:3.1f} E[N(T)] = λT = {e_teo:.1f}",
                     media_conteos(replicas), e_teo, tol_pct=3.0)

        # Varianza ≈ λT (propiedad de Poisson)
        check_approx(f"  Hom. λ={lam:3.1f} Var[N(T)] ≈ λT",
                     varianza_conteos(replicas), e_teo, tol_pct=8.0)

    # ── Propiedades de una sola trayectoria ───────────────────────────────
    eventos = proceso_poisson_homogeneo(2.0, T, seed=42)

    check_bool("  Todos los eventos en (0, T]",
               all(0 < t <= T for t in eventos),
               f"{len(eventos)} eventos ✓")

    check_bool("  Tiempos estrictamente crecientes",
               all(eventos[i] < eventos[i+1] for i in range(len(eventos)-1)),
               "orden ✓")

    # ── Lista vacía cuando λ→0 ────────────────────────────────────────────
    # λ muy pequeño → casi todas las réplicas tienen 0 eventos en T pequeño
    # P(N(T)=0) = e^(-λT) ≈ e^(-0.0001) ≈ 0.9999
    replicas_tiny = replicaciones_homogeneo(0.001, 0.1, 500, seed=42)
    perc_vacias = sum(1 for r in replicas_tiny if len(r) == 0) / 500
    p_cero_teo = math.exp(-0.001 * 0.1)   # ≈ 0.9999
    check_approx("  λ=0.001, T=0.1: P(réplica vacía) ≈ e^(-λT)",
                 perc_vacias, p_cero_teo, tol_pct=1.0)

    # ── Reproducibilidad ─────────────────────────────────────────────────
    e1 = proceso_poisson_homogeneo(2.0, T, seed=7)
    e2 = proceso_poisson_homogeneo(2.0, T, seed=7)
    check_bool("  Reproducibilidad: misma semilla → mismos tiempos", e1 == e2, "✓")


# ══════════════════════════════════════════════════════════════════════════════
#  5. TESTS PROCESO NO HOMOGÉNEO
# ══════════════════════════════════════════════════════════════════════════════

def test_proceso_no_homogeneo():
    section("5 · PROCESO DE POISSON NO HOMOGÉNEO (Thinning)")
    R = 5_000

    # ── Caso degenerado: λ(t) = constante → equivale al homogéneo ─────────
    LAM_CONST = 3.0
    T_CONST   = 5.0
    f_const   = lambda t: LAM_CONST

    replicas_nh  = replicaciones_no_homogeneo(f_const, LAM_CONST, T_CONST, R, seed=42)
    replicas_hom = replicaciones_homogeneo(LAM_CONST, T_CONST, R, seed=42)

    check_approx("  NH constante ≈ Homogéneo: E[N(T)]",
                 media_conteos(replicas_nh),
                 media_conteos(replicas_hom),
                 tol_pct=3.0)

    # ── λ(t) = 2 + sin(t) ─────────────────────────────────────────────────
    LAM_MAX = 3.0
    T_SEN   = 2 * math.pi

    def lam_senoidal(t): return 2.0 + math.sin(t)

    replicas_sen = replicaciones_no_homogeneo(
        lam_senoidal, LAM_MAX, T_SEN, R, seed=42)
    e_teo = conteo_esperado_no_homogeneo(lam_senoidal, T_SEN)

    check_approx("  NH λ(t)=2+sin(t) E[N(T)] teórico",
                 media_conteos(replicas_sen), e_teo, tol_pct=3.0)

    # ── λ(t) = t (rampa lineal) ────────────────────────────────────────────
    T_RAMP  = 4.0
    LAM_RAMP = T_RAMP
    def lam_ramp(t): return t  # λ_max = T

    replicas_ramp = replicaciones_no_homogeneo(
        lam_ramp, LAM_RAMP, T_RAMP, R, seed=42)
    # E[N(T)] = ∫₀ᵀ t dt = T²/2
    e_teo_ramp = T_RAMP**2 / 2
    check_approx("  NH λ(t)=t E[N(T)] = T²/2",
                 media_conteos(replicas_ramp), e_teo_ramp, tol_pct=3.0)

    # ── Propiedades de trayectoria ─────────────────────────────────────────
    eventos_nh = proceso_poisson_no_homogeneo(
        lam_senoidal, LAM_MAX, T_SEN, seed=42)

    check_bool("  Todos los eventos en (0, T]",
               all(0 < t <= T_SEN for t in eventos_nh),
               f"{len(eventos_nh)} eventos ✓")

    if len(eventos_nh) > 1:
        check_bool("  Tiempos estrictamente crecientes",
                   all(eventos_nh[i] < eventos_nh[i+1]
                       for i in range(len(eventos_nh)-1)),
                   "orden ✓")
    else:
        check_bool("  Orden (≤1 evento, trivialmente válido)", True, "✓")

    # ── λ(t) = 0 en todo momento → no debe haber eventos ─────────────────
    def lam_cero(t): return 0.0
    replicas_cero = replicaciones_no_homogeneo(
        lam_cero, 1.0, 5.0, 100, seed=42)
    check_bool("  λ(t)=0 → 0 eventos en todas las réplicas",
               all(len(r) == 0 for r in replicas_cero),
               "0 eventos ✓")

    # ── Reproducibilidad ─────────────────────────────────────────────────
    e1 = proceso_poisson_no_homogeneo(lam_senoidal, LAM_MAX, T_SEN, seed=77)
    e2 = proceso_poisson_no_homogeneo(lam_senoidal, LAM_MAX, T_SEN, seed=77)
    check_bool("  Reproducibilidad: misma semilla → mismos tiempos", e1 == e2, "✓")


# ══════════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    total  = len(results_summary)
    passed = sum(results_summary)
    failed = total - passed
    pct    = 100 * passed / total if total else 0

    print(f"\n{'═'*78}")
    print(f"  RESUMEN FINAL")
    print(f"{'═'*78}")
    print(f"  Total tests : {total}")
    print(f"  Pasados     : \033[92m{passed}\033[0m")
    print(f"  Fallados    : \033[91m{failed}\033[0m")
    print(f"  Resultado   : {pct:.1f}%")
    print('═'*78)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    print("\n" + "█"*78)
    print("  SUITE DE TESTS — SIMULACIÓN ESTOCÁSTICA")
    print("  TTI · Aceptación/Rechazo · Poisson Uniformes · Procesos Poisson")
    print("█"*78)

    test_tti()
    test_aceptacion_rechazo()
    test_poisson_uniformes()
    test_proceso_homogeneo()
    test_proceso_no_homogeneo()

    print_summary()
