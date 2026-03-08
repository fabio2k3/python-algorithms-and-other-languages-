"""
test_servidores_serie.py
=========================
Suite de tests para el Modelo de dos Servidores en Serie.

Estrategia de validación:
  Para M/M/1 → M/M/1 en serie (red de Jackson):
    - Cada etapa se comporta como M/M/1 independiente con la misma tasa λ.
    - ρ_i = λ / μ_i
    - W_i = 1 / (μ_i - λ)
    - W_total = W1 + W2

  Se usan r replicaciones con T largo y se compara la media empírica
  con el valor teórico con tolerancia del 6-8%.

Bloques:
  1. Correctitud estructural
  2. Valores teóricos (ρ, W1, W2, W_total) en varios puntos
  3. Aditividad W = W1 + W2
  4. Casos límite (etapas simétricas, cuello de botella)
  5. Reproducibilidad y semillas
  6. Consistencia interna (orden temporal S1 → S2)
"""

import sys, math, random
from typing import List
from servidores_serie import (
    simular_serie_doble,
    replicar_serie_doble,
    ResultadoSerieDoble,
    gen_exp,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
_resultados: List[bool] = []

def check(label, got, exp, tol_pct=6.0):
    if abs(exp) < 1e-9:
        ok = abs(got - exp) < tol_pct / 100
    else:
        ok = abs(got - exp) / abs(exp) <= tol_pct / 100
    _resultados.append(ok)
    d = abs(got - exp) / max(abs(exp), 1e-9) * 100
    print(f"  [{PASS if ok else FAIL}] {label:<62} got={got:.5f}  exp={exp:.5f}  Δ={d:.1f}%")

def check_bool(label, cond, detalle=""):
    _resultados.append(cond)
    print(f"  [{PASS if cond else FAIL}] {label:<62} {detalle}")

def check_range(label, got, lo, hi):
    ok = lo <= got <= hi
    _resultados.append(ok)
    print(f"  [{PASS if ok else FAIL}] {label:<62} got={got:.5f}  rango=[{lo:.4f},{hi:.4f}]")

def section(t):
    print(f"\n{'═'*78}\n  {t}\n{'═'*78}")

def media(lst): return sum(lst) / len(lst)

T_SIM = 600.0
R     = 800

def _metricas(lam, mu1, mu2, tol=6.0):
    reps = replicar_serie_doble(gen_exp(lam), gen_exp(mu1), gen_exp(mu2), T_SIM, R, seed=42)
    return (
        media([r.W1   for r in reps]),
        media([r.W2   for r in reps]),
        media([r.W    for r in reps]),
        media([r.rho1 for r in reps]),
        media([r.rho2 for r in reps]),
        reps,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  1. CORRECTITUD ESTRUCTURAL
# ══════════════════════════════════════════════════════════════════════════════

def test_estructura():
    section("1 · CORRECTITUD ESTRUCTURAL")
    r = simular_serie_doble(gen_exp(2.0), gen_exp(4.0), gen_exp(5.0), 300.0, seed=7)

    check_bool("NA > 0",                r.NA > 0,         f"NA={r.NA}")
    check_bool("ND ≤ NA",               r.ND <= r.NA,     f"ND={r.ND} NA={r.NA}")
    check_bool("ND > 0",                r.ND > 0,         f"ND={r.ND}")
    check_bool("tiempo_sim > 0",        r.tiempo_sim > 0, f"t={r.tiempo_sim:.2f}")

    check_bool("tiempos_en_S1 todos > 0",
               all(w > 0 for w in r.tiempos_en_S1), f"n={len(r.tiempos_en_S1)}")
    check_bool("tiempos_en_S2 todos > 0",
               all(w > 0 for w in r.tiempos_en_S2), f"n={len(r.tiempos_en_S2)}")
    check_bool("tiempos_totales todos > 0",
               all(w > 0 for w in r.tiempos_totales), f"n={len(r.tiempos_totales)}")

    check_bool("W_total = W1 + W2  (aprox por media de réplica)",
               abs(r.W - (r.W1 + r.W2)) < 0.5,
               f"W={r.W:.4f} W1+W2={r.W1+r.W2:.4f}")

    check_range("ρ1 ∈ (0,1)",  r.rho1, 0.0, 1.0)
    check_range("ρ2 ∈ (0,1)",  r.rho2, 0.0, 1.0)

    check_bool("W1 > 0 y W2 > 0",
               r.W1 > 0 and r.W2 > 0, f"W1={r.W1:.4f} W2={r.W2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  2. VALORES TEÓRICOS (Red de Jackson M/M/1 → M/M/1)
# ══════════════════════════════════════════════════════════════════════════════

def test_valores_teoricos():
    section("2 · VALORES TEÓRICOS (Red de Jackson)")

    casos = [
        # (lam, mu1, mu2, tol%)
        (2.0, 4.0, 5.0,  6.0),
        (1.0, 3.0, 4.0,  6.0),
        (2.0, 3.0, 6.0,  8.0),
        (1.5, 4.0, 3.0,  7.0),   # cuello de botella en S2
        (0.5, 2.0, 2.0,  6.0),   # etapas simétricas
    ]

    for lam, mu1, mu2, tol in casos:
        rho1_t = lam / mu1
        rho2_t = lam / mu2
        W1_t   = 1.0 / (mu1 - lam)
        W2_t   = 1.0 / (mu2 - lam)
        Wt_t   = W1_t + W2_t

        W1_e, W2_e, Wt_e, rho1_e, rho2_e, _ = _metricas(lam, mu1, mu2)

        check(f"λ={lam} μ1={mu1} μ2={mu2} → ρ1",     rho1_e, rho1_t, tol)
        check(f"λ={lam} μ1={mu1} μ2={mu2} → ρ2",     rho2_e, rho2_t, tol)
        check(f"λ={lam} μ1={mu1} μ2={mu2} → W1",     W1_e,   W1_t,   tol)
        check(f"λ={lam} μ1={mu1} μ2={mu2} → W2",     W2_e,   W2_t,   tol)
        check(f"λ={lam} μ1={mu1} μ2={mu2} → W_total",Wt_e,   Wt_t,   tol)

# ══════════════════════════════════════════════════════════════════════════════
#  3. ADITIVIDAD  W = W1 + W2
# ══════════════════════════════════════════════════════════════════════════════

def test_aditividad():
    section("3 · ADITIVIDAD  W_total ≈ W1 + W2")

    for lam, mu1, mu2 in [(2.0, 4.0, 5.0), (1.0, 3.0, 6.0), (1.5, 4.0, 3.0)]:
        W1_e, W2_e, Wt_e, _, _, _ = _metricas(lam, mu1, mu2)
        check(f"λ={lam} μ1={mu1} μ2={mu2}: W ≈ W1+W2",
              Wt_e, W1_e + W2_e, tol_pct=3.0)

# ══════════════════════════════════════════════════════════════════════════════
#  4. CASOS LÍMITE
# ══════════════════════════════════════════════════════════════════════════════

def test_casos_limite():
    section("4 · CASOS LÍMITE")

    # Etapas simétricas: W1 ≈ W2 y ρ1 ≈ ρ2
    W1_e, W2_e, _, rho1_e, rho2_e, _ = _metricas(1.0, 3.0, 3.0)
    check("Etapas simétricas: W1 ≈ W2", W1_e, W2_e, tol_pct=5.0)
    check("Etapas simétricas: ρ1 ≈ ρ2", rho1_e, rho2_e, tol_pct=5.0)

    # Cuello de botella en S1 (μ1 ≪ μ2): ρ1 ≫ ρ2 y W1 ≫ W2
    W1_e2, W2_e2, _, rho1_e2, rho2_e2, _ = _metricas(2.0, 3.0, 10.0)
    check_bool("Cuello botella S1: ρ1 > ρ2",
               rho1_e2 > rho2_e2, f"ρ1={rho1_e2:.4f} ρ2={rho2_e2:.4f}")
    check_bool("Cuello botella S1: W1 > W2",
               W1_e2 > W2_e2, f"W1={W1_e2:.4f} W2={W2_e2:.4f}")

    # Cuello de botella en S2
    W1_e3, W2_e3, _, rho1_e3, rho2_e3, _ = _metricas(2.0, 10.0, 3.0)
    check_bool("Cuello botella S2: ρ2 > ρ1",
               rho2_e3 > rho1_e3, f"ρ1={rho1_e3:.4f} ρ2={rho2_e3:.4f}")
    check_bool("Cuello botella S2: W2 > W1",
               W2_e3 > W1_e3, f"W1={W1_e3:.4f} W2={W2_e3:.4f}")

    # Mayor λ → mayor tiempo en sistema
    _, _, Wt_bajo, _, _, _ = _metricas(0.5, 3.0, 4.0)
    _, _, Wt_alto, _, _, _ = _metricas(2.0, 3.0, 4.0)
    check_bool("Mayor λ → mayor W_total",
               Wt_alto > Wt_bajo, f"W(λ=2)={Wt_alto:.4f} W(λ=0.5)={Wt_bajo:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  5. REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def test_reproducibilidad():
    section("5 · REPRODUCIBILIDAD")

    r1 = simular_serie_doble(gen_exp(2.0), gen_exp(4.0), gen_exp(5.0), 100.0, seed=77)
    r2 = simular_serie_doble(gen_exp(2.0), gen_exp(4.0), gen_exp(5.0), 100.0, seed=77)
    check_bool("Misma semilla → mismo NA",  r1.NA == r2.NA, f"NA={r1.NA}")
    check_bool("Misma semilla → mismo ND",  r1.ND == r2.ND, f"ND={r1.ND}")
    check_bool("Misma semilla → mismo W1",  abs(r1.W1 - r2.W1) < 1e-12, f"W1={r1.W1:.6f}")
    check_bool("Misma semilla → mismo W2",  abs(r1.W2 - r2.W2) < 1e-12, f"W2={r1.W2:.6f}")

    r3 = simular_serie_doble(gen_exp(2.0), gen_exp(4.0), gen_exp(5.0), 100.0, seed=78)
    check_bool("Distinta semilla → distinto resultado",
               r1.NA != r3.NA or abs(r1.W - r3.W) > 1e-6, "✓")

# ══════════════════════════════════════════════════════════════════════════════
#  6. CONSISTENCIA INTERNA
# ══════════════════════════════════════════════════════════════════════════════

def test_consistencia():
    section("6 · CONSISTENCIA INTERNA")

    for seed in [1, 42, 123, 500]:
        r = simular_serie_doble(gen_exp(1.5), gen_exp(3.0), gen_exp(4.0), 300.0, seed=seed)

        check_bool(f"seed={seed}: ND ≤ NA",
                   r.ND <= r.NA, f"ND={r.ND} NA={r.NA}")
        check_bool(f"seed={seed}: |tiempos_S1| coincide con atendidos S1",
                   len(r.tiempos_en_S1) >= r.ND, f"len={len(r.tiempos_en_S1)} ND={r.ND}")
        check_bool(f"seed={seed}: ρ1 y ρ2 ∈ (0,1)",
                   0 < r.rho1 < 1 and 0 < r.rho2 < 1,
                   f"ρ1={r.rho1:.3f} ρ2={r.rho2:.3f}")
        check_bool(f"seed={seed}: W_total > max(W1, W2)",
                   r.W > max(r.W1, r.W2) - 1e-9,
                   f"W={r.W:.4f} W1={r.W1:.4f} W2={r.W2:.4f}")

    # Throughput ≈ λ
    reps = replicar_serie_doble(gen_exp(2.0), gen_exp(4.0), gen_exp(5.0), T_SIM, R, seed=42)
    thr  = media([r.throughput for r in reps])
    check("Throughput ≈ λ=2.0", thr, 2.0, tol_pct=6.0)

# ══════════════════════════════════════════════════════════════════════════════
#  Resumen
# ══════════════════════════════════════════════════════════════════════════════

def resumen():
    total = len(_resultados); ok = sum(_resultados); fail = total - ok
    print(f"\n{'═'*78}")
    print(f"  RESUMEN  |  Total={total}  Pasados=\033[92m{ok}\033[0m  Fallados=\033[91m{fail}\033[0m  ({100*ok/total:.1f}%)")
    print(f"{'═'*78}")
    sys.exit(0 if fail == 0 else 1)

if __name__ == "__main__":
    print("\n" + "█"*78)
    print("  TEST — MODELO DE DOS SERVIDORES EN SERIE")
    print("█"*78)
    test_estructura()
    test_valores_teoricos()
    test_aditividad()
    test_casos_limite()
    test_reproducibilidad()
    test_consistencia()
    resumen()
