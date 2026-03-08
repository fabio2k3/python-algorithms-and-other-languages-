"""
test_servidores_paralelo.py
============================
Suite de tests para el Modelo de dos Servidores en Paralelo (M/M/2).

Estrategia de validación:
  Para M/M/2 existen fórmulas exactas de Erlang-C:
      ρ   = λ / (2·μ)        (tasa de ocupación por servidor)
      C   = P(esperar) = 2ρ²/(1 + 2ρ - 2ρ² + 2ρ²)  — simplificada
      W_q = C / (2μ·(1-ρ))   — fórmula Erlang-C completa
      W   = W_q + 1/μ

  Para simplificar se usa comparación directa con resultados analíticos
  calculados aquí y tolerancias más generosas (8-10%) dado que las
  fórmulas de M/M/2 son más complejas.

Bloques:
  1. Correctitud estructural
  2. Utilización de servidores (ρ ≈ λ/2μ)
  3. Valores teóricos M/M/2 (W, W_q)
  4. Comparación M/M/1 vs M/M/2 (mismo λ, misma carga total)
  5. Reproducibilidad y semillas
  6. Consistencia interna (cola común, orden FIFO, tiempos)
"""

import sys, math, random
from typing import List
from servidores_paralelo import (
    simular_paralelo,
    replicar_paralelo,
    ResultadoParalelo,
    gen_exp,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Fórmulas exactas M/M/2
# ══════════════════════════════════════════════════════════════════════════════

def mm2_teorico(lam, mu):
    """
    Devuelve (W, W_q, rho) teóricos para M/M/2.
    ρ = λ/(2μ)  debe ser < 1.
    Fórmulas de Erlang-C para c=2 servidores.
    """
    rho = lam / (2 * mu)
    assert rho < 1, "Sistema inestable"
    # P0 (prob de sistema vacío)
    P0 = 1.0 / (1.0 + 2*rho + 2*rho**2 / (1 - rho))
    # P(esperar) — Erlang-C con c=2
    C = (2 * rho**2 / (1 - rho)) * P0
    W_q = C / (2 * mu * (1 - rho))
    W   = W_q + 1.0 / mu
    return W, W_q, rho

# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
_resultados: List[bool] = []

def check(label, got, exp, tol_pct=8.0):
    if abs(exp) < 1e-9:
        ok = abs(got - exp) < tol_pct / 100
    else:
        ok = abs(got - exp) / abs(exp) <= tol_pct / 100
    _resultados.append(ok)
    d = abs(got - exp) / max(abs(exp), 1e-9) * 100
    print(f"  [{PASS if ok else FAIL}] {label:<64} got={got:.5f}  exp={exp:.5f}  Δ={d:.1f}%")

def check_bool(label, cond, detalle=""):
    _resultados.append(cond)
    print(f"  [{PASS if cond else FAIL}] {label:<64} {detalle}")

def check_range(label, got, lo, hi):
    ok = lo <= got <= hi
    _resultados.append(ok)
    print(f"  [{PASS if ok else FAIL}] {label:<64} got={got:.5f}  rango=[{lo:.4f},{hi:.4f}]")

def section(t):
    print(f"\n{'═'*80}\n  {t}\n{'═'*80}")

def media(lst): return sum(lst) / len(lst)

T_SIM = 600.0
R     = 800

def _metricas(lam, mu):
    reps = replicar_paralelo(gen_exp(lam), gen_exp(mu), T_SIM, R, seed=42)
    return (
        media([r.W    for r in reps]),
        media([r.W_q  for r in reps]),
        media([r.rho  for r in reps]),
        media([r.rho1 for r in reps]),
        media([r.rho2 for r in reps]),
        reps,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  1. CORRECTITUD ESTRUCTURAL
# ══════════════════════════════════════════════════════════════════════════════

def test_estructura():
    section("1 · CORRECTITUD ESTRUCTURAL")
    r = simular_paralelo(gen_exp(3.0), gen_exp(2.0), 300.0, seed=7)

    check_bool("NA > 0",         r.NA > 0,       f"NA={r.NA}")
    check_bool("ND ≤ NA",        r.ND <= r.NA,   f"ND={r.ND} NA={r.NA}")
    check_bool("ND > 0",         r.ND > 0,       f"ND={r.ND}")
    check_bool("tiempo_sim > 0", r.tiempo_sim > 0, f"t={r.tiempo_sim:.2f}")

    check_bool("Tiempos espera cola ≥ 0",
               all(w >= 0 for w in r.tiempos_espera_cola), "✓")
    check_bool("Tiempos en sistema > 0",
               all(w > 0  for w in r.tiempos_en_sistema), "✓")
    check_bool("|tiempos_sistema| == ND",
               len(r.tiempos_en_sistema) == r.ND, f"len={len(r.tiempos_en_sistema)} ND={r.ND}")
    check_bool("|tiempos_espera| == ND",
               len(r.tiempos_espera_cola) == r.ND, f"len={len(r.tiempos_espera_cola)} ND={r.ND}")

    check_range("ρ1 ∈ (0,1)",   r.rho1, 0.0, 1.0)
    check_range("ρ2 ∈ (0,1)",   r.rho2, 0.0, 1.0)
    check_range("ρ_media ∈ (0,1)", r.rho, 0.0, 1.0)
    check_bool("W_q ≤ W",       r.W_q <= r.W + 1e-9, f"W_q={r.W_q:.4f} W={r.W:.4f}")
    check_bool("W > 0",         r.W > 0, f"W={r.W:.4f}")

    check_bool("area_n ≥ 0", r.area_n >= 0, f"area_n={r.area_n:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  2. UTILIZACIÓN DE SERVIDORES
# ══════════════════════════════════════════════════════════════════════════════

def test_utilizacion():
    section("2 · UTILIZACIÓN DE SERVIDORES  ρ ≈ λ/(2μ)")

    for lam, mu in [(2.0, 2.0), (3.0, 4.0), (1.0, 2.0), (4.0, 5.0)]:
        rho_teo = lam / (2 * mu)
        _, _, rho_e, rho1_e, rho2_e, _ = _metricas(lam, mu)

        check(f"λ={lam} μ={mu}: ρ_media ≈ ρ_teo={rho_teo:.3f}",
              rho_e, rho_teo, tol_pct=7.0)
        # S1 siempre atiende primero (preferencia en código), por lo que
        # ρ1 ≥ ρ2 es esperado; ambos deben estar en (0,1) y su media ≈ ρ_teo
        check_bool(f"λ={lam} μ={mu}: ρ1 ≥ ρ2 (S1 tiene preferencia de asignación)",
                   rho1_e >= rho2_e - 0.01,
                   f"ρ1={rho1_e:.4f} ρ2={rho2_e:.4f}")
        check_range(f"λ={lam} μ={mu}: ρ2 > 0  (S2 también trabaja)",
                    rho2_e, 0.01, 1.0)

# ══════════════════════════════════════════════════════════════════════════════
#  3. VALORES TEÓRICOS M/M/2
# ══════════════════════════════════════════════════════════════════════════════

def test_valores_mm2():
    section("3 · VALORES TEÓRICOS M/M/2  (Erlang-C)")

    casos = [
        (2.0, 2.0,  9.0),   # ρ=0.50
        (3.0, 4.0,  9.0),   # ρ=0.375
        (3.0, 2.0, 10.0),   # ρ=0.75  (alta carga → mayor varianza)
        (1.0, 3.0,  8.0),   # ρ=0.167
    ]

    for lam, mu, tol in casos:
        W_t, Wq_t, rho_t = mm2_teorico(lam, mu)
        W_e, Wq_e, rho_e, _, _, _ = _metricas(lam, mu)

        check(f"λ={lam} μ={mu}: W",   W_e,   W_t,   tol)
        check(f"λ={lam} μ={mu}: W_q", Wq_e,  Wq_t,  tol)
        check(f"λ={lam} μ={mu}: ρ",   rho_e, rho_t, tol)

# ══════════════════════════════════════════════════════════════════════════════
#  4. COMPARACIÓN M/M/1 vs M/M/2
# ══════════════════════════════════════════════════════════════════════════════

def test_comparacion_mm1_mm2():
    section("4 · COMPARACIÓN M/M/2 vs M/M/1 equivalente")
    from servidor_simple import replicar_servidor_simple as rep_s1
    from servidor_simple import gen_exp as gen_exp_s1

    # M/M/2 con λ=2, μ=2  vs  M/M/1 con λ=2, μ=4 (mismo throughput, misma carga total)
    lam = 2.0
    mu_par = 2.0    # cada servidor en paralelo
    mu_uni = 4.0    # servidor único con capacidad doble

    reps_par = replicar_paralelo(gen_exp(lam), gen_exp(mu_par), T_SIM, R, seed=42)
    reps_uni = rep_s1(gen_exp_s1(lam), gen_exp_s1(mu_uni), T_SIM, R, seed=42)

    W_par = media([r.W for r in reps_par])
    W_uni = media([r.W for r in reps_uni])

    # M/M/2 debería tener W menor que M/M/1 con μ=2 pero comparable a μ=4
    check_bool("M/M/2 (2×μ=2) da W razonable < M/M/1 (μ=2) hipotético",
               W_par < 1.0 / (mu_uni - lam) * 3,   # cota holgada
               f"W_M/M/2={W_par:.4f}  W_M/M/1(μ=4)={W_uni:.4f}")

    # Carga alta: M/M/2 con λ=3, μ=2 (ρ=0.75) vs M/M/1 con λ=3, μ=4 (ρ=0.75)
    reps_p2 = replicar_paralelo(gen_exp(3.0), gen_exp(2.0), T_SIM, R, seed=42)
    reps_u2 = rep_s1(gen_exp_s1(3.0), gen_exp_s1(4.0), T_SIM, R, seed=42)
    Wq_p2 = media([r.W_q for r in reps_p2])
    Wq_u2 = media([r.W_q for r in reps_u2])

    # A misma ρ, M/M/2 tiene W_q menor que M/M/1 (mayor paralelismo)
    check_bool("A igual ρ=0.75: W_q(M/M/2) < W_q(M/M/1)",
               Wq_p2 < Wq_u2, f"W_q M/M/2={Wq_p2:.4f} M/M/1={Wq_u2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  5. REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def test_reproducibilidad():
    section("5 · REPRODUCIBILIDAD")

    r1 = simular_paralelo(gen_exp(3.0), gen_exp(2.0), 150.0, seed=55)
    r2 = simular_paralelo(gen_exp(3.0), gen_exp(2.0), 150.0, seed=55)
    check_bool("Misma semilla → mismo NA",  r1.NA == r2.NA, f"NA={r1.NA}")
    check_bool("Misma semilla → mismo ND",  r1.ND == r2.ND, f"ND={r1.ND}")
    check_bool("Misma semilla → mismo W",   abs(r1.W   - r2.W)   < 1e-12, f"W={r1.W:.6f}")
    check_bool("Misma semilla → mismo W_q", abs(r1.W_q - r2.W_q) < 1e-12, "✓")

    r3 = simular_paralelo(gen_exp(3.0), gen_exp(2.0), 150.0, seed=56)
    check_bool("Distinta semilla → distinto resultado",
               r1.NA != r3.NA or abs(r1.W - r3.W) > 1e-6, "✓")

# ══════════════════════════════════════════════════════════════════════════════
#  6. CONSISTENCIA INTERNA
# ══════════════════════════════════════════════════════════════════════════════

def test_consistencia():
    section("6 · CONSISTENCIA INTERNA")

    for seed in [1, 42, 200, 777]:
        r = simular_paralelo(gen_exp(3.0), gen_exp(2.0), 200.0, seed=seed)

        check_bool(f"seed={seed}: ND ≤ NA",
                   r.ND <= r.NA, f"ND={r.ND} NA={r.NA}")
        check_bool(f"seed={seed}: |espera| == ND",
                   len(r.tiempos_espera_cola) == r.ND,
                   f"len={len(r.tiempos_espera_cola)} ND={r.ND}")
        check_bool(f"seed={seed}: |sistema| == ND",
                   len(r.tiempos_en_sistema) == r.ND,
                   f"len={len(r.tiempos_en_sistema)}")
        check_bool(f"seed={seed}: todos W_q ≤ W (por cliente)",
                   all(e <= s + 1e-9 for e, s in
                       zip(r.tiempos_espera_cola, r.tiempos_en_sistema)),
                   "W_q ≤ W ✓")

    # Throughput ≈ λ
    reps = replicar_paralelo(gen_exp(3.0), gen_exp(4.0), T_SIM, R, seed=42)
    thr  = media([r.throughput for r in reps])
    check("Throughput ≈ λ=3.0", thr, 3.0, tol_pct=5.0)

    # Cuando ρ→0: casi nadie espera en cola
    reps_vacio = replicar_paralelo(gen_exp(0.1), gen_exp(5.0), T_SIM, 200, seed=42)
    p_espera = media([
        sum(1 for w in r.tiempos_espera_cola if w > 0.001) / max(r.ND, 1)
        for r in reps_vacio
    ])
    check_bool("ρ≈0.01: fracción con espera > 0 es muy pequeña",
               p_espera < 0.05, f"p_espera={p_espera:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  Resumen
# ══════════════════════════════════════════════════════════════════════════════

def resumen():
    total = len(_resultados); ok = sum(_resultados); fail = total - ok
    print(f"\n{'═'*80}")
    print(f"  RESUMEN  |  Total={total}  Pasados=\033[92m{ok}\033[0m  Fallados=\033[91m{fail}\033[0m  ({100*ok/total:.1f}%)")
    print(f"{'═'*80}")
    sys.exit(0 if fail == 0 else 1)

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("  TEST — MODELO DE DOS SERVIDORES EN PARALELO (M/M/2)")
    print("█"*80)
    test_estructura()
    test_utilizacion()
    test_valores_mm2()
    test_comparacion_mm1_mm2()
    test_reproducibilidad()
    test_consistencia()
    resumen()
