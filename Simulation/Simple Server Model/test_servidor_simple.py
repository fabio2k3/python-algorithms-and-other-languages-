"""
test_servidor_simple.py
========================
Suite de tests para el Modelo de un Servidor Simple.

Estrategia de validación:
  Para M/M/1 existen fórmulas exactas de Teoría de Colas:
      ρ    = λ / μ
      W    = 1 / (μ - λ)
      W_q  = ρ / (μ - λ)  =  λ / (μ·(μ - λ))
      L    = ρ / (1 - ρ)
      L_q  = ρ² / (1 - ρ)

  Se ejecutan r replicaciones largas y se compara la media empírica
  con el valor teórico usando tolerancia relativa del 5-8%.

Bloques:
  1. Correctitud estructural (tipos, rangos, invariantes)
  2. Valores teóricos M/M/1 en varios puntos de operación
  3. Ley de Little:  L = λ·W  (relación universal)
  4. Casos límite (carga alta, carga baja, servidor rápido)
  5. Reproducibilidad y semillas
  6. Consistencia entre NA, ND y tiempos registrados
"""

import sys, math, random
from typing import List
from servidor_simple import (
    simular_servidor_simple,
    replicar_servidor_simple,
    ResultadoServidorSimple,
    gen_exp,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
_resultados: List[bool] = []

def check(label, got, exp, tol_pct=5.0):
    if abs(exp) < 1e-9:
        ok = abs(got - exp) < tol_pct / 100
    else:
        ok = abs(got - exp) / abs(exp) <= tol_pct / 100
    _resultados.append(ok)
    d = abs(got - exp) / max(abs(exp), 1e-9) * 100
    print(f"  [{PASS if ok else FAIL}] {label:<60} got={got:.5f}  exp={exp:.5f}  Δ={d:.1f}%")

def check_bool(label, cond, detalle=""):
    _resultados.append(cond)
    print(f"  [{PASS if cond else FAIL}] {label:<60} {detalle}")

def check_range(label, got, lo, hi):
    ok = lo <= got <= hi
    _resultados.append(ok)
    print(f"  [{PASS if ok else FAIL}] {label:<60} got={got:.5f}  rango=[{lo:.4f},{hi:.4f}]")

def section(t):
    print(f"\n{'═'*76}\n  {t}\n{'═'*76}")

def media(lst): return sum(lst) / len(lst)

# ── Parámetros para replicaciones ─────────────────────────────────────────────
T_SIM = 500.0     # horizonte por réplica (largo para estabilidad estadística)
R     = 800       # réplicas

def _metricas(lam, mu):
    """Simula R réplicas y devuelve medias empíricas."""
    reps = replicar_servidor_simple(gen_exp(lam), gen_exp(mu), T_SIM, R, seed=42)
    return (
        media([r.W   for r in reps]),
        media([r.W_q for r in reps]),
        media([r.rho for r in reps]),
        media([r.L   for r in reps]),
        reps,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  1. CORRECTITUD ESTRUCTURAL
# ══════════════════════════════════════════════════════════════════════════════

def test_estructura():
    section("1 · CORRECTITUD ESTRUCTURAL")
    r = simular_servidor_simple(gen_exp(2.0), gen_exp(4.0), 200.0, seed=7)

    check_bool("NA > 0  (al menos un arribo)",
               r.NA > 0, f"NA={r.NA}")
    check_bool("ND ≤ NA  (no pueden salir más de los que entraron)",
               r.ND <= r.NA, f"ND={r.ND} NA={r.NA}")
    check_bool("ND > 0  (al menos una partida)",
               r.ND > 0, f"ND={r.ND}")
    check_bool("Tiempo sim > 0",
               r.tiempo_sim > 0, f"t={r.tiempo_sim:.2f}")
    check_bool("Claves A van de 1..NA",
               set(r.A.keys()) == set(range(1, r.NA + 1)), f"claves={sorted(r.A.keys())[:5]}…")
    check_bool("Claves D van de 1..ND",
               set(r.D.keys()) == set(range(1, r.ND + 1)), f"claves={sorted(r.D.keys())[:5]}…")
    check_bool("Todos los tiempos A > 0",
               all(v > 0 for v in r.A.values()), "✓")
    check_bool("Todos los tiempos A ≤ T",
               all(v <= 200.0 + 1e-9 for v in r.A.values()), "✓")
    check_bool("Todos los tiempos D > 0",
               all(v > 0 for v in r.D.values()), "✓")
    check_bool("Tiempos de espera en cola ≥ 0",
               all(w >= 0 for w in r.tiempos_espera_cola), "✓")
    check_bool("Tiempos en sistema > 0",
               all(w > 0 for w in r.tiempos_en_sistema), "✓")
    check_range("ρ ∈ (0, 1)",  r.rho, 0.0, 1.0)
    check_bool("W_q ≤ W  (espera en cola ≤ tiempo total en sistema)",
               r.W_q <= r.W + 1e-9, f"W_q={r.W_q:.4f} W={r.W:.4f}")
    check_bool("W > 0",  r.W > 0,  f"W={r.W:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  2. VALORES TEÓRICOS M/M/1
# ══════════════════════════════════════════════════════════════════════════════

def test_valores_mm1():
    section("2 · VALORES TEÓRICOS M/M/1")

    casos = [
        # (lam, mu,  tol%)
        (1.0, 2.0,  6.0),   # ρ=0.5 — carga moderada
        (2.0, 4.0,  6.0),   # ρ=0.5 — misma ρ, distintos parámetros
        (3.0, 4.0,  8.0),   # ρ=0.75
        (1.0, 4.0,  6.0),   # ρ=0.25 — carga baja
        (2.0, 3.0,  8.0),   # ρ=0.667
    ]

    for lam, mu, tol in casos:
        rho_t = lam / mu
        W_t   = 1.0 / (mu - lam)
        Wq_t  = rho_t / (mu - lam)

        W_e, Wq_e, rho_e, _, _ = _metricas(lam, mu)

        check(f"λ={lam} μ={mu} → ρ",  rho_e, rho_t, tol)
        check(f"λ={lam} μ={mu} → W",   W_e,   W_t,   tol)
        check(f"λ={lam} μ={mu} → W_q", Wq_e,  Wq_t,  tol)

# ══════════════════════════════════════════════════════════════════════════════
#  3. LEY DE LITTLE  L = λ · W
# ══════════════════════════════════════════════════════════════════════════════

def test_ley_little():
    section("3 · LEY DE LITTLE  (L = λ · W)")

    for lam, mu in [(2.0, 4.0), (3.0, 4.0), (1.0, 3.0)]:
        W_e, _, _, L_e, _ = _metricas(lam, mu)
        L_little = lam * W_e    # L calculado con Little
        # Comparar L directo (integración) vs λ·W
        check(f"λ={lam} μ={mu}: L_directo ≈ λ·W", L_e, L_little, tol_pct=8.0)

# ══════════════════════════════════════════════════════════════════════════════
#  4. CASOS LÍMITE
# ══════════════════════════════════════════════════════════════════════════════

def test_casos_limite():
    section("4 · CASOS LÍMITE")

    # Carga muy baja (ρ→0): casi nunca hay espera en cola
    reps_bajo = replicar_servidor_simple(gen_exp(0.1), gen_exp(5.0), T_SIM, R, seed=42)
    Wq_bajo   = media([r.W_q for r in reps_bajo])
    W_bajo    = media([r.W   for r in reps_bajo])
    Wq_teo    = (0.1/5.0) / (5.0 - 0.1)
    check("ρ=0.02: W_q muy pequeño", Wq_bajo, Wq_teo, tol_pct=20.0)
    check_bool("ρ=0.02: W_q ≪ W  (servicio domina)",
               Wq_bajo < W_bajo * 0.1, f"W_q={Wq_bajo:.4f} W={W_bajo:.4f}")

    # Servidor muy rápido (μ≫λ): W ≈ 1/μ (servicio casi sin espera)
    reps_rapido = replicar_servidor_simple(gen_exp(1.0), gen_exp(20.0), T_SIM, R, seed=42)
    W_rapido  = media([r.W for r in reps_rapido])
    W_teo_rap = 1.0 / (20.0 - 1.0)
    check("μ≫λ: W ≈ 1/(μ-λ)", W_rapido, W_teo_rap, tol_pct=8.0)

    # Todos atendidos antes de T → ND ≤ NA siempre
    reps_fin = replicar_servidor_simple(gen_exp(1.0), gen_exp(3.0), 50.0, 200, seed=42)
    check_bool("ND ≤ NA en todas las réplicas",
               all(r.ND <= r.NA for r in reps_fin), "✓")

    # Mayor carga → mayor espera en cola
    _, Wq_25, _, _, _ = _metricas(1.0, 4.0)   # ρ=0.25
    _, Wq_75, _, _, _ = _metricas(3.0, 4.0)   # ρ=0.75
    check_bool("Mayor ρ → mayor W_q",
               Wq_75 > Wq_25, f"Wq(ρ=0.75)={Wq_75:.4f} > Wq(ρ=0.25)={Wq_25:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  5. REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def test_reproducibilidad():
    section("5 · REPRODUCIBILIDAD")

    r1 = simular_servidor_simple(gen_exp(2.0), gen_exp(4.0), 100.0, seed=99)
    r2 = simular_servidor_simple(gen_exp(2.0), gen_exp(4.0), 100.0, seed=99)
    check_bool("Misma semilla → mismo NA",  r1.NA == r2.NA,  f"NA={r1.NA}")
    check_bool("Misma semilla → mismo ND",  r1.ND == r2.ND,  f"ND={r1.ND}")
    check_bool("Misma semilla → mismo W",   abs(r1.W - r2.W) < 1e-12, f"W={r1.W:.6f}")
    check_bool("Misma semilla → mismo W_q", abs(r1.W_q - r2.W_q) < 1e-12, "✓")

    r3 = simular_servidor_simple(gen_exp(2.0), gen_exp(4.0), 100.0, seed=100)
    check_bool("Distinta semilla → distinto NA (casi seguro)",
               r1.NA != r3.NA or r1.ND != r3.ND, "✓")

# ══════════════════════════════════════════════════════════════════════════════
#  6. CONSISTENCIA INTERNA
# ══════════════════════════════════════════════════════════════════════════════

def test_consistencia():
    section("6 · CONSISTENCIA INTERNA")

    for seed in [1, 42, 123, 999]:
        r = simular_servidor_simple(gen_exp(2.0), gen_exp(4.0), 200.0, seed=seed)

        check_bool(f"seed={seed}: |tiempos_espera| ≤ ND",
                   len(r.tiempos_espera_cola) <= r.ND + 1,
                   f"len={len(r.tiempos_espera_cola)} ND={r.ND}")
        check_bool(f"seed={seed}: |tiempos_sistema| ≤ ND",
                   len(r.tiempos_en_sistema) <= r.ND + 1,
                   f"len={len(r.tiempos_en_sistema)} ND={r.ND}")
        check_bool(f"seed={seed}: utilización ∈ (0,1)",
                   0 < r.rho < 1, f"ρ={r.rho:.4f}")
        check_bool(f"seed={seed}: W ≥ W_q",
                   r.W >= r.W_q - 1e-9, f"W={r.W:.4f} W_q={r.W_q:.4f}")

    # Throughput ≈ λ para sistema estable
    reps = replicar_servidor_simple(gen_exp(2.0), gen_exp(5.0), T_SIM, R, seed=42)
    thr  = media([r.throughput for r in reps])
    check("Throughput ≈ λ=2.0 (sistema estable)", thr, 2.0, tol_pct=5.0)

# ══════════════════════════════════════════════════════════════════════════════
#  Resumen
# ══════════════════════════════════════════════════════════════════════════════

def resumen():
    total  = len(_resultados)
    ok     = sum(_resultados)
    fail   = total - ok
    print(f"\n{'═'*76}")
    print(f"  RESUMEN  |  Total={total}  Pasados=\033[92m{ok}\033[0m  Fallados=\033[91m{fail}\033[0m  ({100*ok/total:.1f}%)")
    print(f"{'═'*76}")
    sys.exit(0 if fail == 0 else 1)

if __name__ == "__main__":
    print("\n" + "█"*76)
    print("  TEST — MODELO DE UN SERVIDOR SIMPLE")
    print("█"*76)
    test_estructura()
    test_valores_mm1()
    test_ley_little()
    test_casos_limite()
    test_reproducibilidad()
    test_consistencia()
    resumen()
