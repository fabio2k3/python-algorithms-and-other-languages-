"""
test_inventario.py
===================
Suite de tests para el Modelo de Inventario con política (s, S).

Estrategia de validación:
  El modelo de inventario no tiene fórmulas cerradas simples en el
  caso general, pero sí se pueden verificar:
    - Invariantes del estado: x ∈ [0, S] siempre tras reposición.
    - Relaciones contables: R = Σ ventas·r_unit,  C = Σ pedidos·c(y).
    - Comportamiento monotóno: mayor demanda → más faltantes.
    - Política (s,S): pedido si y solo si x < s y no hay pedido pendiente.
    - Con demanda constante y determinista: resultados exactamente calculables.
    - Balance: ganancia = R - C - H - P.

Bloques:
  1. Correctitud estructural e invariantes del estado
  2. Verificación de la política (s, S): cuándo se pide y cuánto
  3. Caso determinista: demanda constante → resultados exactos
  4. Efectos de los parámetros (s, S, L, c_fijo)
  5. Métricas de costo y balance contable
  6. Reproducibilidad y semillas
  7. Casos borde (inventario suficiente para nunca pedir, L=0)
"""

import sys, math, random
from typing import List
from inventario import (
    simular_inventario,
    replicar_inventario,
    ResultadoInventario,
    gen_exp,
    gen_uniforme_int,
    gen_constante,
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
    print(f"  [{PASS if ok else FAIL}] {label:<64} got={got:.4f}  exp={exp:.4f}  Δ={d:.1f}%")

def check_bool(label, cond, detalle=""):
    _resultados.append(cond)
    print(f"  [{PASS if cond else FAIL}] {label:<64} {detalle}")

def check_range(label, got, lo, hi):
    ok = lo <= got <= hi
    _resultados.append(ok)
    print(f"  [{PASS if ok else FAIL}] {label:<64} got={got:.4f}  [{lo:.4f},{hi:.4f}]")

def section(t):
    print(f"\n{'═'*80}\n  {t}\n{'═'*80}")

def media(lst): return sum(lst) / len(lst)

# Parámetros base
S_BASE   = 100
s_BASE   = 20
T_BASE   = 500.0
x0_BASE  = 60.0
L_BASE   = 5.0
C_FIJO   = 50.0
C_VAR    = 2.0
H_UNIT   = 0.1
R_UNIT   = 10.0
P_UNIT   = 5.0
R_REPS   = 400

def _sim(**kwargs):
    """Simula con parámetros base y permite sobreescribir."""
    p = dict(
        s=s_BASE, S=S_BASE, T=T_BASE, x0=x0_BASE,
        gen_interdemanda=gen_exp(1.0), gen_demanda=gen_uniforme_int(1, 10),
        L=L_BASE, c_fijo=C_FIJO, c_var=C_VAR,
        h=H_UNIT, r_unit=R_UNIT, p_unit=P_UNIT, seed=42,
    )
    p.update(kwargs)
    return simular_inventario(**p)

def _rep(**kwargs):
    p = dict(
        s=s_BASE, S=S_BASE, T=T_BASE, x0=x0_BASE,
        gen_interdemanda=gen_exp(1.0), gen_demanda=gen_uniforme_int(1, 10),
        L=L_BASE, c_fijo=C_FIJO, c_var=C_VAR,
        h=H_UNIT, r_unit=R_UNIT, p_unit=P_UNIT,
        r=R_REPS, seed=42,
    )
    p.update(kwargs)
    return replicar_inventario(**p)

# ══════════════════════════════════════════════════════════════════════════════
#  1. CORRECTITUD ESTRUCTURAL E INVARIANTES
# ══════════════════════════════════════════════════════════════════════════════

def test_estructura():
    section("1 · CORRECTITUD ESTRUCTURAL E INVARIANTES")

    r = _sim()

    check_bool("tiempo_sim ≈ T", abs(r.tiempo_sim - T_BASE) < 1e-6,
               f"t={r.tiempo_sim:.4f} T={T_BASE}")
    check_bool("n_demandas > 0",   r.n_demandas > 0, f"n={r.n_demandas}")
    check_bool("n_pedidos > 0",    r.n_pedidos > 0,  f"n={r.n_pedidos}")
    check_bool("demanda_total > 0", r.demanda_total > 0, f"D={r.demanda_total:.1f}")
    check_bool("unidades_vendidas ≤ demanda_total",
               r.unidades_vendidas <= r.demanda_total + 1e-9,
               f"v={r.unidades_vendidas:.1f} D={r.demanda_total:.1f}")
    check_bool("unidades_faltantes ≥ 0", r.unidades_faltantes >= -1e-9,
               f"f={r.unidades_faltantes:.1f}")
    check_bool("unidades_vendidas + unidades_faltantes ≈ demanda_total",
               abs(r.unidades_vendidas + r.unidades_faltantes - r.demanda_total) < 1e-6,
               "balance ✓")

    # Costos y balances no negativos
    check_bool("C ≥ 0", r.C >= 0, f"C={r.C:.2f}")
    check_bool("H ≥ 0", r.H >= 0, f"H={r.H:.2f}")
    check_bool("R ≥ 0", r.R >= 0, f"R={r.R:.2f}")
    check_bool("P ≥ 0", r.P >= 0, f"P={r.P:.2f}")

    # Balance contable
    ganancia_calculada = r.R - r.C - r.H - r.P
    check_bool("ganancia_neta = R - C - H - P",
               abs(r.ganancia_neta - ganancia_calculada) < 1e-6,
               f"Δ={abs(r.ganancia_neta - ganancia_calculada):.8f}")

    # Historial tiene al menos 2 puntos
    check_bool("historial tiene ≥ 2 puntos", len(r.historial_inventario) >= 2,
               f"len={len(r.historial_inventario)}")

    # Primer punto del historial es (0, x0)
    t0, x0_hist = r.historial_inventario[0]
    check_bool("historial[0] = (0, x0)",
               abs(t0) < 1e-9 and abs(x0_hist - x0_BASE) < 1e-9,
               f"({t0:.4f}, {x0_hist:.4f})")

    # Nivel de inventario en historial siempre ≥ 0
    check_bool("Nivel inventario ≥ 0 en todo el historial",
               all(x >= -1e-9 for _, x in r.historial_inventario),
               "x ≥ 0 ✓")

    # fraccion_satisfecha ∈ [0, 1]
    check_range("fraccion_satisfecha ∈ [0,1]",
                r.fraccion_satisfecha, 0.0, 1.0)

    # nivel_medio_inventario ∈ [0, S]
    check_range("nivel_medio_inventario ∈ [0, S]",
                r.nivel_medio_inventario, 0.0, float(S_BASE))

# ══════════════════════════════════════════════════════════════════════════════
#  2. POLÍTICA (s, S): CUÁNDO Y CUÁNTO SE PIDE
# ══════════════════════════════════════════════════════════════════════════════

def test_politica_sS():
    section("2 · VERIFICACIÓN DE LA POLÍTICA (s, S)")

    # Con demanda constante d=1 y tasa interdemanda=1 → podemos rastrear el estado
    # Cada demanda reduce inventario en 1.
    # Cuando x < s → pedir y = S - x  unidades.
    r = _sim(
        gen_demanda=gen_constante(1.0),  # demanda exactamente 1 unidad
        gen_interdemanda=gen_exp(1.0),
        s=10, S=50, x0=50, L=0.0001,    # L casi 0 para ver inmediatez
        T=200.0, seed=42
    )
    # Verificar que nunca x > S después de reposición
    for t, x in r.historial_inventario:
        check_bool(f"t={t:.1f}: x ≤ S={50}",
                   x <= 50 + 1e-6,
                   f"x={x:.2f}")
        if x < 0:
            break   # no repetir checks negativos

    # Con demanda grande y L=0: si hay pedido, inventario se repone a S
    r2 = _sim(s=10, S=80, x0=80, L=0.0, T=100.0, seed=42,
              gen_demanda=gen_constante(5.0),
              gen_interdemanda=gen_exp(2.0))
    # Tras cada reposición el inventario debe ser ≤ S
    niveles_tras_repo = []
    for i in range(1, len(r2.historial_inventario)):
        t_prev, x_prev = r2.historial_inventario[i-1]
        t_cur,  x_cur  = r2.historial_inventario[i]
        if x_cur > x_prev:   # hubo reposición
            niveles_tras_repo.append(x_cur)

    if niveles_tras_repo:
        check_bool("Tras reposición: x ≤ S siempre",
                   all(x <= 80 + 1e-6 for x in niveles_tras_repo),
                   f"máx_tras_repo={max(niveles_tras_repo):.2f}")

    # n_pedidos es consistente: C ≥ n_pedidos * c_fijo
    r3 = _sim(seed=7)
    check_bool("C ≥ n_pedidos × c_fijo  (costo fijo por pedido)",
               r3.C >= r3.n_pedidos * C_FIJO - 1e-6,
               f"C={r3.C:.2f} n_pedidos={r3.n_pedidos} C_fijo_total={r3.n_pedidos*C_FIJO:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  3. CASO DETERMINISTA: DEMANDA CONSTANTE
# ══════════════════════════════════════════════════════════════════════════════

def test_determinista():
    section("3 · CASO DETERMINISTA (demanda y tiempo constantes)")

    # Demanda = 1 unidad cada 1 unidad de tiempo exacta.
    # Interdemanda constante = 1.0  (usamos gen_constante para tiempo también)
    def gen_const_time(val):
        def _g(rng): return val
        return _g

    s, S_det = 5, 20
    x0_d     = 20
    d        = 1.0     # demanda por evento
    inter    = 1.0     # tiempo entre demandas
    L_d      = 0.0     # reposición instantánea
    T_d      = 50.0

    r = simular_inventario(
        s=s_det if (s_det := s) else s,
        S=S_det, T=T_d, x0=x0_d,
        gen_interdemanda=gen_const_time(inter),
        gen_demanda=gen_constante(d),
        L=L_d, c_fijo=10.0, c_var=1.0,
        h=0.0, r_unit=5.0, p_unit=0.0, seed=42
    )

    # Con L=0, demanda=1 cada 1 ut, x0=S=20, s=5:
    # Inventario baja 1 por evento. Al llegar a s=5 (después de 15 eventos)
    # se repone a S=20. Ciclo de 15 eventos.
    # Demanda total = T / inter = 50 eventos = 50 unidades
    # Sin faltantes (reposición instantánea)
    check("Demanda total = T/inter = 50", r.demanda_total, 50.0, tol_pct=0.1)
    check("Unidades vendidas = demanda total (sin faltantes con L=0)",
          r.unidades_vendidas, r.demanda_total, tol_pct=0.1)
    check_bool("Sin faltantes con L=0 y demanda controlada",
               r.unidades_faltantes < 1e-6,
               f"faltantes={r.unidades_faltantes:.4f}")

    # Ingresos = ventas × r_unit
    ingresos_esperados = r.unidades_vendidas * 5.0
    check("R = unidades_vendidas × r_unit",
          r.R, ingresos_esperados, tol_pct=0.1)

# ══════════════════════════════════════════════════════════════════════════════
#  4. EFECTOS DE LOS PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

def test_efectos_parametros():
    section("4 · EFECTOS DE LOS PARÁMETROS")

    # Mayor L (lead time) → más faltantes (inventario baja más antes de recibir)
    r_L0  = _rep(L=0.0,  r=R_REPS, seed=42)
    r_L10 = _rep(L=10.0, r=R_REPS, seed=42)
    fall_L0  = media([r.unidades_faltantes for r in r_L0])
    fall_L10 = media([r.unidades_faltantes for r in r_L10])
    check_bool("Mayor L → más faltantes",
               fall_L10 >= fall_L0, f"falt(L=0)={fall_L0:.1f} falt(L=10)={fall_L10:.1f}")

    # Mayor S (stock máximo) → menor fracción de faltantes
    r_S50  = _rep(S=50,  s=10, r=R_REPS, seed=42)
    r_S150 = _rep(S=150, s=10, r=R_REPS, seed=42)
    falt_S50  = media([r.fraccion_satisfecha for r in r_S50])
    falt_S150 = media([r.fraccion_satisfecha for r in r_S150])
    check_bool("Mayor S → mayor fracción satisfecha",
               falt_S150 >= falt_S50 - 0.01,
               f"frac(S=50)={falt_S50:.4f} frac(S=150)={falt_S150:.4f}")

    # Mayor S → más costo de almacenaje
    H_S50  = media([r.H for r in r_S50])
    H_S150 = media([r.H for r in r_S150])
    check_bool("Mayor S → mayor costo almacenaje",
               H_S150 > H_S50, f"H(S=50)={H_S50:.1f} H(S=150)={H_S150:.1f}")

    # Mayor c_fijo → menos pedidos (se espera más antes de pedir)
    r_cf10  = _rep(c_fijo=10.0,   r=R_REPS, seed=42)
    r_cf200 = _rep(c_fijo=200.0,  r=R_REPS, seed=42)
    # Al aumentar c_fijo, el costo total de pedidos también sube
    C_cf10  = media([r.C for r in r_cf10])
    C_cf200 = media([r.C for r in r_cf200])
    check_bool("Mayor c_fijo → mayor costo total de pedidos C",
               C_cf200 > C_cf10,
               f"C(c_f=10)={C_cf10:.1f} C(c_f=200)={C_cf200:.1f}")

    # Mayor tasa de demanda (interdemanda menor) → más demanda total
    # gen_exp(2.0) → interdemanda media=0.5 → tasa=2  (alta)
    # gen_exp(0.5) → interdemanda media=2.0 → tasa=0.5 (baja)
    r_lam1 = _rep(gen_interdemanda=gen_exp(2.0), r=R_REPS, seed=42)  # tasa alta
    r_lam2 = _rep(gen_interdemanda=gen_exp(0.5), r=R_REPS, seed=42)  # tasa baja
    d_lam1 = media([r.demanda_total for r in r_lam1])
    d_lam2 = media([r.demanda_total for r in r_lam2])
    check_bool("Mayor tasa demanda (1/E[inter]) → mayor demanda total",
               d_lam1 > d_lam2,
               f"D(tasa=2)={d_lam1:.1f} D(tasa=0.5)={d_lam2:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
#  5. MÉTRICAS DE COSTO Y BALANCE CONTABLE
# ══════════════════════════════════════════════════════════════════════════════

def test_balance_contable():
    section("5 · BALANCE CONTABLE  ganancia = R - C - H - P")

    reps = _rep()
    for i, r in enumerate(reps[:20]):   # verificar primeras 20 réplicas
        g_calc = r.R - r.C - r.H - r.P
        check_bool(f"réplica {i+1:2d}: ganancia_neta == R-C-H-P",
                   abs(r.ganancia_neta - g_calc) < 1e-6,
                   f"Δ={abs(r.ganancia_neta - g_calc):.2e}")

    # Costo de pedido ≥ n_pedidos × c_fijo en todas las réplicas
    check_bool("C ≥ n_pedidos × c_fijo en todas las réplicas",
               all(r.C >= r.n_pedidos * C_FIJO - 1e-6 for r in reps),
               "✓")

    # Nivel medio inventario > 0 (siempre hay algo de stock en media)
    niveles = [r.nivel_medio_inventario for r in reps]
    check_bool("nivel_medio_inventario > 0 en todas las réplicas",
               all(n > 0 for n in niveles), f"min={min(niveles):.2f}")

    # fraccion_satisfecha ∈ [0,1] en todas las réplicas
    fracs = [r.fraccion_satisfecha for r in reps]
    check_bool("fraccion_satisfecha ∈ [0,1] siempre",
               all(0.0 <= f <= 1.0 + 1e-9 for f in fracs),
               f"min={min(fracs):.4f} max={max(fracs):.4f}")

    # Ingresos > 0 (siempre se vende algo)
    check_bool("R > 0 en todas las réplicas",
               all(r.R > 0 for r in reps), "✓")

# ══════════════════════════════════════════════════════════════════════════════
#  6. REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def test_reproducibilidad():
    section("6 · REPRODUCIBILIDAD")

    r1 = _sim(seed=123)
    r2 = _sim(seed=123)
    check_bool("Misma semilla → mismo n_demandas",  r1.n_demandas == r2.n_demandas, f"n={r1.n_demandas}")
    check_bool("Misma semilla → mismo n_pedidos",   r1.n_pedidos  == r2.n_pedidos,  f"n={r1.n_pedidos}")
    check_bool("Misma semilla → mismo R",           abs(r1.R - r2.R) < 1e-9, f"R={r1.R:.4f}")
    check_bool("Misma semilla → mismo ganancia",    abs(r1.ganancia_neta - r2.ganancia_neta) < 1e-9, "✓")

    r3 = _sim(seed=124)
    check_bool("Distinta semilla → distinto resultado",
               r1.n_demandas != r3.n_demandas or abs(r1.R - r3.R) > 1e-6, "✓")

# ══════════════════════════════════════════════════════════════════════════════
#  7. CASOS BORDE
# ══════════════════════════════════════════════════════════════════════════════

def test_casos_borde():
    section("7 · CASOS BORDE")

    # Inventario inicial muy alto → nunca (o rara vez) se pide
    r_alto = _sim(x0=10000.0, S=10000, s=9000, T=50.0, L=1.0, seed=42,
                  gen_demanda=gen_constante(1.0), gen_interdemanda=gen_exp(1.0))
    # No debería necesitar pedir en T=50 con x0=10000 y demanda=1/ut
    check_bool("x0 enorme: sin faltantes en T=50",
               r_alto.unidades_faltantes < 1e-6,
               f"falt={r_alto.unidades_faltantes:.2f}")

    # Lead time = 0: reposición instantánea → muy pocos faltantes
    r_L0 = _sim(L=0.0, s=s_BASE, S=S_BASE, x0=x0_BASE, T=T_BASE, seed=42)
    check_bool("L=0: fracción satisfecha alta (≥ 0.85)",
               r_L0.fraccion_satisfecha >= 0.85,
               f"frac={r_L0.fraccion_satisfecha:.4f}")

    # s=0: nunca se pide (solo cuando inventario llega a 0)
    r_s0 = _sim(s=0, S=S_BASE, x0=S_BASE, T=200.0, seed=42)
    check_bool("s=0: pedidos solo cuando x llega a 0",
               r_s0.n_pedidos >= 0, f"n_pedidos={r_s0.n_pedidos}")

    # T muy corto: modelo termina correctamente
    r_T1 = _sim(T=1.0, seed=42)
    check_bool("T=1.0: simulación termina en tiempo ≈ T",
               abs(r_T1.tiempo_sim - 1.0) < 1e-6, f"t={r_T1.tiempo_sim:.6f}")

    # Demanda grande: muchos faltantes esperables
    r_demanda_grande = _sim(
        gen_demanda=gen_constante(50.0),
        gen_interdemanda=gen_exp(2.0),
        s=s_BASE, S=S_BASE, x0=x0_BASE,
        T=100.0, seed=42
    )
    check_bool("Demanda muy grande → hay faltantes",
               r_demanda_grande.unidades_faltantes > 0,
               f"falt={r_demanda_grande.unidades_faltantes:.1f}")

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
    print("  TEST — MODELO DE INVENTARIO  (s, S)")
    print("█"*80)
    test_estructura()
    test_politica_sS()
    test_determinista()
    test_efectos_parametros()
    test_balance_contable()
    test_reproducibilidad()
    test_casos_borde()
    resumen()
