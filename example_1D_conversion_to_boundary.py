"""
Cuadratura 1D mediante Dual Reciprocity
----------------------------------------
Problema : I = integral_a^b g(x) dx
Estrategia: aproximar g(x) = sum_i alpha_i f_i(x)
            con f_i(x) = 1 + |x - x_i|^3  (RBF 1D)
            y  phi_i(x) tal que d(phi_i)/dx = f_i(x)

Obtencion de phi_i:
    d(phi_i)/dx = 1 + |x - x_i|^3
    integrando:
    phi_i(x) = x + (x - x_i)*|x - x_i|^3 / 4

Verificacion:
    d/dx [ x + (x-xi)*|x-xi|^3 / 4 ]
    = 1 + [ |x-xi|^3 + (x-xi)*3|x-xi|^2*sgn(x-xi) ] / 4
    = 1 + [ |x-xi|^3 + 3|x-xi|^3 ] / 4
    = 1 + |x-xi|^3   checkmark

Resultado final:
    I = sum_i alpha_i [ phi_i(b) - phi_i(a) ]

Solucion analitica de referencia:
    g(x) = x^2,  [a,b] = [0,1]  =>  I = 1/3
"""

import numpy as np

# ─────────────────────────────────────────────
# 1. FUNCIONES BASE
# ─────────────────────────────────────────────

def f_rbf(x, xi):
    """RBF 1D: f(x; xi) = 1 + |x - xi|^3"""
    r = x - xi
    return 1.0 + np.abs(r)**3

def phi(x, xi):
    """Solucion particular 1D: d(phi)/dx = f
       phi(x; xi) = x + (x - xi)*|x - xi|^3 / 4"""
    r = x - xi
    return x + r * np.abs(r)**3 / 4.0

def g_func(x):
    """Funcion a integrar"""
    return x**2

# ─────────────────────────────────────────────
# 2. DOMINIO Y SOLUCION ANALITICA
# ─────────────────────────────────────────────

a = 0.0
b = 1.0
I_analitica = 1.0 / 3.0   # integral de x^2 en [0,1]

# ─────────────────────────────────────────────
# 3. NODOS DE COLOCACION INTERIORES
#    (no en los extremos a, b para evitar
#     singularidad cuando x_i = a o x_i = b)
# ─────────────────────────────────────────────

N = 5
col_nodes = np.linspace(a, b, N+2)[1:-1]   # N puntos interiores
print(f"Nodos de colocacion: {col_nodes}")

# ─────────────────────────────────────────────
# 4. SISTEMA DE COLOCACION: F * alpha = g
# ─────────────────────────────────────────────

F_mat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        F_mat[i, j] = f_rbf(col_nodes[i], col_nodes[j])

g_vec = np.array([g_func(x) for x in col_nodes])

alpha = np.linalg.solve(F_mat, g_vec)

print(f"\nCoeficientes alpha: {alpha}")
print(f"Cond(F) = {np.linalg.cond(F_mat):.4f}")

# ─────────────────────────────────────────────
# 5. EVALUACION EN LA "FRONTERA" {a, b}
#    I = sum_i alpha_i [ phi_i(b) - phi_i(a) ]
# ─────────────────────────────────────────────

I_dr = 0.0
for j in range(N):
    xi  = col_nodes[j]
    I_dr += alpha[j] * (phi(b, xi) - phi(a, xi))

error = abs(I_dr - I_analitica) / I_analitica * 100.0

print(f"\n--- RESULTADO ---")
print(f"Integral analitica    : {I_analitica:.10f}")
print(f"Integral DR (1D)      : {I_dr:.10f}")
print(f"Error relativo        : {error:.6f} %")

# ─────────────────────────────────────────────
# 6. VERIFICACION: reconstruccion de g(x)
#    con los alpha obtenidos
# ─────────────────────────────────────────────

x_test = np.linspace(a, b, 200)
g_rbf  = np.zeros_like(x_test)
for j in range(N):
    g_rbf += alpha[j] * f_rbf(x_test, col_nodes[j])

g_exact = g_func(x_test)
error_g = np.max(np.abs(g_rbf - g_exact))
print(f"Error max reconstruccion g(x): {error_g:.2e}")

# ─────────────────────────────────────────────
# 7. ESTUDIO DE CONVERGENCIA vs N
# ─────────────────────────────────────────────

print(f"\n--- CONVERGENCIA vs N (nodos de colocacion) ---")
print(f"{'N':>6}  {'I_DR':>14}  {'Error (%)':>12}  {'Cond(F)':>12}")
print("-" * 52)

for N_test in [2, 3, 5, 8, 12, 20]:

    nodes_t = np.linspace(a, b, N_test+2)[1:-1]

    F_t = np.zeros((N_test, N_test))
    for i in range(N_test):
        for j in range(N_test):
            F_t[i, j] = f_rbf(nodes_t[i], nodes_t[j])

    g_t   = np.array([g_func(x) for x in nodes_t])
    alp_t = np.linalg.solve(F_t, g_t)

    I_t = sum(alp_t[j] * (phi(b, nodes_t[j]) - phi(a, nodes_t[j]))
              for j in range(N_test))

    err_t  = abs(I_t - I_analitica) / I_analitica * 100.0
    cond_t = np.linalg.cond(F_t)

    print(f"{N_test:>6}  {I_t:>14.10f}  {err_t:>12.6f}  {cond_t:>12.2f}")

# ─────────────────────────────────────────────
# 8. PRUEBA CON OTRAS FUNCIONES g(x)
# ─────────────────────────────────────────────

print(f"\n--- PRUEBA CON DIFERENTES g(x), N=8 ---")
print(f"{'g(x)':>15}  {'I_exact':>12}  {'I_DR':>12}  {'Error (%)':>10}")
print("-" * 55)

N_test = 8
nodes_t = np.linspace(a, b, N_test+2)[1:-1]
F_t = np.zeros((N_test, N_test))
for i in range(N_test):
    for j in range(N_test):
        F_t[i, j] = f_rbf(nodes_t[i], nodes_t[j])

test_cases = [
    ("x^2",    lambda x: x**2,           1.0/3.0),
    ("x^3",    lambda x: x**3,           1.0/4.0),
    ("x^4",    lambda x: x**4,           1.0/5.0),
    ("sin(px)", lambda x: np.sin(np.pi*x), 2.0/np.pi),
    ("exp(x)", lambda x: np.exp(x),       np.e - 1.0),
]

for name, gfunc, I_ex in test_cases:
    g_t   = np.array([gfunc(x) for x in nodes_t])
    alp_t = np.linalg.solve(F_t, g_t)
    I_t   = sum(alp_t[j] * (phi(b, nodes_t[j]) - phi(a, nodes_t[j]))
                for j in range(N_test))
    err_t = abs(I_t - I_ex) / abs(I_ex) * 100.0
    print(f"{name:>15}  {I_ex:>12.8f}  {I_t:>12.8f}  {err_t:>10.6f}")