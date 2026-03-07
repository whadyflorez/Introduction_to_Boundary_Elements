"""
Evaluacion de integral de dominio mediante Dual Reciprocity (DR)
Dominio: disco circular de radio R
Funcion: b(x,y) = x^2 + y^2
RBF: f(r) = 1 + r^3  =>  phi(r) = r^2/4 + r^5/25
Elementos constantes: Ne elementos sobre el circulo
Solucion analitica: I = pi * R^4 / 2
"""

import numpy as np

# ─────────────────────────────────────────────
# 1. PARAMETROS DEL PROBLEMA
# ─────────────────────────────────────────────

R      = 1.0    # radio del disco
Ne     = 16     # numero de elementos de contorno
n_col  = 4      # nodos de colocacion por radio (grilla polar)
n_gauss = 4     # puntos de Gauss por elemento

# ─────────────────────────────────────────────
# 2. FUNCIONES RBF Y SOLUCION PARTICULAR
# ─────────────────────────────────────────────

def f_rbf(r):
    """RBF: f(r) = 1 + r^3"""
    return 1.0 + r**3

def dphi_dr(r):
    """d/dr [ r^2/4 + r^5/25 ] = r/2 + r^4/5"""
    return r / 2.0 + r**4 / 5.0

def b_func(x, y):
    """Funcion a integrar: b = x^2 + y^2"""
    return x**2 + y**2

# ─────────────────────────────────────────────
# 3. GEOMETRIA: Ne ELEMENTOS CONSTANTES
#    sobre el circulo de radio R
# ─────────────────────────────────────────────
#
#  Los angulos de los Ne+1 nodos extremos son:
#  theta_k = 2*pi*k/Ne,  k = 0, 1, ..., Ne
#
#  Cada elemento k tiene:
#    extremo 1: (R cos(theta_k),   R sin(theta_k))
#    extremo 2: (R cos(theta_k+1), R sin(theta_k+1))
#    nodo central: promedio de extremos (ligeramente dentro del arco)
#    normal exterior: apunta hacia afuera del disco

angles = np.linspace(0.0, 2.0*np.pi, Ne+1)  # Ne+1 puntos, el ultimo = el primero

# Nodos extremos de cada elemento
x_nodes = R * np.cos(angles)
y_nodes = R * np.sin(angles)

# Construir elementos: [x1, y1, x2, y2]
elements = np.zeros((Ne, 4))
for k in range(Ne):
    elements[k] = [x_nodes[k], y_nodes[k], x_nodes[k+1], y_nodes[k+1]]

# Normal exterior de cada elemento (perpendicular a la cuerda, apuntando afuera)
# La normal exacta en el punto medio del arco apunta radialmente
# Con elementos constantes usamos la normal de la cuerda
normals = np.zeros((Ne, 2))
jacobians = np.zeros(Ne)
for k in range(Ne):
    x1, y1, x2, y2 = elements[k]
    dx = x2 - x1
    dy = y2 - y1
    L  = np.sqrt(dx**2 + dy**2)
    # normal exterior = rotar tangente 90 grados hacia afuera
    normals[k]   = np.array([dy, -dx]) / L
    jacobians[k] = L / 2.0   # jacobiano de t in [-1,1] -> longitud real

# ─────────────────────────────────────────────
# 4. PUNTOS DE GAUSS en [-1,1]
# ─────────────────────────────────────────────

gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(n_gauss)

def get_point(elem, t):
    """Punto sobre el elemento k en parametro t in [-1,1]"""
    x1, y1, x2, y2 = elem
    x = 0.5*(1-t)*x1 + 0.5*(1+t)*x2
    y = 0.5*(1-t)*y1 + 0.5*(1+t)*y2
    return np.array([x, y])

# ─────────────────────────────────────────────
# 5. NODOS DE COLOCACION INTERNOS
#    Grilla polar: n_rings anillos, n_theta puntos por anillo
# ─────────────────────────────────────────────

n_rings = 4
n_theta = 8
col_nodes = []

for i in range(1, n_rings+1):
    rho = R * i / (n_rings + 1)   # radios interiores, nunca sobre Gamma
    for j in range(n_theta):
        theta = 2.0 * np.pi * j / n_theta
        col_nodes.append([rho * np.cos(theta), rho * np.sin(theta)])

col_nodes = np.array(col_nodes)
N = len(col_nodes)

print(f"Numero de elementos de contorno : {Ne}")
print(f"Numero de nodos de colocacion   : {N}")

# ─────────────────────────────────────────────
# 6. SISTEMA RBF: F * alpha = b
# ─────────────────────────────────────────────

F_mat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        r = np.linalg.norm(col_nodes[i] - col_nodes[j])
        F_mat[i, j] = f_rbf(r)

b_vec = np.array([b_func(p[0], p[1]) for p in col_nodes])
alpha = np.linalg.solve(F_mat, b_vec)

# ─────────────────────────────────────────────
# 7. VECTOR q: integrales de frontera
# ─────────────────────────────────────────────

q_vec = np.zeros(N)

for j in range(N):
    xj  = col_nodes[j]
    q_j = 0.0

    for k in range(Ne):
        elem = elements[k]
        nk   = normals[k]
        Jk   = jacobians[k]

        for p in range(n_gauss):
            tp   = gauss_pts[p]
            wp   = gauss_wts[p]
            xp   = get_point(elem, tp)
            rvec = xp - xj
            r    = np.linalg.norm(rvec)

            if r < 1e-14:
                continue

            dphi = dphi_dr(r)
            dot  = np.dot(rvec, nk)
            q_j += wp * Jk * (dphi / r) * dot

    q_vec[j] = q_j

# ─────────────────────────────────────────────
# 8. RESULTADO FINAL
# ─────────────────────────────────────────────

I_dr        = np.dot(q_vec, alpha)
I_analitica = np.pi * R**4 / 2.0
error       = abs(I_dr - I_analitica) / I_analitica * 100

print(f"\n--- RESULTADO ---")
print(f"Integral analitica        : {I_analitica:.10f}")
print(f"Integral DR (numerica)    : {I_dr:.10f}")
print(f"Error relativo            : {error:.6f} %")

# ─────────────────────────────────────────────
# 9. ESTUDIO DE CONVERGENCIA vs Ne
# ─────────────────────────────────────────────

print(f"\n--- CONVERGENCIA vs numero de elementos Ne ---")
print(f"{'Ne':>6}  {'I_DR':>14}  {'Error (%)':>12}")
print("-" * 38)

for Ne_test in [4, 8, 16, 32, 64]:

    angles_t  = np.linspace(0.0, 2.0*np.pi, Ne_test+1)
    xn = R * np.cos(angles_t)
    yn = R * np.sin(angles_t)

    elems_t = np.zeros((Ne_test, 4))
    norms_t = np.zeros((Ne_test, 2))
    jacs_t  = np.zeros(Ne_test)

    for k in range(Ne_test):
        x1, y1, x2, y2 = xn[k], yn[k], xn[k+1], yn[k+1]
        elems_t[k] = [x1, y1, x2, y2]
        dx = x2 - x1
        dy = y2 - y1
        L  = np.sqrt(dx**2 + dy**2)
        norms_t[k] = [dy/L, -dx/L]
        jacs_t[k]  = L / 2.0

    q_t = np.zeros(N)
    for j in range(N):
        xj  = col_nodes[j]
        q_j = 0.0
        for k in range(Ne_test):
            nk = norms_t[k]
            Jk = jacs_t[k]
            for p in range(n_gauss):
                xp   = get_point(elems_t[k], gauss_pts[p])
                rvec = xp - xj
                r    = np.linalg.norm(rvec)
                if r < 1e-14:
                    continue
                dot  = np.dot(rvec, nk)
                q_j += gauss_wts[p] * Jk * (dphi_dr(r) / r) * dot
        q_t[j] = q_j

    I_t   = np.dot(q_t, alpha)
    err_t = abs(I_t - I_analitica) / I_analitica * 100
    print(f"{Ne_test:>6}  {I_t:>14.10f}  {err_t:>12.6f}")