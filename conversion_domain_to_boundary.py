"""
Evaluacion de integral de dominio mediante Dual Reciprocity (DR)
Dominio: cuadrado unitario [0,1]^2
Funcion: b(x,y) = x^2 + y^2
RBF: f(r) = 1 + r^3  =>  phi(r) = r^2/4 + r^5/25
Elementos constantes: 4 elementos, uno por cada lado
Solucion analitica: I = 2/3
"""

import numpy as np

# ─────────────────────────────────────────────
# 1. FUNCIONES RBF Y SOLUCION PARTICULAR
# ─────────────────────────────────────────────

def f_rbf(r):
    """RBF: f(r) = 1 + r^3"""
    return 1.0 + r**3

def phi(r):
    """Solucion particular: nabla^2 phi = f(r)
       phi(r) = r^2/4 + r^5/25"""
    return r**2 / 4.0 + r**5 / 25.0

def dphi_dr(r):
    """Derivada de phi respecto a r
       dphi/dr = r/2 + r^4/5"""
    return r / 2.0 + r**4 / 5.0

def b_func(x, y):
    """Funcion a integrar"""
    return x**2 + y**2

# ─────────────────────────────────────────────
# 2. GEOMETRIA: 4 ELEMENTOS CONSTANTES
# ─────────────────────────────────────────────
#
#  Lado 1 (bottom): y=0, x de 0 a 1, normal = (0,-1)
#  Lado 2 (right):  x=1, y de 0 a 1, normal = (1, 0)
#  Lado 3 (top):    y=1, x de 1 a 0, normal = (0, 1)
#  Lado 4 (left):   x=0, y de 1 a 0, normal = (-1,0)

# Cada elemento: [x1, y1, x2, y2]  (nodos extremos)
elements = np.array([
    [0.0, 0.0,  1.0, 0.0],   # bottom
    [1.0, 0.0,  1.0, 1.0],   # right
    [1.0, 1.0,  0.0, 1.0],   # top
    [0.0, 1.0,  0.0, 0.0],   # left
])

# Normales exteriores (constantes por elemento)
normals = np.array([
    [ 0.0, -1.0],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
])

# Longitud de cada elemento y jacobiano
def get_jacobian(elem):
    x1, y1, x2, y2 = elem
    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return L / 2.0   # jacobiano de t in [-1,1] -> longitud real

# Punto sobre el elemento dado parametro t in [-1,1]
def get_point(elem, t):
    x1, y1, x2, y2 = elem
    x = 0.5*(1-t)*x1 + 0.5*(1+t)*x2
    y = 0.5*(1-t)*y1 + 0.5*(1+t)*y2
    return np.array([x, y])

# ─────────────────────────────────────────────
# 3. PUNTOS DE GAUSS en [-1,1]
# ─────────────────────────────────────────────

n_gauss = 4
gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(n_gauss)

# ─────────────────────────────────────────────
# 4. NODOS DE COLOCACION INTERNOS
# ─────────────────────────────────────────────
#
# Grilla regular de puntos interiores (no sobre la frontera)

n_side = 4   # nodos por lado en la grilla interior
coords = np.linspace(0.1, 0.9, n_side)
collocation_nodes = []
for xi in coords:
    for yi in coords:
        collocation_nodes.append([xi, yi])
collocation_nodes = np.array(collocation_nodes)
N = len(collocation_nodes)

print(f"Numero de nodos de colocacion internos: {N}")

# ─────────────────────────────────────────────
# 5. SISTEMA DE COLOCACION RBF: F * alpha = b
# ─────────────────────────────────────────────

# Armar matriz F
F_mat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        xi = collocation_nodes[i]
        xj = collocation_nodes[j]
        r = np.linalg.norm(xi - xj)
        F_mat[i, j] = f_rbf(r)

# Vector b evaluado en los nodos de colocacion
b_vec = np.array([b_func(p[0], p[1]) for p in collocation_nodes])

# Resolver para alpha
alpha = np.linalg.solve(F_mat, b_vec)

# ─────────────────────────────────────────────
# 6. VECTOR q: integrales de frontera
# ─────────────────────────────────────────────
#
# q_j = sum_k J_k * sum_p w_p * (dphi/dr / r) * (x(tp)-xj).n_k

q_vec = np.zeros(N)

for j in range(N):
    xj = collocation_nodes[j]
    q_j = 0.0

    for k in range(4):
        elem = elements[k]
        nk   = normals[k]
        Jk   = get_jacobian(elem)

        for p in range(n_gauss):
            tp  = gauss_pts[p]
            wp  = gauss_wts[p]
            xp  = get_point(elem, tp)

            rvec = xp - xj
            r    = np.linalg.norm(rvec)

            if r < 1e-14:
                continue   # no deberia ocurrir con nodos interiores

            dphi = dphi_dr(r)
            dot  = np.dot(rvec, nk)

            q_j += wp * Jk * (dphi / r) * dot

    q_vec[j] = q_j

# ─────────────────────────────────────────────
# 7. RESULTADO FINAL
# ─────────────────────────────────────────────

I_dr       = np.dot(q_vec, alpha)
I_analitica = 2.0 / 3.0
error       = abs(I_dr - I_analitica) / I_analitica * 100

print(f"\n--- RESULTADO ---")
print(f"Integral analitica        : {I_analitica:.10f}")
print(f"Integral DR (numerica)    : {I_dr:.10f}")
print(f"Error relativo            : {error:.6f} %")