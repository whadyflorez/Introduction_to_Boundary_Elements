# =============================================================================
# BEM - Ecuación de Laplace 2D con elementos constantes
# Dominio: cuadrado unitario [0,1] x [0,1]
# Malla:   4 elementos de contorno (uno por lado) + 1 nodo interno
#
# Condiciones de contorno:
#   Lado inferior  (y=0): q = 0  (Neumann, flujo nulo)
#   Lado derecho   (x=1): u = 20 (Dirichlet)
#   Lado superior  (y=1): u = 100(Dirichlet)
#   Lado izquierdo (x=0): q = 0  (Neumann, flujo nulo)
#
# Formulación integral de contorno (elemento constante):
#   c(x)*u(x) + ∫ q*(x,y) u(y) dΓ = ∫ u*(x,y) q(y) dΓ
#
# donde:
#   u*(x,y) = -(1/2π) ln(r)          solución fundamental (Laplace 2D)
#   q*(x,y) = -(1/2π) ∂(ln r)/∂n     derivada normal de u*
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. CUADRATURA DE GAUSS-LEGENDRE (6 puntos)
# ─────────────────────────────────────────────────────────────────────────────
# Pesos y puntos en el intervalo [-1, 1]
gw = np.array([0.3607615730481386,  0.3607615730481386,
               0.4679139345726910,  0.4679139345726910,
               0.1713244923791704,  0.1713244923791704])
gp = np.array([0.6612093864662645, -0.6612093864662645,
              -0.2386191860831969,  0.2386191860831969,
              -0.9324695142031521,  0.9324695142031521])
NGP = 6   # número de puntos de Gauss

# ─────────────────────────────────────────────────────────────────────────────
# 2. GEOMETRÍA: 4 NODOS DE ESQUINA y 4 NODOS CENTRALES (colocación)
# ─────────────────────────────────────────────────────────────────────────────
#
#   3 ──────── 2
#   |          |
#   |          |
#   0 ──────── 1
#
#  Elementos (sentido antihorario, normal apuntando hacia afuera):
#   Elem 0: nodo 0 → nodo 1  (lado inferior,  y=0)
#   Elem 1: nodo 1 → nodo 2  (lado derecho,   x=1)
#   Elem 2: nodo 2 → nodo 3  (lado superior,  y=1)
#   Elem 3: nodo 3 → nodo 0  (lado izquierdo, x=0)

nodos = np.array([
    [0.0, 0.0],   # nodo 0
    [1.0, 0.0],   # nodo 1
    [1.0, 1.0],   # nodo 2
    [0.0, 1.0],   # nodo 3
])

# Conectividad: cada fila es [nodo_inicio, nodo_fin]
conectividad = np.array([
    [0, 1],   # elem 0: inferior
    [1, 2],   # elem 1: derecho
    [2, 3],   # elem 2: superior
    [3, 0],   # elem 3: izquierdo
])
N_ELEM = 4   # número de elementos de contorno

# Punto de colocación = centro de cada elemento
colocacion = np.array([
    (nodos[conectividad[e, 0]] + nodos[conectividad[e, 1]]) / 2
    for e in range(N_ELEM)
])

# Nodo interno donde evaluaremos u al final
interno = np.array([[0.5, 0.5]])

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONDICIONES DE CONTORNO
# ─────────────────────────────────────────────────────────────────────────────
# Por cada elemento se impone UNA condición:
#   tipo_bc[e] = 'D' → Dirichlet: u conocido, q incógnita
#   tipo_bc[e] = 'N' → Neumann:   q conocido, u incógnita

tipo_bc  = ['N', 'D', 'D', 'N']   # inferior, derecho, superior, izquierdo
valor_bc = [0.0, 20.0, 100.0, 0.0]  # valor conocido en cada elemento

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNCIONES DE FORMA LINEALES (para mapear puntos sobre el elemento)
# ─────────────────────────────────────────────────────────────────────────────
def shape(xi):
    """Funciones de forma en coordenada paramétrica xi ∈ [-1, 1]."""
    return np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])

def dshape(xi):
    """Derivadas de las funciones de forma (constantes)."""
    return np.array([-0.5, 0.5])

# ─────────────────────────────────────────────────────────────────────────────
# 5. SOLUCIONES FUNDAMENTALES DE LAPLACE 2D
# ─────────────────────────────────────────────────────────────────────────────
def u_star(x, y):
    """
    Solución fundamental de Laplace 2D.
    u*(x,y) = -(1/2π) ln(r),   r = |x - y|
    """
    r = np.linalg.norm(x - y)
    return -(1.0 / (2 * np.pi)) * np.log(r)

def q_star(x, y, n):
    """
    Derivada normal de la solución fundamental.
    q*(x,y,n) = (1/2π) (x-y)·n / r²
    """
    rv = x - y                          # vector x - y
    r  = np.linalg.norm(rv)
    return (1.0 / (2 * np.pi)) * np.dot(rv, n) / r**2

# ─────────────────────────────────────────────────────────────────────────────
# 6. INTEGRALES DE ELEMENTO: coeficientes H_ij y G_ij
# ─────────────────────────────────────────────────────────────────────────────
def calcular_H(x_col, elem_nodos):
    """
    Coeficiente H_ij: integral de q*(x_col, y) sobre el elemento j.
    Si el punto de colocación coincide con el centro del elemento (diagonal),
    por el criterio de cuerpo rígido (rigid body) H_ii se calcula después.
    Aquí retornamos 0 para ese caso.
    """
    x1, x2 = elem_nodos
    x_mid   = 0.5 * (x1 + x2)

    # Caso singular: punto de colocación en el elemento → integral = 0
    # (se completará luego con el criterio de cuerpo rígido)
    if np.linalg.norm(x_col - x_mid) < 1e-12:
        return 0.0

    integral = 0.0
    for k in range(NGP):
        N  = shape(gp[k])
        dN = dshape(gp[k])
        y  = N[0] * x1 + N[1] * x2           # punto en el elemento
        dy = dN[0] * x1 + dN[1] * x2         # tangente (dx/dxi)
        ds = np.linalg.norm(dy)               # jacobiano
        n  = np.array([dy[1], -dy[0]]) / ds   # normal unitaria (saliente)
        integral += gw[k] * q_star(x_col, y, n) * ds
    return integral

def calcular_G(x_col, elem_nodos):
    """
    Coeficiente G_ij: integral de u*(x_col, y) sobre el elemento j.
    Para la integral singular (punto de colocación sobre el elemento)
    se usa la fórmula analítica exacta.
    """
    x1, x2 = elem_nodos
    x_mid   = 0.5 * (x1 + x2)
    L       = np.linalg.norm(x2 - x1)        # longitud del elemento

    # Caso singular: fórmula analítica
    # G_diag = (L/2π) [ln(2/L) + 1]
    if np.linalg.norm(x_col - x_mid) < 1e-12:
        return (L / (2 * np.pi)) * (np.log(2.0 / L) + 1.0)

    integral = 0.0
    for k in range(NGP):
        N  = shape(gp[k])
        dN = dshape(gp[k])
        y  = N[0] * x1 + N[1] * x2
        dy = dN[0] * x1 + dN[1] * x2
        ds = np.linalg.norm(dy)
        integral += gw[k] * u_star(x_col, y) * ds
    return integral

# ─────────────────────────────────────────────────────────────────────────────
# 7. ENSAMBLE DE LAS MATRICES H y G
# ─────────────────────────────────────────────────────────────────────────────
# Para N_ELEM puntos de colocación de contorno + n_int nodos internos
N_INT   = len(interno)
N_TOTAL = N_ELEM + N_INT   # filas totales

H = np.zeros((N_TOTAL, N_ELEM))
G = np.zeros((N_TOTAL, N_ELEM))

# Todos los puntos de colocación (contorno + internos)
todos_col = np.vstack([colocacion, interno])

for i, x_col in enumerate(todos_col):
    for j in range(N_ELEM):
        elem_nodos = [nodos[conectividad[j, 0]], nodos[conectividad[j, 1]]]
        H[i, j] = calcular_H(x_col, elem_nodos)
        G[i, j] = calcular_G(x_col, elem_nodos)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CRITERIO DE CUERPO RÍGIDO para la diagonal de H (solo filas de contorno)
#    c(x)*1 + Σ H_ij * 1 = 0  →  H_ii = -Σ_{j≠i} H_ij
#    Para puntos en contorno suave c = 0.5, pero la suma total debe = 0.
# ─────────────────────────────────────────────────────────────────────────────
for i in range(N_ELEM):
    H[i, i] = -np.sum(H[i, :])   # suma de toda la fila (H_ii ya era 0)

# Para nodos internos c = 1, la suma Σ H_ij = 0 también (campo constante)
for i in range(N_ELEM, N_TOTAL):
    H[i, i % N_ELEM] = -np.sum(H[i, :])   # ajuste si coincide (no ocurre aquí)

# ─────────────────────────────────────────────────────────────────────────────
# 9. ENSAMBLE DEL SISTEMA  A * X = B
#
#  Incógnitas X:
#    - u en elementos Neumann   (q conocido)
#    - q en elementos Dirichlet (u conocido)
#
#  Para cada elemento:
#    Neumann  → columna de H  va a A  (u desconocido)
#              columna de G  va a B  (multiplicada por q conocido, con signo -)
#    Dirichlet→ columna de G  va a A  con signo -  (q desconocido)
#              columna de H  va a B  (multiplicada por u conocido, con signo -)
# ─────────────────────────────────────────────────────────────────────────────
# Usamos solo las N_ELEM filas de contorno para el sistema (las internas
# se usan al final para post-proceso)
A = np.zeros((N_ELEM, N_ELEM))
B = np.zeros(N_ELEM)

# Mapeo: incógnita k → (elemento, tipo)
incognitas = []   # lista de (indice_elemento, 'u' o 'q')

for j in range(N_ELEM):
    if tipo_bc[j] == 'N':
        # q conocido → u es incógnita → columna H va a A
        A[:, len(incognitas)] = H[:N_ELEM, j]
        B -= valor_bc[j] * G[:N_ELEM, j]
        incognitas.append((j, 'u'))
    else:
        # u conocido → q es incógnita → columna -G va a A
        A[:, len(incognitas)] = -G[:N_ELEM, j]
        B -= valor_bc[j] * H[:N_ELEM, j]
        incognitas.append((j, 'q'))

# ─────────────────────────────────────────────────────────────────────────────
# 10. RESOLUCIÓN DEL SISTEMA
# ─────────────────────────────────────────────────────────────────────────────
X = np.linalg.solve(A, B)

# ─────────────────────────────────────────────────────────────────────────────
# 11. RECONSTRUCCIÓN DE u y q en todos los elementos
# ─────────────────────────────────────────────────────────────────────────────
u_contorno = np.zeros(N_ELEM)
q_contorno = np.zeros(N_ELEM)

for k, (elem_idx, tipo) in enumerate(incognitas):
    if tipo == 'u':
        u_contorno[elem_idx] = X[k]
    else:
        q_contorno[elem_idx] = X[k]

# Asignar valores conocidos
for j in range(N_ELEM):
    if tipo_bc[j] == 'D':
        u_contorno[j] = valor_bc[j]
    else:
        q_contorno[j] = valor_bc[j]

# ─────────────────────────────────────────────────────────────────────────────
# 12. POST-PROCESO: u en el nodo interno con la representación integral
#     u(x) = Σ_j G_ij * q_j - Σ_j H_ij * u_j   (c=1 para puntos internos)
# ─────────────────────────────────────────────────────────────────────────────
u_interno = np.zeros(N_INT)
for i in range(N_INT):
    fila = N_ELEM + i
    u_interno[i] = (np.dot(G[fila, :], q_contorno)
                  - np.dot(H[fila, :], u_contorno))

# ─────────────────────────────────────────────────────────────────────────────
# 13. RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
nombres_lados = ['Inferior (y=0)', 'Derecho (x=1)',
                 'Superior (y=1)', 'Izquierdo (x=0)']

print("=" * 55)
print("  SOLUCIÓN BEM - Ecuación de Laplace 2D  ")
print("  4 elementos constantes de contorno      ")
print("=" * 55)
print(f"\n{'Elemento':<22} {'u (potencial)':>15} {'q (flujo)':>12}")
print("-" * 52)
for j in range(N_ELEM):
    print(f"  {nombres_lados[j]:<20} {u_contorno[j]:>15.4f} {q_contorno[j]:>12.4f}")

print(f"\n{'Nodo interno (0.5, 0.5)':<22} u = {u_interno[0]:.4f}")
print("\nNota: solución analítica en (0.5,0.5) ≈ 57.08")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# 14. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.15, 1.15)
ax.set_title("BEM - Laplace 2D\n4 elementos constantes", fontsize=12)
ax.set_xlabel("x"); ax.set_ylabel("y")

colores_bc = {'N': '#4C9BE8', 'D': '#E8694C'}

for j in range(N_ELEM):
    x1 = nodos[conectividad[j, 0]]
    x2 = nodos[conectividad[j, 1]]
    color = colores_bc[tipo_bc[j]]
    ax.plot([x1[0], x2[0]], [x1[1], x2[1]], '-', color=color, lw=4)
    xm = colocacion[j]
    label = (f"u={u_contorno[j]:.1f}" if tipo_bc[j]=='D'
             else f"q={q_contorno[j]:.2f}")
    ax.annotate(label, xy=xm, fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Nodo interno
ax.plot(*interno[0], 'ko', ms=8)
ax.annotate(f"u={u_interno[0]:.2f}", xy=interno[0],
            xytext=(0.55, 0.45), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='k'),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

from matplotlib.lines import Line2D
leyenda = [Line2D([0],[0], color='#E8694C', lw=4, label='Dirichlet (u conocido)'),
           Line2D([0],[0], color='#4C9BE8', lw=4, label='Neumann   (q conocido)')]
ax.legend(handles=leyenda, loc='lower right', fontsize=9)
ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("bem_laplace_4elem.png", dpi=150)
plt.show()
print("\nFigura guardada: bem_laplace_4elem.png")
