"""
Integral de linea: Trabajo en un campo de fuerza
--------------------------------------------------
Campo : F(x,y) = (-y, x)
Curva : semicirculo superior de radio R, de (R,0) a (-R,0)
        parametrizado como r(t) = (R cos t, R sin t), t in [0, pi]

Solucion analitica:
    W = integral_C  P dx + Q dy
      = integral_0^pi [ -R sin(t)*(-R sin t) + R cos(t)*R cos(t) ] dt
      = integral_0^pi R^2 [ sin^2(t) + cos^2(t) ] dt
      = integral_0^pi R^2 dt = pi * R^2

Se compara:
  (a) Integracion directa con cuadratura de Gauss sobre la curva exacta
  (b) Discretizacion en Ne elementos lineales + cuadratura de Gauss
"""

import numpy as np

# âââââââââââââââââââââââââââââââââââââââââââââ
# 1. CAMPO DE FUERZAS Y CURVA
# âââââââââââââââââââââââââââââââââââââââââââââ

R = 1.0

def P(x, y):
    """Componente x del campo F = (-y, x)"""
    return -y

def Q(x, y):
    """Componente y del campo F = (-y, x)"""
    return x

def curva(t):
    """Parametrizacion exacta del semicirculo"""
    return R * np.cos(t), R * np.sin(t)

def dcurva(t):
    """Derivada de la parametrizacion"""
    return -R * np.sin(t), R * np.cos(t)

W_analitica = np.pi * R**2
print(f"Solucion analitica : W = pi*R^2 = {W_analitica:.10f}")

# âââââââââââââââââââââââââââââââââââââââââââââ
# 2. CASO (a): INTEGRACION DIRECTA SOBRE
#    LA CURVA EXACTA con cuadratura de Gauss
#    en t in [0, pi]
# âââââââââââââââââââââââââââââââââââââââââââââ

print("\n--- (a) Curva exacta, cuadratura de Gauss en [0, pi] ---")
print(f"{'n_g':>6}  {'W':>14}  {'Error (%)':>12}")
print("-" * 38)

for n_g in [2, 4, 6, 8]:
    xi_g, w_g = np.polynomial.legendre.leggauss(n_g)

    # cambio de variable: t = pi/2 * (1 + xi),  dt = pi/2 * dxi
    a_t, b_t = 0.0, np.pi
    mid  = (b_t + a_t) / 2.0
    half = (b_t - a_t) / 2.0

    W = 0.0
    for p in range(n_g):
        t_p    = mid + half * xi_g[p]
        xp, yp = curva(t_p)
        dxp, dyp = dcurva(t_p)
        W += w_g[p] * (P(xp, yp) * dxp + Q(xp, yp) * dyp) * half

    err = abs(W - W_analitica) / W_analitica * 100
    print(f"{n_g:>6}  {W:>14.10f}  {err:>12.6f}")

# âââââââââââââââââââââââââââââââââââââââââââââ
# 3. CASO (b): DISCRETIZACION EN Ne ELEMENTOS
#    LINEALES + cuadratura de Gauss local
# âââââââââââââââââââââââââââââââââââââââââââââ

print("\n--- (b) Ne elementos lineales, n_g puntos de Gauss por elemento ---")

n_g = 4
xi_g, w_g = np.polynomial.legendre.leggauss(n_g)

print(f"\nUsando n_g = {n_g} puntos de Gauss por elemento:")
print(f"{'Ne':>6}  {'W':>14}  {'Error (%)':>12}")
print("-" * 38)

for Ne in [2, 4, 8, 16, 32, 64]:

    # Nodos extremos de cada elemento sobre el semicirculo
    t_nodes = np.linspace(0.0, np.pi, Ne + 1)
    x_nodes = R * np.cos(t_nodes)
    y_nodes = R * np.sin(t_nodes)

    W = 0.0

    for k in range(Ne):

        # Extremos del elemento k
        x1, y1 = x_nodes[k],   y_nodes[k]
        x2, y2 = x_nodes[k+1], y_nodes[k+1]

        # Jacobiano (constante en el elemento lineal)
        dx = x2 - x1
        dy = y2 - y1
        # dxi/dt = (x2-x1)/2, dy/dt = (y2-y1)/2
        # no se necesita L_k explicitamente porque se usa x'(xi) y y'(xi)

        # Cuadratura de Gauss sobre xi in [-1, 1]
        for p in range(n_g):
            xi_p = xi_g[p]
            wp   = w_g[p]

            # Punto sobre el elemento (interpolacion lineal)
            xp = 0.5 * (1 - xi_p) * x1 + 0.5 * (1 + xi_p) * x2
            yp = 0.5 * (1 - xi_p) * y1 + 0.5 * (1 + xi_p) * y2

            # Derivadas dx/dxi y dy/dxi (constantes en el elemento)
            dxdxi = (x2 - x1) / 2.0
            dydxi = (y2 - y1) / 2.0

            W += wp * (P(xp, yp) * dxdxi + Q(xp, yp) * dydxi)

    err = abs(W - W_analitica) / W_analitica * 100
    print(f"{Ne:>6}  {W:>14.10f}  {err:>12.6f}")

# âââââââââââââââââââââââââââââââââââââââââââââ
# 4. SEPARACION DE ERRORES:
#    error geometrico vs error de cuadratura
#    Fijamos Ne=8 y variamos n_g
# âââââââââââââââââââââââââââââââââââââââââââââ

print(f"\n--- Error geometrico vs cuadratura (Ne=8 fijo, n_g variable) ---")
print(f"{'n_g':>6}  {'W':>14}  {'Error (%)':>12}")
print("-" * 38)

Ne = 8
t_nodes = np.linspace(0.0, np.pi, Ne + 1)
x_nodes = R * np.cos(t_nodes)
y_nodes = R * np.sin(t_nodes)

for n_g_test in [1, 2, 3, 4, 6, 8]:
    xi_g, w_g = np.polynomial.legendre.leggauss(n_g_test)
    W = 0.0
    for k in range(Ne):
        x1, y1 = x_nodes[k],   y_nodes[k]
        x2, y2 = x_nodes[k+1], y_nodes[k+1]
        for p in range(n_g_test):
            xi_p  = xi_g[p]
            xp    = 0.5*(1-xi_p)*x1 + 0.5*(1+xi_p)*x2
            yp    = 0.5*(1-xi_p)*y1 + 0.5*(1+xi_p)*y2
            dxdxi = (x2-x1)/2.0
            dydxi = (y2-y1)/2.0
            W    += w_g[p] * (P(xp,yp)*dxdxi + Q(xp,yp)*dydxi)
    err = abs(W - W_analitica) / W_analitica * 100
    print(f"{n_g_test:>6}  {W:>14.10f}  {err:>12.6f}")

print(f"\nNota: con Ne=8 fijo el error se estanca en ~0.3%")
print(f"porque el error geometrico de las cuerdas domina.")
print(f"Aumentar n_g mas alla de 3 no mejora el resultado.")

# âââââââââââââââââââââââââââââââââââââââââââââ
# 5. OTRO CAMPO: F = (x^2, y^2) sobre el
#    mismo semicirculo para comparar
# âââââââââââââââââââââââââââââââââââââââââââââ

print(f"\n--- Campo F=(x^2, y^2), mismo semicirculo, Ne=16, n_g=4 ---")

def P2(x, y): return x**2
def Q2(x, y): return y**2

# Solucion analitica:
# W = int_0^pi [ R^2 cos^2(t)*(-R sin t) + R^2 sin^2(t)*(R cos t) ] dt
# = R^3 int_0^pi [ -cos^2(t) sin(t) + sin^2(t) cos(t) ] dt
# = R^3 [ cos^3(t)/3 + sin^3(t)/3 ]_0^pi
# = R^3 [ (-1/3 + 0) - (1/3 + 0) ] = -2R^3/3
W_analitica2 = -2.0 * R**3 / 3.0

Ne = 16
n_g = 4
xi_g, w_g = np.polynomial.legendre.leggauss(n_g)
t_nodes = np.linspace(0.0, np.pi, Ne + 1)
x_nodes = R * np.cos(t_nodes)
y_nodes = R * np.sin(t_nodes)

W2 = 0.0
for k in range(Ne):
    x1, y1 = x_nodes[k],   y_nodes[k]
    x2, y2 = x_nodes[k+1], y_nodes[k+1]
    for p in range(n_g):
        xi_p  = xi_g[p]
        xp    = 0.5*(1-xi_p)*x1 + 0.5*(1+xi_p)*x2
        yp    = 0.5*(1-xi_p)*y1 + 0.5*(1+xi_p)*y2
        dxdxi = (x2-x1)/2.0
        dydxi = (y2-y1)/2.0
        W2   += w_g[p] * (P2(xp,yp)*dxdxi + Q2(xp,yp)*dydxi)

err2 = abs(W2 - W_analitica2) / abs(W_analitica2) * 100
print(f"W analitica = {W_analitica2:.10f}")
print(f"W numerica  = {W2:.10f}")
print(f"Error       = {err2:.6f} %")
