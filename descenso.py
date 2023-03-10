import sympy as sp
import gradient as grad

TOL = 1e-6

x, y, a = sp.symbols('x y a')
f = sp.Function('f')

v = sp.Matrix([x, y])

f = x ** 2 + y ** 2 - (4 * x) - (2 * y) + 7


x_0 = 0
y_0 = 5000
gradient = grad.GetGradient(f)

def GetAlpha(h, x0):
    alpha = 0
    factor_h = sp.factor(h.diff(a))
    eval_alpha = factor_h.evalf(subs={x: x0[0], y: x0[1]})
    alpha_list = sp.solve(eval_alpha)

    for value in alpha_list:
            if(value > 0):
                alpha = value

    return alpha

magnitude = 1
k = 1

while True:
    d = - gradient
    xy_alpha = v + (a * d)
    f_alpha = f.subs([(x, xy_alpha[0]), (y, xy_alpha[1])])
    h = f_alpha - f
    alpha = GetAlpha(h, [x_0, y_0])

    if(alpha == 0): break

    x_0 = x_0 + (alpha * d[0].subs([(x, x_0), (y, y_0)]))
    y_0 = y_0 + (alpha * d[1].subs([(x, x_0), (y, y_0)]))

    magnitude = grad.GetMagnitude(gradient, [x_0, y_0])

    if(magnitude < TOL): break

    print(f"Vector: {[x_0, y_0]}; Magnitude: {magnitude}; Iteration: {k}")

    k += 1


print(f"Converge en el punto {[x_0, y_0]} con una tolerancia del {magnitude} tras {k} iteraciones.")