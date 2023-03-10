import sympy as sp
import gradient as grad
import hessian as hess
import general as general

TOL = 1e-6
x, y, a = sp.symbols('x y a')
f = sp.Function('f')
v = sp.Matrix([x, y])
f = x ** 2 + y ** 2 - (4 * x) - (2 * y) + 7

x_0 = 1
y_0 = 2

magnitude = 1
k = 1
while True:
    vector = [x_0, y_0]
    gradient = grad.GetGradient(f)
    hessian = hess.GetHessiana(f)
    inv_hess = hess.GetInverseHessian(hessian)
    det_inv_hess = general.GetDet(inv_hess)

    x_0 -= det_inv_hess.evalf(subs={x: x_0, y: y_0}) * gradient[0].evalf(subs={x: x_0, y: y_0})
    y_0 -= det_inv_hess.evalf(subs={x: x_0, y: y_0}) * gradient[1].evalf(subs={x: x_0, y: y_0})

    magnitude = grad.GetMagnitude(gradient, vector)

    if(magnitude < TOL): break

    print(f"Vector: {[x_0, y_0]}; Magnitude: {magnitude}; Iteration: {k}")
    k += 1

print(f"Converge en el punto {[x_0, y_0]} con una tolerancia del {magnitude} tras {k} iteraciones.")

