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
lambd = 300

magnitude = 1
k = 1

while True:
    vector = [x_0, y_0]
    gradient = grad.GetGradient(f)
    hessian = hess.GetHessiana(f)
    inv_hessianlambda = hess.GetInverseHessian(hessian.applyfunc(lambda x: x + lambd))
    det_inv_hessianlambda = general.GetDet(inv_hessianlambda)

    s = - det_inv_hessianlambda * gradient

    x_1 = x_0 + s[0].evalf(subs={x: x_0, y: y_0})
    y_1 = y_0 + s[1].evalf(subs={x: x_0, y: y_0})

    print(vector)
    print(x_1)
    print(y_1)

    f0 = f.evalf(subs={x: x_0, y: y_0})
    f1 = f.evalf(subs={x: x_1, y: y_1})

    if(f1 < f0):
        lambd = lambd / 2
    else:
        lambd = 2 * lambd

    magnitude = grad.GetMagnitude(gradient, vector)

    if(abs(f1 - f0) < TOL and magnitude < TOL): break

    x_0 = x_1
    y_0 = y_1

    print(f"Vector: {[x_0, y_0]}; Magnitude: {magnitude}; Iteration: {k}")
    k += 1

print(f"Converge en el punto {[x_0, y_0]} con una tolerancia del {magnitude} tras {k} iteraciones.")
