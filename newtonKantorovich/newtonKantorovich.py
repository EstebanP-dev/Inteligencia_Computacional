import numpy as np
from numpy.linalg import inv

def f(x, y, z):
    return np.array([
        x**2 + y**2 + z**2 - 3,
        x**2 - y + z**3 + 1,
        np.sin(x) + np.cos(y) - z
    ])

def jacobian(f, x, y, z):
    h = 1e-6
    jacobian = np.zeros((len(f(x, y, z)), len([x, y, z])))
    for i, x_i in enumerate([x, y, z]):
        x_plus_h = np.array([x, y, z])
        x_plus_h[i] += h
        jacobian[:,i] = (f(x_plus_h[0], x_plus_h[1], x_plus_h[2]) - f(x, y, z)) / h
    return jacobian

def hessian(f, x, y, z):
    h = 1e-6
    hessian = np.zeros((len(f(x, y, z)), len([x, y, z]), len([x, y, z])))
    for i, x_i in enumerate([x, y, z]):
        x_plus_h = np.array([x, y, z])
        x_plus_h[i] += h
        jacobian_plus_h = jacobian(f, x_plus_h[0], x_plus_h[1], x_plus_h[2])
        jacobian_minus_h = jacobian(f, x, y, z)
        hessian[:,:,i] = (jacobian_plus_h - jacobian_minus_h) / h
    return hessian

def newton_kantorovich(f, jacobian, hessian, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        j = jacobian(f, x[0], x[1], x[2])
        h = hessian(f, x[0], x[1], x[2])
        g = -1 * f(x[0], x[1], x[2])
        b = g.reshape(len(g), 1, 1)
        print(len(h))
        print(h.shape)
        print(g.shape)
        #print(A.shape)
        print(b.shape)
        dx = np.dot(inv(h), b).flatten()
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x

x0 = np.array([1, 1, 1])
root = newton_kantorovich(f, jacobian, hessian, x0)
print("Aproximación de la raíz:", root)
print("Valor de la función en la raíz:", f(root[0], root[1], root[2]))