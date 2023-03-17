import numpy as np
from scipy.optimize import least_squares

# Función de activación sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Función de coste
def funcion_costo(weights, x, y):
    w11, w12, b1, w21, w22, b2, w31, w32, b3 = weights
    z1 = sigmoide(w11 * x[0] + w21 * x[1] + b1)
    z2 = sigmoide(w12 * x[0] + w22 * x[1] + b2)
    y_esperado = sigmoide(w31 * z1 + w32 * z2 + b3)
    return (y_esperado - y)**2

# Valores iniciales de los pesos sinápticos y los sesgos
pesos_inicial = np.array([0.6, 0.3, 0.5, 0.4, 0.8, -0.5, 0.5, -0.4, 0.4])

# Datos de entrada y salida
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Minimizar la función de coste utilizando el método de Levenberg-Marquardt
result = least_squares(funcion_costo, pesos_inicial, args=(x.T, y))

# Mostrar los pesos minimizados
print(f"Pesos minimizados: {result.x}")