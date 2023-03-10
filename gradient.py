import sympy as sp

def GetGradient(fuction):
    return sp.Matrix([fuction.diff(v) for v in list(fuction.free_symbols)])

def GetGradientEvaluate(fuction, x0):
    return sp.Matrix([fuction.diff(v).subs([(list(fuction.free_symbols)[i], x0[i]) for i in range(len(x0))]) for v in list(fuction.free_symbols)])

def GetMagnitude(gradient, x0):
    return gradient.norm().subs([(list(gradient.free_symbols)[i], x0[i]) for i in range(len(x0))])