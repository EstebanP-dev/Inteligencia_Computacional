import sympy as sp

def GetHessiana(fuction):
    symbols = list(fuction.free_symbols)
    n = len(symbols)
    hessian = sp.zeros(n)

    for i in range(n):
        for k in range(n):
            hessian[i, k] = fuction.diff(symbols[i], symbols[k])
    
    return hessian

def GetHessianaEvaluate(fuction, x0):
    symbols = list(fuction.free_symbols)
    n = len(symbols)
    hessian = sp.zeros(n)

    for i in range(n):
        for j in range(n):
            hessian[i, j] = fuction.diff(symbols[i], symbols[j]).subs([(symbols[k], x0[k]) for k in range(n)])
    
    return hessian


def GetInverseHessian(hessian):
    return hessian.inv()