import numpy as np

def linearFitting(_x: np, _y: np, xFrom:float=None, xTo:float=None):
    x = _x.copy()
    y = _y.copy()

    if (xFrom != None) or (xTo != None):
        fromIndex = None
        toIndex = None
        if xFrom != None:
            for i, val in enumerate(x):
                if val >= xFrom:
                    fromIndex = i
                    break
        if xTo != None:
            for i, val in enumerate(x):
                if val > xTo:
                    toIndex = i-1
                    break
        x = x[fromIndex:toIndex]
        y = y[fromIndex:toIndex]

    results = {}

    coeffs = np.polyfit(x, y, 1)

     # Polynomial Coefficients
    results['polynomial'] = coeffs

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results