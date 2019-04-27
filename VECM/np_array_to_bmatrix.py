import numpy as np

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\ ' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return ''.join(rv)

# Source: Roland Smith, https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
