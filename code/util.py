import numpy as np

def pr2T(p,R):
    """ convert pose to transformation matrix """
    p0 = p.ravel()
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T