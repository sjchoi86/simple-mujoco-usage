
import time
import numpy as np
import matplotlib.pyplot as plt

def pr2T(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel()
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def rpy2r(rpy):
    """
        roll,pitch,yaw to R
    """
    roll  = rpy[0]
    pitch = rpy[1]
    yaw   = rpy[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    rot   = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert rot.shape == (3, 3)
    return rot

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

class TicTocClass():
    def __init__(self,name='tictoc'):
        """
            Init tic-toc
        """
        self.name  = name
        # Reset
        self.reset()

    def reset(self):
        """
            Reset 't_init'
        """
        self.t_init    = time.time()
        self.t_elapsed = 0.0

    def toc(self,VERBOSE=False):
        """
            Compute elapsed time
        """
        self.t_elapsed = time.time() - self.t_init
        if VERBOSE:
            print ("[%s] [%.3f]sec elapsed."%(self.name,self.t_elapsed))

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def get_colors(n,cm=plt.cm.rainbow):
    """
        Get different colors
    """
    colors = cm(np.linspace(0.0,1.0,n))
    return colors
