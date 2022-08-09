
import time
import numpy as np

def pr2T(p,R):
    """ convert pose to transformation matrix """
    p0 = p.ravel()
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

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