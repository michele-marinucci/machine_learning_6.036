"""Neural networks, part 1"""

import numpy as np

#ReLU activation
def dReLU_dz(z):
    v=[]
    [[v.append(0) if i<=0 else v.append(1)]for i in z]
    return np.array([v]).T

#not much code because this homework was more conceptual













