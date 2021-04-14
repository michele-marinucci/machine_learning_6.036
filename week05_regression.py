"""linear regression"""

import numpy as np
#gradient of the squared loss objective in linear regression

def d_lin_reg_th(x, th, th0):
    """
    Parameters:
        x is d by n : input data
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        d by n array : gradient of lin_reg(x, th, th0) with respect to th
    """
    return x
    
    
    return None
    
def d_square_loss_th(x, y, th, th0):
    """
    Parameters:
        x is d by n : input data
        y is 1 by n : output regression values
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        d by n array : gradient of square_loss(x, y, th, th0) with respect to th.
    
    This function should be a one-line expression that uses lin_reg and
    d_lin_reg_th.
    """
    return -2*(d_lin_reg_th(x, th, th0)*(y-lin_reg(x, th, th0)))

def d_mean_square_loss_th(x, y, th, th0):
    """
    Parameters:
        Same as above
    Returns:
        d by 1 array : gradient of mean_square_loss(x, y, th, th0) with respect to th.
    
    This function should be a one-line expression that uses d_square_loss_th.
    """
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)


def d_lin_reg_th0(x, th, th0):
    """
    Parameters:
        x is d by n : input data
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        1 by n array : gradient of lin_reg(x, th, th0) with respect to th0
    """
    return np.ones((1,x.shape[1]))
    
def d_square_loss_th0(x, y, th, th0):
    """
    Parameters:
        x is d by n : input data
        y is 1 by n : output regression values
        th is d by 1 : weights
        th0 is 1 by 1 or scalar
    Returns:
        1 by n array : gradient of square_loss(x, y, th, th0) with respect to th0.
    
    This function should be a one-line expression that uses lin_reg and
    d_lin_reg_th0.
    """
    return -2*(y-lin_reg(x, th, th0))

def d_mean_square_loss_th0(x, y, th, th0):
    """
    Parameters:
        Same as above
    Returns:
        1 by 1 array : gradient of mean_square_loss(x, y, th, th0) with respect to th0.
    
    This function should be a one-line expression that uses d_square_loss_th0.
    """
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

#add a regulizer
def d_ridge_obj_th(x, y, th, th0, lam):
    return d_mean_square_loss_th(x, y, th, th0)+2*lam*th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return d_mean_square_loss_th0(x, y, th, th0)

#stochastic gradient
def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    n = y.shape[1]
    prev_w = w0
    ws = []
    fs = []
    for i in range(max_iter):
        j = np.random.randint(n)
        Xj = X[:,j:j+1]; yj = y[:,j:j+1]
        ws.append(prev_w)
        fs.append(J(Xj, yj, prev_w))
        prev_w = prev_w - step_size_fn(i) * dJ(Xj, yj, prev_w)
    return prev_w, fs, ws
	












































