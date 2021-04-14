"""logistic regression"""
import numpy as np

#gradient descent
def gd(f, df, x0, step_size_fn, num_steps):
  x=x0
  for i in range(num_steps):
    x=x-step_size_fn(1)*df(x)
  return (x,f(x))

#numerical gradient
def num_grad(f, delta=0.001):
    def df(x):
        g = np.zeros(x.shape)
        for i in range(x.shape[0]):
            a, b = x.copy(), x.copy()
            a[i, 0] -= delta
            b[i, 0] += delta
            fa, fb = f(a), f(b)
            g[i, 0] = (fb - fa)/(2*delta)
        return g
    return df

#Applying gradient descent to Linear Logistic Classifier objective
# returns a vector of the same shape as z
def sigmoid(z):
    return 1/(1+np.exp(-z)) 

# X is dxn, y is 1xn, th is dx1, th0 is 1x1
# returns a (1,n) array for the nll loss for each data point given th and th0 
def nll_loss(X, y, th, th0):
    g=np.zeros(y.shape)
    g=sigmoid(th.T.dot(X)+th0)
    return -(y*np.log(g)+(1-y)*np.log(1-g))
    
    
# X is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
# returns a np.float for the llc objective over the dataset
def llc_obj(X, y, th, th0, lam):
    return np.sum(nll_loss(X, y, th, th0))/X.shape[1]+lam*np.linalg.norm(th)**2
    
# returns an array of the same shape as z for the gradient of sigmoid(z)
def d_sigmoid(z):
    return (1-sigmoid(z))*sigmoid(z)

# returns a (d,n) array for the gradient of nll_loss(X, y, th, th0) with respect to th for each data point
def d_nll_loss_th(X, y, th, th0):
    g=sigmoid(np.dot(th.T,X)+th0)
    return X*(g-y)

# returns a (1,n) array for the gradient of nll_loss(X, y, th, th0) with respect to th0
def d_nll_loss_th0(X, y, th, th0):
    g=sigmoid(np.dot(th.T,X)+th0)
    return g-y


# returns a (d,1) array for the gradient of llc_obj(X, y, th, th0) with respect to th
def d_llc_obj_th(X, y, th, th0, lam):
    return np.mean(d_nll_loss_th(X, y, th, th0),axis=1,keepdims=True)+2*lam*th

# returns a (1,1) array for the gradient of llc_obj(X, y, th, th0) with respect to th0
def d_llc_obj_th0(X, y, th, th0, lam):
    return np.mean(d_nll_loss_th0(X, y, th, th0),axis=1,keepdims=True)

# returns a (d+1, 1) array for the full gradient as a single vector (which includes both th, th0)
def llc_obj_grad(X, y, th, th0, lam):
    return np.vstack((d_llc_obj_th(X, y, th, th0, lam),d_llc_obj_th0(X, y, th, th0, lam)))

def llc_min(data, labels, lam):
    """
    Parameters:
        data: dxn
        labels: 1xn
        lam: scalar
    Returns:
        same output as gd
    """
    #th0=np.array([[0]])
    #th=np.zeros(data.shape[0],1)
    th_adj=np.zeros((data.shape[0]+1,1))
    
    def llc_min_step_size_fn(i):
       return 2/(i+1)**0.5
    
    def f(th):
        return llc_obj(data, labels, th[:-1,:], th[-1,:], lam)
        
    def df(th):
        return llc_obj_grad(data, labels, th[:-1,:], th[-1,:], lam)
        
        
    return gd(f,df, th_adj, llc_min_step_size_fn, 10)
    
    
   

















