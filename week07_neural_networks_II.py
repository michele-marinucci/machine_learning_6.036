"""Neural networks part 2"""

import numpy as np



#linear module
class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights

class Linear(Module):
    def __init__(self, m, n):
        # initializes the weights randomly and offsets as 0
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        # store the input matrix for future use
        self.A = A   # (m x b)
        self.b = self.A.shape[1]
        return np.dot(self.W.T,self.A)+self.W0.dot(np.ones((1,self.b))) # (n x b)

    def backward(self, dLdZ):  
        # dLdZ is (n x b), uses stored self.A
        # store the derivatives for use in sgd_step and returd dLdA
        self.dLdW = self.A.dot(dLdZ.T)       
        self.dLdW0 = np.dot(dLdZ,np.ones((1,self.b)).T)      
        return self.W.dot(dLdZ)           

    def sgd_step(self, lrate):  # Gradient descent step
        self.W = self.W -lrate*self.dLdW          
        self.W0 = self.W0 -lrate*self.dLdW0          

#activation modules
class Tanh(Module):            # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)            
        return self.A

    def backward(self, dLdA):    # Uses stored self.A
        return dLdA*(1-self.A**2)       

#ReLU
class ReLU(Module):              # Layer activation
    def forward(self, Z):
        self.A = np.maximum(Z,0)            
        return self.A

    def backward(self, dLdA):    # uses stored self.A
        return dLdA          
    
#softmax
class SoftMax(Module):           # Output activation
    def forward(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z), axis=0)             

    def backward(self, dLdZ):    # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  
        # Returns the index of the most likely class for each point as vector of shape (b,)
        return np.argmax(Ypred, axis=0)           

#loss module
class NLL(Module):       # Loss
    def forward(self, Ypred, Y):
        # returns loss as a float
        self.Ypred = Ypred
        self.Y = Y
        return -np.sum(self.Y*np.log(self.Ypred)) 

    def backward(self):  # Use stored self.Ypred, self.Y
        # note, this is the derivative of loss with respect to the input of softmax
        return self.Ypred-self.Y      
	
#sequential	
class Sequential:
    def __init__(self, modules, loss):            # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        sum_loss = 0
        for it in range(iters):
            rand_index=np.random.randint(0,N)
            Xt=X[:,[rand_index]]
            Yt=Y[:,[rand_index]]
            Ypred=self.forward(Xt)
            sum_loss+=self.loss.forward(Ypred,Yt)            
            self.backward(self.loss.backward())
            self.sgd_step(lrate)
        return -sum_loss                                              # Your code

    def forward(self, Xt):                        # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                    # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):                    # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints current loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '\tAcc =', acc, '\tLoss =', cur_loss, flush=True)
    
#rest of the homework provided some background on PyTorch












