"""Convolutional Neural Networks"""

import numpy as np
import math

#Implementing Mini-batch Gradient Descent and Batch Normalization

class Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:, indices]  # Your code
            Y = Y[:, indices]  # Your code
            
            sum_loss = 0
            for j in range(math.floor(N/K)):
                if num_updates >= iters: break
                
                # Implement the main part of mini_gd here
                Xt = X[:,(j*K):(j*K+K)] # Your code
                Yt = Y[:,(j*K):(j*K+K)] # Your code
                Ypred=self.forward(Xt)
                sum_loss+=self.loss.forward(Ypred,Yt)            
                self.backward(self.loss.backward())
                self.sgd_step(lrate)
                
                num_updates += 1
        return -sum_loss



                

    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):    
        for m in self.modules: m.sgd_step(lrate)
		
		
class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1]) # m x 1
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1]) # m x 1
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, Z):# Z is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.Z = Z
        self.K = Z.shape[1]
        
        self.mus = np.mean(self.Z,axis=1,keepdims=True)  # Your Code
        self.vars = np.var(self.Z,axis=1,keepdims=True)  # Your Code

        # Normalize inputs using their mean and standard deviation
        self.norm = (self.Z-self.mus)/(self.vars**(.5)+self.eps)  # Your Code
            
        # Return scaled and shifted versions of self.norm
        return self.B+ self.norm*self.G  # Your Code

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        Z_min_mu = self.Z-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * Z_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(Z_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * Z_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def sgd_step(self, lrate):
        self.B = self.B-lrate*self.dLdB  # Your Code
        self.G = self.G-lrate*self.dLdG  # Your Code
        return 
































