"""feature engineering and encoding"""

#one hot encoding
import numpy as np

def one_hot(x, k):
    v = np.zeros((k, 1))
    v[x-1, 0] = 1
    return v

#feature extraction
	
def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    return np.mean(x, axis=1,keepdims=True)

def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.mean(x, axis=0,keepdims=True).T

def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m,n=x.shape
    top_mean=np.mean(x[:m//2,:])
    bottom_mean=np.mean(x[m//2:,:])
    return np.array([[top_mean,bottom_mean]]).T

#MNIST Feature Extraction
def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples,m,n = x.shape
    return np.reshape(x, (n_samples,m*n)).T

def row_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    n_samples,m,n = x.shape
    result = np.zeros((m,n_samples))
    for i in range(n_samples):
        result[:,[i]]=np.mean(x[i], axis=1,keepdims=True)
    return result

def col_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    n_samples,m,n = x.shape
    result = np.zeros((n,n_samples))
    for i in range(n_samples):
        result[:,[i]]=np.mean(x[i], axis=0,keepdims=True).T
    return result

def top_bottom_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n_samples,m,n = x.shape
    result = np.zeros((2,n_samples))
    for i in range(n_samples):
        top_mean=np.mean(x[i][:m//2,:])
        bottom_mean=np.mean(x[i][m//2:,:])
        result[:,[i]]= np.array([[top_mean,bottom_mean]]).T
    return result











