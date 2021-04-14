"""Decision Trees and Nearest Neighbors"""

import numpy as np

#decision tree classifier
def calculate_entropy(data, labels):
    """
    Parameters:
        data is a size n list of data points (x1, x2)
        labels is a 1 by n array of the labels corresponding to the datapoints in data
    Returns:
        A float denoting the entropy in the dataset.
    """
    pm1=np.where(labels>0,1,0).sum()/len(data)
    pm0=np.where(labels<0,1,0).sum()/len(data)
    
    entr1=0 if pm1 in [0,1] else pm1*np.log2(pm1) 
    entr0=0 if pm0 in [0,1] else pm0*np.log2(pm0)
    
    return -entr1-entr0
    
def find_best_split(data, labels):
    """
    Parameters:
        data is a size n list of data points (x1, x2)
        labels is a 1 by n array of the labels corresponding to the datapoints in data
    Returns:
        A tuple where the first element denotes the axis of the split, either "x1" or "x2" and the second element denotes the value of the place to split along the axis
    """
    #line 45
    print(data,labels)
    min_entropy,best_split_value,best_split_axis=1e100,0,0
    for axes in [0,1]:
        
        coords=[i[axes] for i in data]
        
        index=[i for i in range(len(coords))]
        index.sort(key=coords.__getitem__)
        
        coords=[coords[i] for i in index]
        data_sorted=[data[i] for i in index]
        labels_sorted= labels[:,index]
        #labels_sorted=[labels[:,i] for i in index]
        
        for i in range(len(coords)-1):
            if coords[i]!=coords[i+1]: 
                split_value=(coords[i]+coords[i+1])/2
                #check if x+1 is correct
                lower_data,lower_labels= data_sorted[:i+1],labels_sorted[:,:i+1]
                higher_data,higher_labels=data_sorted[i+1:],labels_sorted[:,i+1:]
                
                print(lower_data,higher_data)
                print(higher_labels,'\n',lower_labels)
                #data_sorted,'\n',labels_sorted)
                
                #check weighted average thing
                higher_w=len(higher_data)/len(data)
                lower_w=len(lower_data)/len(data)
                entropy=higher_w*calculate_entropy(higher_data,higher_labels)+lower_w*calculate_entropy(lower_data,lower_labels)
                
                
                if entropy<min_entropy:
                    min_entropy=entropy
                    best_split_value=split_value
                    best_split_axis=axes
            
                print(entropy,best_split_value)
            
    best_split_axis=('x1','x2')[best_split_axis]
            
    return (best_split_axis,best_split_value)
    
def return_label(data, labels):
    """
    Parameters:
        data is a size n list of data points (x1, x2)
        labels is a 1 by n array of the labels corresponding to the datapoints in data
    Returns:
        a label, either 1 or -1, denoting what label to assign for the partition over data, labels
    """
    return (-1,1)[np.where(labels>0,1,0).sum()>np.where(labels<0,1,0).sum()]
    
    
def predict_datapoint(data, labels, datapoint,depth=2):
    """
    Parameters: 
        data is a size n list of datapoints (x1, x2)
        labels is a 1 by n array of labels corresponding to the datapoints in data
        datapoint is a tuple (x1, x2)
    Returns:
        A label, either -1 or 1, denoting what label to assign datapoint using a two-layer decision tree trained on data, labels
    """
    if depth==0:
        return return_label(data, labels)
    else:
        depth-=1
        axis,split=find_best_split(data, labels)
        axis_dict={'x1':0,'x2':1}
        axis=axis_dict[axis]
        coords=[i[axis] for i in data]
        
        #create left and right data and labels
        left_data,right_data=[],[]
        left_labels,right_labels=[],[]
        for i in range(len(data)):
            point=data[i]
            if point[axis]<split:
                left_data.append(point)
                left_labels.append(labels[0,i])
            else:
                right_data.append(point)
                right_labels.append(labels[0,i])
        right_labels=np.array([right_labels])
        left_labels=np.array([left_labels])
        #classify datapoint
        if datapoint[axis]<split:
            return predict_datapoint(left_data, left_labels, datapoint,depth)
        else:
            return predict_datapoint(right_data, right_labels, datapoint,depth)
		
		
		
#Nearest neighbor IMplementation
def euclidean(a,b):
    """
    Parameters:
        a is d by n array
        b is d by 1 array
    Returns :
        (n,) (i.e. one-dimensional, n elements) array: the pairwise Euclidean distance of b with respect to individual samples in  a
    """
    sum = []
    for i in a.T:
        sum.append(np.sqrt(np.sum((i-b.T)**2)))
    return np.array(sum)

def manhattan(a,b): 
    """
    Parameters:
        a is d by n array 
        b is d by 1 array
    Returns :
        (n,) (i.e. one-dimensional, n elements) array: the pairwise Manhattan distance of b with respect to individual samples in  a
    """
    sum = []
    for i in a.T:
        sum.append((np.sum(np.absolute(i-b.T))))
    return np.array(sum)

class KNN:
    def __init__(self, K, distance_metric, trainX, trainY):
        """
        Parameters:
            K is an int representing the number of closest neighbors to consider
            distance_metric is one of euclidean or manhattan
            trainX is d by n array
            trainY is 1 by n array
        """
        self.trainX = trainX
        self.trainY = trainY
        self.K = K
        self.metric = distance_metric
        
    def calc_distances(self, testX):
        """
        Parameters:
            testX is d by m np array
        Returns:
            an m x n np array D where D[i, j] is the distance between test sample i and train sample j
        """
        
        distances=self.metric(self.trainX,testX[:,[0]])
        for i in range(1,testX.shape[1]):        
            distances=np.vstack([distances,self.metric(self.trainX,testX[:,[i]])])
        return distances
        
    def find_top_neighbor_labels(self, dists):
        """
        Parameters:
            dists is  m x n np array D where D[i, j] is the distance between test sample i and train sample j
        Returns:
            an m x K np array L where L[i, j] is the label of the jth closest neighbor to test sample i
            in case of ties, the neighbor which appears first in the training set is chosen
        """
        #self.trainX[np.argsort(dists)]
        
        return np.squeeze(self.trainY[:,np.argsort(dists)])[:,:self.K]
        
    def predict(self, testX):
        """
        Parameters:
            testX is d by m np array
        Returns:
            predicted is (m,) np array P where P[i] is the predicted label for test sample i 
        """
        
        top_labels=self.find_top_neighbor_labels(self.calc_distances(testX))
        predictions=[]
        for i in top_labels:
            predictions.append(np.argmax(np.bincount(i)))
        return np.array(predictions)
        

    def score(self, testX, testY):
        """
        Parameters:
            testX is d by m np array of input data
            testY is 1 by m np array of labels for the input data
        Returns:
            a scalar: the accuracy of the KNN predictions across the test set
        """
        predictions = self.predict(testX)
        count_trues = (predictions.T == testY).sum()
        return count_trues/len(predictions)









































































