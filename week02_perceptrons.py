"""excercises on perceptrons"""
import numpy as np

#implement perceptron
def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    d=data.shape[0]
    n=data.shape[1]

    th=np.zeros((d,1))
    th0=0.    
    for t in range(T):
      changed= False
      for i in range(n):
        if labels[:,i]*(np.dot(np.transpose(th),data[:,[i]])+th0)<=0:
          th=th+data[:,[i]]*labels[:,i]	
          th0=th0+labels[:,[i]]	
          changed= True
      if not changed: 
        break
    return (th,th0)

def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    d=data.shape[0]
    n=data.shape[1]

    th=np.zeros((d,1)); ths=np.zeros((d,1))
    th0=0.0; th0s=0.0

    for t in range(T):
      for i in range(n):
        if labels[:,i]*(np.dot(np.transpose(th),data[:,[i]])+th0)<=0:
          th=th+data[:,[i]]*labels[:,i]	
          th0=th0+labels[:,[i]]	
        ths = ths + th
        th0s = th0s + th0
    return (ths/(n*T),th0s/(n*T))

#evaluate classifier
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th,th0 = learner(data_train,labels_train)
    return score(data_test,labels_test, th, th0)/data_test.shape[1] #from previous week

#evaluate learning algo
def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    accuracy_tot=0.0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        accuracy_tot+=eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return accuracy_tot/it 

#evaluate learning algorithm with fixed dataset
def xval_learning_alg(learner, data, labels, k):
    #cross validation of learning algorithm
    D = np.array_split(data,k,axis=1)
    L= np.array_split(labels,k,axis=1)
    score_sum=0.0
    for j in range(k):
      D_j=D[j]
      D_minus_j=np.concatenate(D[0:j]+D[j+1:],axis=1)
      L_j=L[j]
      L_minus_j=np.concatenate(L[0:j]+L[j+1:],axis=1)
      score_sum+=eval_classifier(learner, D_minus_j, L_minus_j, D_j , L_j)
    return score_sum/k
