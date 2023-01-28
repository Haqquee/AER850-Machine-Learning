import numpy as np
import csv
import matplotlib.pyplot as plt

def getdata(filename):
    with open (filename,'r') as csvfile:
        Dataset_1_test = csv.reader(csvfile)   
        X = []
        Y = []
        for row in Dataset_1_test:
            x = row[0]
            y = row[1]
            X.append(x)
            Y.append(y)   
        
    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)
    # convert rank 1 array to rank 2 array
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    return X , Y

X_train, Y_train = getdata("Dataset1/Dataset_1_train.csv")
print(X_train.shape)
print(Y_train.shape)
X_valid, Y_valid = getdata("Dataset1/Dataset_1_valid.csv")
print(X_valid.shape)
print(Y_valid.shape)

def getfeaturematrix (X,polynomial):
    Xm = []
    for i in range(0,polynomial+1):
        a = np.power(X,i)
        Xm.append(a)
    Xm = np.array(Xm)
    Xm = np.squeeze(Xm, axis=(2,)).T
    print(Xm)
    return Xm

Xm_train = getfeaturematrix (X_train,20)
feature_no = Xm_train.shape[1]
print(Xm_train.shape)

Xm_valid = getfeaturematrix (X_valid,20)
print(Xm_valid.shape)
Xm_train = getfeaturematrix (X_train,20)
feature_no = Xm_train.shape[1]
print(Xm_train.shape)
Xm_valid = getfeaturematrix (X_valid,20)
print(Xm_valid.shape)


def computeW (Xm_train,Y_train):
    a = np.dot(Xm_train.T,Xm_train) 
    a = np.linalg.pinv(a)
    W = np.dot(a, np.dot(Xm_train.T,Y_train))
    return W

def computeMSE (Hypothesis,Y):
    MSE = np.mean((Hypothesis-Y)**2) 
    return MSE


MSE_traindata = []
MSE_validdata = []
W = computeW (Xm_train,Y_train)
Hypothesis_train = np.dot(Xm_train,W)
Hypothesis_valid = np.dot(Xm_valid,W)
MSE_traindata = computeMSE (Hypothesis_train,Y_train)
MSE_validdata = computeMSE (Hypothesis_valid,Y_valid)

print(MSE_traindata)
print(MSE_validdata)

plt.scatter(X_train,Y_train)
plt.scatter(X_train,Hypothesis_train)
plt.legend(['Y (train)','Y (train) approximation'])
plt.xlabel('example data')
plt.ylabel('value of Y')
plt.show()

plt.scatter(X_valid,Y_valid)
plt.scatter(X_valid,Hypothesis_valid)
plt.legend(['Y (valid)','Y (valid) approximation'])
plt.xlabel('example data')
plt.ylabel('value of Y')
plt.show()
