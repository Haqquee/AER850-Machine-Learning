import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
def getdata(filename):
    with open (filename,'r') as csvfile:
        csvData = pd.read_csv(csvfile,header=None)
        #Ordering the data properly
        csvData.sort_values(csvData.columns[0],axis=0,inplace=True)
        print(csvData)
        X = csvData.iloc[:,[0]].to_numpy()
        Y = csvData.iloc[:,[1]].to_numpy()
        print(X)
    return X , Y


X_test, Y_test = getdata("Dataset2/Dataset_2_test.csv")
X_train, y_train = getdata("Dataset2/Dataset_2_train.csv")
X_val, y_val = getdata("Dataset2/Dataset_2_valid.csv")






# # build the model
# model = Sequential()
# model.add(Dense(1, input_shape=(1,)))
# model.compile(loss='mean_squared_error', optimizer=SGD(lr=1e-6))
#
# # train the model with stochastic gradient descent
# epochs = 10000
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=1, validation_data=(X_val, y_val))
#
# # compute the MSE on the validation set for every epoch
# val_mse = history.history['val_loss']
# train_mse = history.history['loss']
#
# # plot the training and validation MSE for every epoch
# import matplotlib.pyplot as plt
# plt.plot(range(1, epochs+1), train_mse, label='Training MSE')
# plt.plot(range(1, epochs+1), val_mse, label='Validation MSE')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.legend()
# plt.show()

tf.config.list_physical_devices('GPU')