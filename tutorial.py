from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np

np.random.seed(3)
tf.random.set_seed(3)
## same results printed for every run


## numpy import for calculation
Data_set = np.loadtxt("./dataset/ThoraricSurgery.csv", delimiter=",")
## lung cancer  data
X = Data_set[:,0:17]
## attribute data
Y = Data_set[:,17]
## class data

## allows structure of deep learning to be sequentially built, using model.add()..
model = Sequential()
model.add(Dense(30, input_dim=17, activation="relu"))
model.add(Dense(1,activation='sigmoid'))
## for this example, add function is used twice, to add two layers to this model

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y,epochs=100, batch_size=10)

