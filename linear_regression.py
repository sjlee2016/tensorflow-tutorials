import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = [[2,81], [4,93], [6,91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#plt.figure(figsize=(8,5))
#plt.scatter(x,y)
#plt.show()

x_data = np.array(x, dtype='double')
y_data = np.array(y, dtype='double')

a = 0
b = 0

lr = 0.05
## learning rate
epochs = 2001 
## of repetition

for i in range(epochs):
    y_pred = a * x_data + b
    error = y_data - y_pred 
    a_diff = -(1/len(x_data)) * sum(x_data * error, 0) 
    b_diff = -(1/len(x_data)) * sum(y_data  - y_pred , 0) 
    a = a - lr*a_diff 
    b = b - lr*b_diff 

    if i % 100 == 0 : 
        print("epoch=%.0f, 기울기=%.04f, 절편=%.04f" % (i,a,b))
    
y_pred = a * x_data + b 
plt.scatter(x,y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()