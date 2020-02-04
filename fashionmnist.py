import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
'Shirt', 'Snearker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

##print(train_images[7])
##plt.imshow(train_ismages[7], cmap=plt.cm.binary)
##plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128,activation="relu"), 
	keras.layers.Dense(10, activation="softmax"), ## 10 neurons with activation function of soft max ~> pick values for each neurons that all 
	## neurons add up to 1 => almost like a probability function 
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

## training 
model.fit(train_images, train_labels, epochs=5) ## how many times the model will see the train images

#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested Acc : ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction " + class_names[np.argmax(prediction[i])])
	plt.show()

