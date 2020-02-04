import tensorflow as tf 
from tensorflow import keras
import numpy as np  ## numpy version 1.16.1 needed to use imdb

imdb = keras.datasets.imdb ## movie database

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path="imdb.npz", num_words=10000) 
## only take the words that are top 10000 freqeuntly appearing

## create mapping , but will use tensorflow's given one for this

word_index = imdb.get_word_index()  ## returns tuple (key,value)

word_index = {k:(v+3) for k,v in word_index.items()} # 3 keys for the word
word_index["<PAD>"] = 0 ## padding added to make all the movie reviews same length 
word_index["<START>"] = 1
word_index["<UNK>"] = 2 ## unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) ## let the value point to key instead

## preprocessing data so the length equals to 250 (hence, adding padding to shorter sentences / shortening longer sentences)
## this is needed cause all neurons are same size 
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
	return " ".join([reverse_word_index.get(i,"?") for i in text]) ## if a word cannot be found in the dictionary, it will be decoded with ? 

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,15))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()
print(decode_review(test_data[0]))