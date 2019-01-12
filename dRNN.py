# LSTM for sequence classification in the IMDB dataset
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import LSTM,TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)
timestep = 1
# load the dataset but only keep the top n words, zero the rest
# arr_0=trainx arr_1=train_y arr_2=testx arr3= testy
npzfile = np.load("smaller.npz")
X_train=  npzfile["arr_0"]
y_train = npzfile["arr_1"]
X_test = npzfile["arr_2"]
y_test = npzfile["arr_3"]

# truncate and pad input sequences
max_length = 10000
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
print(X_train.shape)
X_train = X_train.reshape(len(X_train),timestep,max_length)
X_test = X_test.reshape(len(X_test),timestep,max_length)
y_train = y_train.reshape(len(y_train),timestep,1)
y_test = y_test.reshape(len(y_test),timestep,1)
print(X_train.shape)
# create the model
model = Sequential()
model.add(LSTM(1000, activation='tanh', input_shape=(1, max_length), return_sequences=True))
model.add(LSTM(100, activation='tanh', input_shape=(1, max_length), return_sequences=True))
model.add(TimeDistributed(Dense(10, activation='sigmoid')))
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=50, batch_size=100)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))