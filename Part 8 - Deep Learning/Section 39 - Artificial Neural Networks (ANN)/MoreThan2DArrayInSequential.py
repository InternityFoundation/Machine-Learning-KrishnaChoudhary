import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')


from keras.models import Sequential
from keras.layers import Activation,Dense
from keras.layers import Flatten
model = Sequential([
    Dense(12, input_shape=(32,32,3)),
    Flatten(),
    Activation('relu'),
    Dense(output_dim=10)
])
model.add(Dense(12,'relu', input_shape=(32,32,3),Flatten()))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size=10)