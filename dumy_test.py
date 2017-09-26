from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.advanced_activations import PReLU

import keras.backend as K
import tensorflow as T
import numpy as np
import pandas as pd


def mean_squared_error(y_true, y_pred):
    idx = T.is_finite(y_true)
    return K.mean(K.square(T.boolean_mask(y_pred, idx) - T.boolean_mask(y_true, idx)), axis=-1)

# def error

ann_input = Input(shape=(10,), name='Input')
ann = Dense(25, name='Dense_1')(ann_input)
ann = PReLU(name='PReLU_1')(ann)
ann = Dense(3, name='Dense_2')(ann)
ann = PReLU(name='PReLU_2')(ann)

abundance_ann = Model(ann_input, ann)
abundance_ann.compile(optimizer='adam', loss=mean_squared_error, metrics=['accuracy'])
abundance_ann.summary()

x = np.random.rand(50, 10)
y = np.random.rand(50, 3)
y[10:20,1] = np.nan
y[30:40,2] = np.nan
y[0:8,0] = np.nan
print y
abundance_ann.fit(x, y, epochs=125)