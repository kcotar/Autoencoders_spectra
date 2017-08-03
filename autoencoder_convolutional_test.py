import os

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# input data
galah_data_input = '/home/klemen/GALAH_data/'
spectra_file = 'galah_dr52_ccd3_6475_6745_wvlstep_0.03_lin_RF_subset.csv'
spectral_data = pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan',
                            usecols=range(500, 6900)).values

# output data
normalizer_file = 'galah_dr52_ccd3_normalizer.pkl'
convolution_file = 'galah_dr52_ccd3_cae.h5'  # CAE - Convolutional autoencoder
autoencoder_file = 'galah_dr52_ccd3_ae.h5'  # AE - Autoencoder


# normalize data
if os.path.isfile(normalizer_file):
    print 'Reading normalization parameters'
    normalizer = joblib.load(normalizer_file)
else:
    normalizer = StandardScaler()
    normalizer.fit(spectral_data)
    print 'Saving normalization parameters'
    joblib.dump(normalizer, normalizer_file)
spectral_data_norm = normalizer.transform(spectral_data)

# add aditional dimension that is needed by Keras
X_in = np.expand_dims(spectral_data_norm, axis=2)

input_cae = Input(shape=(X_in.shape[1], 1))

# convolution layer 1
C_f_1 = 16  # number of filters
C_k_1 = 5  # size of convolution kernel
P_s_1 = 4  # size of pooling operator

# convolution layer 2
C_f_2 = 16
C_k_2 = 5
P_s_2 = 4

# convolution layer 3
C_f_3 = 8
C_k_3 = 3
P_s_3 = 2

x = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_1')(input_cae)
x = PReLU(name='R_1')(x)
x = MaxPooling1D(P_s_1, padding='same', name='P_1')(x)
if C_f_2 > 0:
    x = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_2')(x)
    x = PReLU(name='R_2')(x)
    x = MaxPooling1D(P_s_2, padding='same', name='P_2')(x)
x = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_3')(x)
x = PReLU(name='R_3')(x)
encoded_cae = MaxPooling1D(P_s_3, padding='same', name='P_3')(x)

x = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_4')(encoded_cae)
x = PReLU(name='R_4')(x)
x = UpSampling1D(P_s_3, name='S_4')(x)
if C_f_2 > 0:
    x = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_5')(x)
    x = PReLU(name='R_5')(x)
    x = UpSampling1D(P_s_2, name='S_5')(x)
x = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_6')(x)
x = PReLU(name='R_6')(x)
x = UpSampling1D(P_s_1, name='S_6')(x)
x = Conv1D(1, C_k_1, activation=None, padding='same', name='C_7')(x)
decoded_cae = PReLU()(x)

# create a model for complete network
convolutional_nn = Model(input_cae, decoded_cae)
convolutional_nn.compile(optimizer='adadelta', loss='mse')
convolutional_nn.summary()

# create a model for the encoder part of the network
convolutional_encoder = Model(input_cae, encoded_cae)

# model file handling
if os.path.isfile(convolution_file):
    print 'Reading NN weighs'
    convolutional_nn.load_weights(convolution_file)
else:
    convolutional_nn.fit(X_in, X_in,
                         epochs=75,
                         batch_size=128,
                         shuffle=True,
                         validation_split=0.1)
    print 'Saving NN weighs'
    convolutional_nn.save_weights(convolution_file)

print 'Predicting encoded and decoded layers'
X_out_cae = convolutional_nn.predict(X_in)
X_out_encoded = convolutional_encoder.predict(X_in)
X_out_encoded_shape = X_out_encoded.shape

# prepare data for the second processing step
aa_vector_length = X_out_encoded_shape[1]*X_out_encoded_shape[2]
X_in_2 = X_out_encoded.reshape((X_out_encoded_shape[0], aa_vector_length))

# second nn network - fully connected layers
input_ae = Input(shape=(aa_vector_length,))

# fully connected layer in between the encoder and decoder part of the CAE
x = Dense(750, activation=None)(input_ae)
x = PReLU()(x)
x = Dense(50, activation=None)(x)
x = PReLU()(x)
x = Dense(750, activation=None)(x)
x = PReLU()(x)
x = Dense(aa_vector_length, activation=None)(x)
decoded_ae = PReLU()(x)

autoencoder_nn = Model(input_ae, decoded_ae)
autoencoder_nn.compile(optimizer='adadelta', loss='mse')
autoencoder_nn.summary()

# model file handling
if os.path.isfile(autoencoder_file):
    print 'Reading NN weighs'
    autoencoder_nn.load_weights(autoencoder_file)
else:
    autoencoder_nn.fit(X_in_2, X_in_2,
                       epochs=150,
                       batch_size=128,
                       shuffle=True,
                       validation_split=0.1)
    print 'Saving NN weighs'
    autoencoder_nn.save_weights(autoencoder_file)

X_out_2 = autoencoder_nn.predict(X_in_2)
# TODO: X_out_encoded_2 =  # this is final output of our NN spectral analysis

# create deconvolution part from the convolutional_nn network
input_decoder_cae = Input(shape=(X_out_encoded_shape[1], X_out_encoded_shape[2]))
decoder_cae = input_decoder_cae
for cae_layer in convolutional_nn.layers[10:]:
    decoder_cae = cae_layer(decoder_cae)
convolutional_decoder = Model(input_decoder_cae, decoder_cae)

# pass X_out_2 trough the decoder to evaluate the result of dimensionality reduction
# in fully connected AE layers
X_out_2 = np.reshape(X_out_2, X_out_encoded_shape)
X_out_cae_2 = convolutional_decoder.predict(X_out_2)

# denormalize data
processed_data = normalizer.inverse_transform(np.squeeze(X_out_cae, axis=2))
processed_data_2 = normalizer.inverse_transform(np.squeeze(X_out_cae_2, axis=2))

print 'Plotting'
id_plots = np.int32(np.random.rand(20)*processed_data.shape[0])
for i in id_plots:
    plt.plot(spectral_data[i], color='black', lw=1)
    plt.plot(processed_data[i], color='red', lw=1)
    plt.plot(processed_data_2[i], color='blue', lw=1)
    plt.ylim((0.4, 1.2))
    plt.savefig(str(i)+'.png', dpi=500)
    plt.close()

