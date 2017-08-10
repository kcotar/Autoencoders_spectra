import os

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# input data
galah_data_input = '/home/klemen/GALAH_data/'
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF.csv',
                     'galah_dr52_ccd3_5640_5880_wvlstep_0.05_lin_RF.csv',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF.csv',
                     'galah_dr52_ccd3_7700_7895_wvlstep_0.07_lin_RF.csv']

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------

# algorithm settings
save_models = True
output_results = True
output_plots = True

# reading settings
spectra_get_cols = [4000, 4000, 4000, 2000]

# AE NN band dependant settings
n_dense_first = [500, 500, 500, 250]  # number of nodes in first and third fully connected layer of AE
n_dense_middle = [40, 40, 40, 40, 20]  # number of nodes in the middle fully connected layer of AE

# configuration of CAE network is the same for every spectral band:
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

# --------------------------------------------------------
# ---------------- MAIN PROGRAM --------------------------
# --------------------------------------------------------

for i_band in range(4):

    spectra_file = spectra_file_list[i_band]
    # data availability check
    if not os.path.isfile(galah_data_input + spectra_file):
        print 'Spectral file not found: '+spectra_file
        continue

    # determine cols to be read from csv file
    spectra_file_split = spectra_file.split('_')
    wlv_begin = float(spectra_file_split[3])
    wlv_end = float(spectra_file_split[4])
    wlv_step = float(spectra_file_split[6])
    wvl_range = np.arange(wlv_begin, wlv_end, wlv_step)
    n_wvl = len(wvl_range)
    # select the middle portion of the spectra, that should be free of nan values
    col_start = int(n_wvl/2. - spectra_get_cols[i_band]/2.)
    col_end = int(col_start + spectra_get_cols[i_band])

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectral data from {:04.2f} to {:04.2f}'.format(wvl_range[col_start], wvl_range[col_end])
    spectral_data = pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan',
                                usecols=range(col_start, col_end)).values

    # possible nan data handling
    idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
    n_bad_spectra = len(idx_bad_spectra[0])
    if n_bad_spectra > 0:
        print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
        spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

    # output data names
    normalizer_file = spectra_file[:-4] + '_normalizer.pkl'
    convolution_file = spectra_file[:-4] + '_cae.h5'  # CAE - Convolutional autoencoder
    autoencoder_file = spectra_file[:-4] + '_ae.h5'  # AE - Autoencoder
    encoded_output_file = spectra_file[:-4] + '_encoded.csv'  # AE - Autoencoder

    # normalize data (flux at every wavelength)
    if os.path.isfile(normalizer_file):
        print 'Reading normalization parameters'
        normalizer = joblib.load(normalizer_file)
    else:
        normalizer = StandardScaler()
        normalizer.fit(spectral_data)
        if save_models:
            print 'Saving normalization parameters'
            joblib.dump(normalizer, normalizer_file)
    spectral_data_norm = normalizer.transform(spectral_data)

    # --------------------------------------------------------
    # ---------------- Convolutional autoencoder -------------
    # --------------------------------------------------------

    # add additional dimension that is needed by Keras Input layer
    X_in = np.expand_dims(spectral_data_norm, axis=2)

    # create Keras Input
    input_cae = Input(shape=(X_in.shape[1], 1))

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
        print 'Reading NN weighs - CAE'
        convolutional_nn.load_weights(convolution_file)
    else:
        convolutional_nn.fit(X_in, X_in,
                             epochs=125,
                             batch_size=256,
                             shuffle=True,
                             validation_split=0.1)
        if save_models:
            print 'Saving NN weighs - CAE'
            convolutional_nn.save_weights(convolution_file)

    print 'Predicting encoded and decoded layers - CAE'
    X_out_cae = convolutional_nn.predict(X_in)
    X_out_encoded = convolutional_encoder.predict(X_in)
    X_out_encoded_shape = X_out_encoded.shape

    # prepare data for the second processing step
    aa_vector_length = X_out_encoded_shape[1]*X_out_encoded_shape[2]
    X_in_2 = X_out_encoded.reshape((X_out_encoded_shape[0], aa_vector_length))

    # --------------------------------------------------------
    # ---------------- Middle fully connected autoencoder ----
    # --------------------------------------------------------

    # second nn network - fully connected layers
    input_ae = Input(shape=(aa_vector_length,))

    # fully connected layer in between the encoder and decoder part of the CAE
    x = Dense(n_dense_first[i_band], activation=None)(input_ae)
    x = PReLU()(x)
    x = Dense(n_dense_middle[i_band], activation=None)(x)
    encoded_ae = PReLU()(x)
    x = Dense(n_dense_first[i_band], activation=None)(encoded_ae)
    x = PReLU()(x)
    x = Dense(aa_vector_length, activation=None)(x)
    decoded_ae = PReLU()(x)

    autoencoder_nn = Model(input_ae, decoded_ae)
    autoencoder_nn.compile(optimizer='adadelta', loss='mse')
    autoencoder_nn.summary()

    # create a model for the encoder part of the network
    autoencoder_encoder = Model(input_ae, encoded_ae)

    # model file handling
    if os.path.isfile(autoencoder_file):
        print 'Reading NN weighs- AE'
        autoencoder_nn.load_weights(autoencoder_file)
    else:
        autoencoder_nn.fit(X_in_2, X_in_2,
                           epochs=250,
                           batch_size=256,
                           shuffle=True,
                           validation_split=0.1)
        if save_models:
            print 'Saving NN weighs - AE'
            autoencoder_nn.save_weights(autoencoder_file)

    print 'Predicting encoded and decoded layers - AE'
    X_out_2 = autoencoder_nn.predict(X_in_2)
    X_out_encoded_2 = autoencoder_encoder.predict(X_in_2)  # IMPORTANT: final output of our NN spectral reduction
    X_out_encoded_2_shape = X_out_encoded_2.shape

    # --------------------------------------------------------
    # ---------------- Check results - data deconvolution ----
    # --------------------------------------------------------

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

    if output_results:
        print 'Saving reduced and encoded spectra'
        np.savetxt(encoded_output_file, X_out_encoded_2, delimiter=',')  #, fmt='%f')

    # denormalize data from both outputs (full and partial network)
    processed_data = normalizer.inverse_transform(np.squeeze(X_out_cae, axis=2))  # partial NN network
    processed_data_2 = normalizer.inverse_transform(np.squeeze(X_out_cae_2, axis=2))  # full NN network

    # --------------------------------------------------------
    # ---------------- Show or plot result of the analysis ---
    # --------------------------------------------------------

    if output_plots:
        print 'Plotting resulting spectra'
        n_random_plots = 50
        id_plots = np.int32(np.random.rand(n_random_plots)*processed_data.shape[0])
        for i in id_plots:
            plt.plot(spectral_data[i], color='black', lw=0.75)
            plt.plot(processed_data[i], color='red', lw=0.75)
            plt.plot(processed_data_2[i], color='blue', lw=0.75)
            plt.ylim((0.4, 1.2))
            plt.savefig(str(i)+'.png', dpi=750)
            plt.close()

    # --------------------------------------------------------
    # ---------------- Analyse decoded values and their distribution
    # --------------------------------------------------------
    # TODO

