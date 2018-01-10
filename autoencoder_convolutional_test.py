import imp, os
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Dropout, AveragePooling1D, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.callbacks import EarlyStopping
from astropy.table import Table
from socket import gethostname
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PC hostname
pc_name = gethostname()

# input data
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
    tsne_path = '/home/klemen/tSNE_test/'
    galah_data_input = '/home/klemen/data4_mount/'
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
    imp.load_source('tsne_functions', '../tSNE_test/tsne_functions.py')
    imp.load_source('spectra_collection_functions', '../Carbon-Spectra/spectra_collection_functions.py')
else:
    tsne_path = '/data4/cotar/'
    galah_data_input = '/data4/cotar/'
from helper_functions import move_to_dir
from tsne_functions import *
from spectra_collection_functions import read_pkl_spectra, save_pkl_spectra

galah_param_file = 'sobject_iraf_52_reduced_20171111.fits'
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_20171111.pkl',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_20171111.pkl',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_20171111.pkl',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_20171111.pkl']

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------

# algorithm settings
save_models = True
output_results = True
output_plots = True
limited_rows = False
snr_cut = True
run_tsne_test = False
train_phase = True
train_subset = 0.05

global_normalization = False
zero_mean_only = False
use_dropout = True
droupout_rate = 0.1

# reading settings
spectra_get_cols = [3200, 3200, 3200, 2016]

# AE NN band dependant settings
n_dense_first = [800, 800, 800, 800]  # number of nodes in first and third fully connected layer of AE
n_dense_middle = [50, 50, 50, 50]  # number of nodes in the middle fully connected layer of AE

# configuration of CAE network is the same for every spectral band:
# convolution layer 1
C_f_1 = 256  # number of filters
C_k_1 = 9  # size of convolution kernel
C_s_1 = 1  # strides value
P_s_1 = 8  # size of pooling operator
# convolution layer 2
C_f_2 = 256
C_k_2 = 7
C_s_2 = 1
P_s_2 = 4
# convolution layer 3
C_f_3 = 128
C_k_3 = 5
C_s_3 = 1
P_s_3 = 4

# --------------------------------------------------------
# ---------------- MAIN PROGRAM --------------------------
# --------------------------------------------------------

galah_param = Table.read(galah_data_input + galah_param_file)

for i_band in [2]:

    spectra_file = spectra_file_list[i_band]
    # data availability check
    if not os.path.isfile(galah_data_input + spectra_file):
        print 'Spectral file not found: '+spectra_file
        continue

    print 'Working on '+spectra_file
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

    suffix = ''
    if snr_cut and not limited_rows:
        snr_percentile = 5.
        snr_col = 'snr_c'+str(i_band+1)+'_iraf'  # as ccd numbering starts with 1
        print 'Cutting off {:.1f}% of spectra with low snr value ('.format(snr_percentile)+snr_col+').'
        snr_percentile_value = np.percentile(galah_param[snr_col], snr_percentile)
        use_rows = np.where(galah_param[snr_col] > snr_percentile_value)[0]
        suffix += '_snrcut'
    elif limited_rows:
        n_first_lines = 15000
        print 'Only limited number ('+str(n_first_lines)+') of spectra rows will be read'
        use_rows = np.arange(n_first_lines, len(galah_param))
        suffix += '_subset'
    else:
        use_rows = None

    # add some additional suffix describing processing parameters
    if global_normalization:
        suffix += '_globalnorm'
    if zero_mean_only:
        suffix += '_zeromean'

    if use_rows is not None and train_phase:
        n_use_rows = len(use_rows)
        use_random = np.int64(np.random.rand(np.int64(train_subset * n_use_rows)) * n_use_rows)
        use_rows = use_rows[np.unique(use_random)]
        print 'Rows used for training purpose:', str(len(use_rows))

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectral data from {:07.2f} to {:07.2f}'.format(wvl_range[col_start], wvl_range[col_end])
    spectral_data = read_pkl_spectra(galah_data_input + spectra_file,
                                     read_cols=range(col_start, col_end), read_rows=use_rows)

    # possible nan data handling
    idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
    n_bad_spectra = len(idx_bad_spectra[0])
    if n_bad_spectra > 0:
        print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
        spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value
    spectral_data[spectral_data > 2.] = 2.

    # run t-SNE projection
    if run_tsne_test:
        print 'Running tSNE on input spectra'
        perp = 40
        theta = 0.4
        seed = 1337
        tsne_result = bh_tsne(spectral_data, no_dims=2, perplexity=perp, theta=theta, randseed=seed, verbose=True,
                              distance='euclidean', path=tsne_path)

    # create suffix for both network parts
    cae_suffix = '_CAE_'+str(C_f_1)+'_'+str(C_k_1)+'_'+str(C_s_1)+'_'+str(P_s_1)+\
                 '_'+str(C_f_2)+'_'+str(C_k_2)+'_'+str(C_s_2)+'_'+str(P_s_2)+\
                 '_'+str(C_f_3)+'_'+str(C_k_3)+'_'+str(C_s_3)+'_'+str(P_s_3)
    ae_suffix = '_AE_'+str(n_dense_first[i_band])+'_'+str(n_dense_middle[i_band])
    # output data names
    normalizer_file = spectra_file[:-4] + '_' + str(spectra_get_cols[i_band]) + suffix + '_normalizer.pkl'
    normalizer_file_global = spectra_file[:-4] + '_' + str(spectra_get_cols[i_band]) + suffix + '_normalizer_global.pkl'
    convolution_file = spectra_file[:-4] + cae_suffix + suffix + '_cae.h5'  # CAE - Convolutional autoencoder
    autoencoder_file = spectra_file[:-4] + cae_suffix + ae_suffix + suffix + '_ae.h5'  # AE - Autoencoder
    encoded_output_file = spectra_file[:-4] + cae_suffix + ae_suffix + suffix + '_encoded.csv'  # AE - Autoencoder

    print 'CAE: ' + cae_suffix
    print 'AE:  ' + ae_suffix

    # normalize data (flux at every wavelength)
    if global_normalization:
        if os.path.isfile(normalizer_file_global):
            print 'Reading normalization parameters'
            global_norm_param = joblib.load(normalizer_file_global)
        else:
            global_norm_param = [np.mean(spectral_data), np.std(spectral_data)]
            if save_models:
                print 'Saving normalization parameters'
                joblib.dump(global_norm_param, normalizer_file_global)
        spectral_data_norm = spectral_data - global_norm_param[0]
        if not zero_mean_only:
            spectral_data_norm /= global_norm_param[1]
    else:
        if os.path.isfile(normalizer_file):
            print 'Reading normalization parameters'
            normalizer = joblib.load(normalizer_file)
        else:
            if zero_mean_only:
                normalizer = StandardScaler(with_mean=True, with_std=False)
            else:
                normalizer = StandardScaler(with_mean=True, with_std=True)
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
    spectral_data_norm = None

    # create Keras Input
    input_cae = Input(shape=(X_in.shape[1], 1))

    x = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_1', strides=C_s_1)(input_cae)
    x = PReLU(name='R_1')(x)
    x = MaxPooling1D(P_s_1, padding='same', name='P_1')(x)
    if C_f_2 > 0:
        x = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_2', strides=C_s_2)(x)
        x = PReLU(name='R_2')(x)
        x = MaxPooling1D(P_s_2, padding='same', name='P_2')(x)
    x = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_3', strides=C_s_3)(x)
    x = PReLU(name='R_3')(x)
    encoded_cae = MaxPooling1D(P_s_3, padding='same', name='P_3')(x)

    x = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_4', dilation_rate=C_s_3)(encoded_cae)
    x = PReLU(name='R_4')(x)
    x = UpSampling1D(P_s_3, name='S_4')(x)
    if C_f_2 > 0:
        x = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_5', dilation_rate=C_s_2)(x)
        x = PReLU(name='R_5')(x)
        x = UpSampling1D(P_s_2, name='S_5')(x)
    x = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_6', dilation_rate=C_s_1)(x)
    x = PReLU(name='R_6')(x)
    x = UpSampling1D(P_s_1, name='S_6')(x)
    x = Conv1D(1, C_k_1, activation=None, padding='same', name='C_7')(x)
    decoded_cae = Activation('sigmoid')(x)
    # decoded_cae = PReLU()(x)

    # create a model for complete network
    convolutional_nn = Model(input_cae, decoded_cae)
    convolutional_nn.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
    convolutional_nn.summary()

    # create a model for the encoder part of the network
    convolutional_encoder = Model(input_cae, encoded_cae)

    # model file handling
    if os.path.isfile(convolution_file):
        print 'Reading NN weighs - CAE'
        convolutional_nn.load_weights(convolution_file, by_name=True)
    else:
        # define early stopping callback
        earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
        # fit the NN model
        convolutional_nn.fit(X_in, X_in,
                             epochs=75,
                             batch_size=256,
                             shuffle=True,
                             callbacks=[earlystop],
                             # TODO: this selects the last part of the data (eq latest observations) for validation
                             # TODO: it should be corrected in future to improve the validation step
                             validation_split=0.1,
                             verbose=1)
        if save_models:
            print 'Saving NN weighs - CAE'
            convolutional_nn.save_weights(convolution_file, overwrite=True)

    print 'Predicting encoded and decoded layers - CAE'
    X_out_cae = convolutional_nn.predict(X_in)
    X_out_encoded = convolutional_encoder.predict(X_in)
    X_out_encoded_shape = X_out_encoded.shape
    X_in = None

    # prepare data for the second processing step
    aa_vector_length = X_out_encoded_shape[1]*X_out_encoded_shape[2]
    X_in_2 = X_out_encoded.reshape((X_out_encoded_shape[0], aa_vector_length))
    X_out_encoded = None

    # --------------------------------------------------------
    # ---------------- Middle fully connected autoencoder ----
    # --------------------------------------------------------

    # second nn network - fully connected layers
    input_ae = Input(shape=(aa_vector_length,))

    # fully connected layer in between the encoder and decoder part of the CAE
    x = Dense(n_dense_first[i_band], activation=None)(input_ae)
    x = PReLU()(x)
    if use_dropout:
        x = Dropout(droupout_rate)(x)
    x = Dense(n_dense_middle[i_band], activation=None)(x)
    encoded_ae = PReLU()(x)
    if use_dropout:
        x = Dropout(droupout_rate)(encoded_ae)
        x = Dense(n_dense_first[i_band], activation=None)(x)
    else:
        x = Dense(n_dense_first[i_band], activation=None)(encoded_ae)
    x = PReLU()(x)
    if use_dropout:
        x = Dropout(droupout_rate)(x)
    x = Dense(aa_vector_length, activation=None)(x)
    decoded_ae = PReLU()(x)

    autoencoder_nn = Model(input_ae, decoded_ae)
    autoencoder_nn.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
    autoencoder_nn.summary()

    # create a model for the encoder part of the network
    autoencoder_encoder = Model(input_ae, encoded_ae)

    # model file handling
    if os.path.isfile(autoencoder_file):
        print 'Reading NN weighs- AE'
        autoencoder_nn.load_weights(autoencoder_file, by_name=True)
    else:
        # define early stopping callback
        earlystop = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
        # fit the NN model
        autoencoder_nn.fit(X_in_2, X_in_2,
                           epochs=150,
                           batch_size=256,
                           shuffle=True,
                           callbacks=[earlystop],
                           validation_split=0.1,  # TODO: same as for CAE part of the network
                           verbose=1)
        if save_models:
            print 'Saving NN weighs - AE'
            autoencoder_nn.save_weights(autoencoder_file, overwrite=True)

    print 'Predicting encoded and decoded layers - AE'
    X_out_2 = autoencoder_nn.predict(X_in_2)
    X_out_encoded_2 = autoencoder_encoder.predict(X_in_2)  # IMPORTANT: final output of our NN spectral reduction
    X_out_encoded_2_shape = X_out_encoded_2.shape
    X_in_2 = None

    # --------------------------------------------------------
    # ---------------- Check results - data deconvolution ----
    # --------------------------------------------------------

    # re-create deconvolution part from the convolutional_nn network
    input_decoder_cae = Input(shape=(X_out_encoded_shape[1], X_out_encoded_shape[2]))
    decoder_cae = input_decoder_cae
    if C_f_2 > 0:
        layer_start = 10
    else:
        layer_start = 7
    for cae_layer in convolutional_nn.layers[layer_start:]:
        decoder_cae = cae_layer(decoder_cae)
    convolutional_decoder = Model(input_decoder_cae, decoder_cae)

    # pass X_out_2 trough the decoder to evaluate the result of dimensionality reduction
    # in fully connected AE layers
    X_out_2 = np.reshape(X_out_2, X_out_encoded_shape)
    X_out_cae_2 = convolutional_decoder.predict(X_out_2)
    X_out_2 = None

    # output of the final results for this NN spectral analysis
    if output_results:
        print 'Saving reduced and encoded spectra'
        np.savetxt(encoded_output_file, X_out_encoded_2, delimiter=',')  #, fmt='%f')

    # denormalize data from both outputs (full and partial network)
    if global_normalization:
        processed_data = np.squeeze(X_out_cae, axis=2)
        processed_data_2 = np.squeeze(X_out_cae_2, axis=2)
        if not zero_mean_only:
            processed_data *= global_norm_param[1]
            processed_data_2 *= global_norm_param[1]
        processed_data += global_norm_param[0]
        processed_data_2 += global_norm_param[0]
    else:
        processed_data = normalizer.inverse_transform(np.squeeze(X_out_cae, axis=2))  # partial NN network
        processed_data_2 = normalizer.inverse_transform(np.squeeze(X_out_cae_2, axis=2))  # full NN network

    # --------------------------------------------------------
    # ---------------- Show or plot result of the analysis ---
    # --------------------------------------------------------

    if output_plots:
        out_plot_dir = spectra_file[:-4] + cae_suffix + ae_suffix + suffix
        move_to_dir(out_plot_dir)
        print 'Plotting results for random spectra'
        n_random_plots = 100
        id_plots = np.int32(np.random.rand(n_random_plots)*processed_data.shape[0])
        for i in id_plots:
            plt.plot(spectral_data[i], color='black', lw=0.75)
            plt.plot(processed_data[i], color='red', lw=0.75)
            plt.plot(processed_data_2[i], color='blue', lw=0.75)
            plt.ylim((0.4, 1.2))
            plt.savefig(spectra_file[:-4] + '_' + str(i) + '.png', dpi=750)
            plt.close()
        os.chdir('..')

    # --------------------------------------------------------
    # ---------------- Clean the data ------------------------
    # --------------------------------------------------------
    X_out_encoded_2 = None
    spectral_data = None
    processed_data = None
    processed_data_2 = None

    # --------------------------------------------------------
    # ---------------- Analyse decoded values and their distributions
    # --------------------------------------------------------
    # TODO

