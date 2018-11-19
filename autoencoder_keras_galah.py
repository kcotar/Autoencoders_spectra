import imp, os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l1
import numpy as np
from astropy.table import Table
from sklearn.externals import joblib
import matplotlib.pyplot as plt

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters, read_pkl_spectra

from keras import backend as K
import tensorflow as tf
tf_config = tf.ConfigProto(intra_op_parallelism_threads=72,
                           inter_op_parallelism_threads=72,
                           allow_soft_placement=True)
session = tf.Session(config=tf_config)
K.set_session(session)

print 'Reading data sets'
galah_data_input = '/data4/cotar/'
date_string = '20180327'
line_file = 'GALAH_Cannon_linelist_newer.csv'
galah_param_file = 'sobject_iraf_53_reduced_'+date_string+'.fits'
galah_param = Table.read(galah_data_input + galah_param_file)
galah_tsne_flag = Table.read(galah_data_input + 'tsne_class_1_0.csv')
galah_tsne_flag = galah_tsne_flag.filled()
galah_tsne_flag = galah_tsne_flag[galah_tsne_flag['published_reduced_class_proj1'] != 'N/A']

min_wvl = np.array([4725, 5665, 6485, 7700])
max_wvl = np.array([4895, 5865, 6725, 7875])

# abund_param_file = 'sobject_iraf_cannon2.1.7.fits'
abund_param_file = 'GALAH_iDR3_ts_DR2.fits'  # can have multiple lines with the same sobject_id - this is on purpose
spectra_file_list = ['galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd2_5640_5880_wvlstep_0.050_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd4_7700_7895_wvlstep_0.070_ext4_'+date_string+'.pkl']

for read_ccd in range(len(spectra_file_list)):
    # parse resampling settings from filename
    read_pkl_file = galah_data_input + spectra_file_list[read_ccd]
    csv_param = CollectionParameters(read_pkl_file)
    wvl_values = csv_param.get_wvl_values()
    wvl_limits = csv_param.get_wvl_range()
    ccd_number = int(csv_param.get_ccd())

    idx_read = np.where(np.logical_and(wvl_values >= min_wvl[read_ccd], wvl_values <= max_wvl[read_ccd]))[0]

    out_dir = galah_data_input+'Autoencoder_dense_test_complex_ccd{:01.0f}'.format(read_ccd+1)
    os.system('mkdir '+out_dir)
    os.chdir(out_dir)

    # print idx_read
    wvl_read = wvl_values[idx_read]
    n_wvl = len(wvl_read)

    idx_get_spectra = np.where(galah_param['snr_c2_guess'] > 35)[0]
    # idx_get_spectra = np.logical_and(np.logical_and(galah_param['teff_guess'] > 5800, galah_param['teff_guess'] < 6000),
    #                                  np.logical_and(galah_param['logg_guess'] > 3.2, galah_param['logg_guess'] < 3.4))

    n_obj = len(idx_get_spectra)
    print n_wvl, n_obj
    spectral_data = read_pkl_spectra(read_pkl_file, read_rows=idx_get_spectra, read_cols=idx_read)

    idx_bad = np.logical_not(np.isfinite(spectral_data))
    if np.sum(idx_bad) > 0:
        print 'Correcting bad values:', np.sum(idx_bad)
        spectral_data[idx_bad] = 1.

    idx_bad = spectral_data > 1.5
    if np.sum(idx_bad) > 0:
        print 'Correcting large/low values:', np.sum(idx_bad)
        spectral_data[idx_bad] = 1.5

    print spectral_data.shape
    activation = 'relu'
    dropout_rate = 0.1
    decoded_layer_name = 'encoded'

    # compute number of nodes in every connected layer
    n_l_1 = int(n_wvl * 0.8)
    n_l_2 = int(n_wvl * 0.6)
    n_l_3 = int(n_wvl * 0.4)
    n_l_4 = int(n_wvl * 0.2)
    n_l_5 = int(n_wvl * 0.1)
    n_l_e = 50

    # normalize data
    spectral_data = 1. - spectral_data

    # create ann model
    autoencoder = Sequential()

    autoencoder.add(Dense(n_l_1, input_shape=(n_wvl,), activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_2, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_3, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_4, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_5, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_e, activation=activation, name=decoded_layer_name))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_5, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_4, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_3, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_2, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_l_1, activation=activation))
    autoencoder.add(Dropout(dropout_rate))
    if activation is None:
        autoencoder.add(PReLU())

    autoencoder.add(Dense(n_wvl, activation='linear', name='recreated'))
    autoencoder.summary()

    # model file handling
    out_model_file = 'model_weights.h5'
    if os.path.isfile(out_model_file):
        autoencoder.load_weights(out_model_file)
    else:
        autoencoder.compile(optimizer='adam', loss='mse')
        ann_fit_hist = autoencoder.fit(spectral_data, spectral_data,
                                       epochs=25,
                                       shuffle=True,
                                       batch_size=2048,
                                       validation_split=0.1,
                                       verbose=2)
        autoencoder.save_weights(out_model_file)

        plt.plot(ann_fit_hist.history['loss'], label='Train')
        plt.plot(ann_fit_hist.history['val_loss'], label='Validation')
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.tight_layout()
        plt.legend()
        plt.savefig('ann_network_loss.png', dpi=250)
        plt.close()

    print 'Predicting values'
    processed_data = autoencoder.predict(spectral_data, verbose=1, batch_size=2048)

    # denormalize data
    print processed_data
    print processed_data.shape

    os.system('mkdir random')
    os.chdir('random')
    for i_r in np.random.random(250)*n_obj:
        i_o = int(np.floor(i_r))
        plt.figure(figsize=(20, 5))
        plt.plot(wvl_values[idx_read], 1. - spectral_data[i_o, :], color='black', lw=0.8)
        plt.plot(wvl_values[idx_read], 1. - processed_data[i_o, :], color='blue', lw=0.8)
        plt.xlim(wvl_values[idx_read][0], wvl_values[idx_read][-1])
        plt.ylim(0.3, 1.1)
        plt.tight_layout()
        # plt.plot(wvl_read, processed_data[i_r]-spectral_data[i_r], color='red')
        plt.savefig('s_{:06.0f}.png'.format(i_r), dpi=350)
        plt.close()
    os.chdir('..')

    os.system('mkdir tsne_flag')
    os.chdir('tsne_flag')
    for s_id in np.random.choice(galah_tsne_flag['sobject_id'], 250, replace=False):
        i_o = np.where(galah_param['sobject_id'] == s_id)[0]
        if len(i_o) != 1:
            continue
        else:
            i_o = i_o[0]
        plt.figure(figsize=(20, 5))
        plt.plot(wvl_values[idx_read], 1. - spectral_data[i_o, :], color='black', lw=0.8)
        plt.plot(wvl_values[idx_read], 1. - processed_data[i_o, :], color='blue', lw=0.8)
        plt.xlim(wvl_values[idx_read][0], wvl_values[idx_read][-1])
        plt.ylim(0.3, 1.1)
        plt.tight_layout()
        # plt.plot(wvl_read, processed_data[i_r]-spectral_data[i_r], color='red')
        plt.savefig('s_{:06.0f}.png'.format(i_r), dpi=350)
        plt.close()
    os.chdir('..')

    # model that will output decode values
    print 'Predicting encoded values'
    autoencoder_encoded = Model(inputs=autoencoder.input,
                                outputs=autoencoder.get_layer(decoded_layer_name).output)
    autoencoder_encoded.summary()

    print autoencoder.get_layer(decoded_layer_name).get_weights()
    print autoencoder_encoded.get_layer(decoded_layer_name).get_weights()

    print 'Getting predictions from encoded layer'
    decoded_spectra = autoencoder_encoded.predict(spectral_data)

    print 'Saving reduced and encoded spectra'
    joblib.dump(decoded_spectra, 'encoded_spectra.pkl')

    os.chdir('..')
