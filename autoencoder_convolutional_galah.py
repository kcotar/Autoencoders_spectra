import imp, os
import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1
from astropy.table import Table
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_input = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_input+'sobject_iraf_52_reduced.fits')
line_list = Table.read(galah_data_input+'GALAH_Cannon_linelist.csv')
spectra_file = 'galah_dr52_ccd3_6475_6745_wvlstep_0.03_lin_RF_renorm.csv'

# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file)
wvl_values = csv_param.get_wvl_values()
wvl_limits = csv_param.get_wvl_range()
ccd_number = int(csv_param.get_ccd())

idx_read = np.where(np.logical_and(wvl_values > 6570,
                                   wvl_values < 6590))

# print idx_read
wvl_read = wvl_values[idx_read]
n_wvl = len(wvl_read)

idx_get_spectra = np.logical_and(np.logical_and(galah_param['teff_guess'] > 5800, galah_param['teff_guess'] < 6000),
                                 np.logical_and(galah_param['logg_guess'] > 3.2, galah_param['logg_guess'] < 3.4))

print n_wvl, np.sum(idx_get_spectra)

autoencoder = Sequential()
autoencoder.add(Conv1D(16, 5, padding='same', input_shape=(600, 1), activation='relu'))
autoencoder.add(MaxPooling1D(2, padding='same'))
autoencoder.add(Conv1D(8, 5, padding='same', activation='relu'))
autoencoder.add(MaxPooling1D(2, padding='same'))
autoencoder.add(Conv1D(8, 5, padding='same', activation='relu'))
autoencoder.add(MaxPooling1D(2, padding='same'))

autoencoder.add(Conv1D(8, 5, padding='same', activation='relu'))
autoencoder.add(UpSampling1D(2))
autoencoder.add(Conv1D(8, 5, padding='same', activation='relu'))
autoencoder.add(UpSampling1D(2))
autoencoder.add(Conv1D(16, 5, padding='same', activation='relu'))
autoencoder.add(UpSampling1D(2))
autoencoder.add(Conv1D(1, 5, padding='same', activation='sigmoid'))

autoencoder.summary()

spectral_data = pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan',
                            usecols=idx_read[0],
                            skiprows=np.where(np.logical_not(idx_get_spectra))[0]).values

# add additional layer to data
spectral_data = np.expand_dims(spectral_data[:, 0:600], axis=2)
spectra_shape = spectral_data.shape
print 'Shape: ', spectra_shape

input_layer = Input(shape=(spectra_shape[1], 1))

# create ann model
# encoder part
# autoencoder.add(Conv1D(16, 3, padding='same', input_shape=(spectra_shape[1], 1)))
# autoencoder.add(LeakyReLU(alpha=.1))
# autoencoder.add(MaxPooling1D(5, padding='same'))
# autoencoder.add(Conv1D(8, 5, padding='same'))
# autoencoder.add(LeakyReLU(alpha=.1))
# autoencoder.add(MaxPooling1D(5, padding='same'))
# # decoder part
#
# # TODO: split both parts in order to get encoded results of the analysis
# autoencoder.add(Conv1D(8, 5, padding='same'))
# autoencoder.add(LeakyReLU(alpha=.1))
# autoencoder.add(UpSampling1D(5))
# autoencoder.add(Conv1D(16, 5, padding='same'))
# autoencoder.add(LeakyReLU(alpha=.1))
# autoencoder.add(UpSampling1D(5))
# autoencoder.add(Conv1D(1, 5, padding='same', activation='sigmoid'))


layer_dict = dict([(layer.name, layer) for layer in autoencoder.layers])

# compile and train the model
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(spectral_data, spectral_data,
                epochs=100,
                batch_size=128,
                shuffle=True)

processed_data = autoencoder.predict(spectral_data)

for i_r in np.random.random(100)*spectra_shape[0]:
    plt.plot(spectral_data[i_r], color='black')
    plt.plot(processed_data[i_r], color='blue', alpha=1)
    plt.show()
