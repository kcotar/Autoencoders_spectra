import imp, os
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1
import pandas as pd
import numpy as np
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

idx_read = np.full_like(wvl_values, False)
for line in line_list:
    idx_line = np.logical_and(wvl_values >= line['line_start'],
                              wvl_values <= line['line_end'])
    idx_read = np.logical_or(idx_read, idx_line)
idx_read = np.where(idx_read)
# OR
# idx_read = np.where(np.logical_and(wvl_values > 6570,
#                                    wvl_values < 6590))

# print idx_read
wvl_read = wvl_values[idx_read]
n_wvl = len(wvl_read)

idx_get_spectra = np.logical_and(np.logical_and(galah_param['teff_guess'] > 5800, galah_param['teff_guess'] < 6000),
                                 np.logical_and(galah_param['logg_guess'] > 3.2, galah_param['logg_guess'] < 3.4))

print n_wvl, np.sum(idx_get_spectra)
spectral_data = pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan',
                            usecols=idx_read[0],
                            skiprows=np.where(np.logical_not(idx_get_spectra))[0]).values

# remove cols with any nan data
cols_use = np.isfinite(spectral_data).all(axis=0)
spectral_data = spectral_data[:, cols_use]
wvl_read = wvl_read[cols_use]
n_wvl = len(wvl_read)

# plt.plot(wvl_read, np.min(spectral_data, axis=0), color='black')
# plt.plot(wvl_read, np.max(spectral_data, axis=0), color='blue')
# plt.show()

print spectral_data.shape

# multilayer or deep encoder
# input_img = Input(shape=(n_wvl,))
# encoded = Dense(int(n_wvl/2), activation='sigmoid')(input_img)
# decoded = Dense(int(n_wvl/2), activation='sigmoid')(encoded)
# decoded = Dense(n_wvl, activation='sigmoid')(decoded)
# autoencoder = Model(input_img, decoded)

# normalize data
normalizer = StandardScaler()
normalizer.fit(spectral_data)
spectral_data_norm = normalizer.transform(spectral_data)

# create ann model
autoencoder = Sequential()
autoencoder.add(Dense(int(n_wvl/2), input_shape=(n_wvl,)))
autoencoder.add(LeakyReLU(alpha=.1))
autoencoder.add(Dense(int(n_wvl/5)))
autoencoder.add(LeakyReLU(alpha=.1))
autoencoder.add(Dense(25))
autoencoder.add(LeakyReLU(alpha=.1))
autoencoder.add(Dense(int(n_wvl/5)))
autoencoder.add(LeakyReLU(alpha=.1))
autoencoder.add(Dense(int(n_wvl/2)))
autoencoder.add(LeakyReLU(alpha=.1))
autoencoder.add(Dense(n_wvl))
autoencoder.add(LeakyReLU(alpha=.1))

# model file handling
out_model_file = 'model_weights.h5'
if os.path.isfile(out_model_file):
    autoencoder.load_weights(out_model_file)
else:
    autoencoder.compile(optimizer='sgd', loss='mse')
    autoencoder.fit(spectral_data_norm, spectral_data_norm, epochs=200, shuffle=True, validation_split=0.2)
    autoencoder.save_weights(out_model_file)

processed_data_norm = autoencoder.predict(spectral_data_norm)

# denormalize data
processed_data = normalizer.inverse_transform(processed_data_norm)

for i_r in np.random.random(100)*n_wvl:
    plt.plot(spectral_data[i_r], color='black')
    plt.plot(processed_data[i_r], color='blue', alpha=1)
    # plt.plot(wvl_read, processed_data[i_r]-spectral_data[i_r], color='red')
    plt.show()

