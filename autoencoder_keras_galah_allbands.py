import imp, os, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'

import matplotlib
matplotlib.use('Agg')

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.layers.advanced_activations import PReLU
import numpy as np
from astropy.table import Table
from sklearn.externals import joblib
import matplotlib.pyplot as plt

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters, read_pkl_spectra

from keras import backend as K
import tensorflow as tf
tf_config = tf.ConfigProto(intra_op_parallelism_threads=144,
                           inter_op_parallelism_threads=144,
                           allow_soft_placement=True)
session = tf.Session(config=tf_config)
K.set_session(session)

# get arguments if they were added to the procedure
in_args = sys.argv
n_l_e = 30
n_epoch = 70
if len(in_args) >= 2:
    n_l_e = int(in_args[1])
if len(in_args) >= 3:
    n_epoch = int(in_args[2])

print 'Reading data sets'
galah_data_input = '/shared/ebla/cotar/'
date_string = '20190801'
line_file = 'GALAH_Cannon_linelist_newer.csv'
galah_param_file = 'sobject_iraf_53_reduced_'+date_string+'.fits'
galah_param = Table.read(galah_data_input + galah_param_file)
galah_tsne_flag = Table.read(galah_data_input + 'tsne_class_1_0.csv')
galah_tsne_flag = galah_tsne_flag.filled()
galah_tsne_flag = galah_tsne_flag[galah_tsne_flag['published_reduced_class_proj1'] != 'N/A']
galah_tsne_flag = galah_tsne_flag[galah_tsne_flag['published_reduced_class_proj1'] != 'problematic']

idx_train_spectra = np.logical_and(galah_param['snr_c2_guess'] > 25, galah_param['red_flag'] == 0)
# idx_get_spectra = np.logical_and(np.logical_and(galah_param['teff_guess'] > 5800, galah_param['teff_guess'] < 6000),
#                                  np.logical_and(galah_param['logg_guess'] > 3.2, galah_param['logg_guess'] < 3.4))

min_wvl = np.array([4725, 5665, 6485, 7700])
max_wvl = np.array([4895, 5865, 6725, 7875])

spectra_file_list = ['galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd2_5640_5880_wvlstep_0.050_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd4_7700_7895_wvlstep_0.070_ext4_'+date_string+'.pkl']

read_ccds = [1, 2, 3, 4]
suffix = '_{:.0f}D_4layer'.format(n_l_e)
out_dir = galah_data_input+'Autoencoder_dense_test_complex_ccd' + ''.join([str(c) for c in read_ccds]) + '_relu'+suffix
os.system('mkdir '+out_dir)
os.chdir(out_dir)

spectral_data_all = list([])
wvl_values_all = list([])
for read_ccd in read_ccds:
    print 'Reading spectra for ccd:', read_ccd
    # parse resampling settings from filename
    read_pkl_file = galah_data_input + spectra_file_list[read_ccd-1]
    csv_param = CollectionParameters(read_pkl_file)
    wvl_values = csv_param.get_wvl_values()
    wvl_limits = csv_param.get_wvl_range()
    ccd_number = int(csv_param.get_ccd())

    idx_read = np.where(np.logical_and(wvl_values >= min_wvl[read_ccd-1], wvl_values <= max_wvl[read_ccd-1]))[0]
    # print idx_read
    wvl_read = wvl_values[idx_read]
    wvl_values_all.append(wvl_read)
    # read data from pickle file
    spectral_data_all.append(read_pkl_spectra(read_pkl_file, read_cols=idx_read))

spectral_data_all = np.hstack(spectral_data_all)
wvl_values_all = np.hstack(wvl_values_all)

n_obj = np.sum(idx_train_spectra)
n_wvl = spectral_data_all.shape[1]
print n_wvl, n_obj

idx_bad = np.logical_not(np.isfinite(spectral_data_all))
if np.sum(idx_bad) > 0:
    print 'Correcting bad values:', np.sum(idx_bad)
    spectral_data_all[idx_bad] = 1.

idx_bad = spectral_data_all > 1.2
if np.sum(idx_bad) > 0:
    print 'spectral_data_all large/low values:', np.sum(idx_bad)
    spectral_data_all[idx_bad] = 1.2

idx_bad = spectral_data_all < 0
if np.sum(idx_bad) > 0:
    print 'spectral_data_all large/low values:', np.sum(idx_bad)
    spectral_data_all[idx_bad] = 0

# normalize data
spectral_data_all = 1. - spectral_data_all

spectral_data = spectral_data_all[idx_train_spectra, :]
print spectral_data.shape
print spectral_data_all.shape

activation = 'relu' # PReLU if set to None
dropout_rate = 0.1
decoded_layer_name = 'encoded'
n_wvl = spectral_data.shape[1]

# compute number of nodes in every connected layer
n_l_1 = int(n_wvl * 0.60)
n_l_2 = int(n_wvl * 0.30)
n_l_3 = 0  # int(n_wvl * 0.40)
n_l_4 = 0  # int(n_wvl * 0.20)
n_l_5 = int(n_wvl * 0.15)
n_l_6 = int(n_wvl * 0.07)

# create ann model
autoencoder = Sequential()

if n_l_1 > 0:
    autoencoder.add(Dense(n_l_1, input_shape=(n_wvl,), activation=activation, name='E_1'))
    autoencoder.add(Dropout(dropout_rate, name='DO_1'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_1'))

if n_l_2 > 0:
    autoencoder.add(Dense(n_l_2, activation=activation, name='E_2'))
    autoencoder.add(Dropout(dropout_rate, name='DO_2'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_2'))

if n_l_3 > 0:
    autoencoder.add(Dense(n_l_3, activation=activation, name='E_3'))
    autoencoder.add(Dropout(dropout_rate, name='DO_3'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_3'))

if n_l_4 > 0:
    autoencoder.add(Dense(n_l_4, activation=activation, name='E_4'))
    autoencoder.add(Dropout(dropout_rate, name='DO_4'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_4'))

if n_l_5 > 0:
    autoencoder.add(Dense(n_l_5, activation=activation, name='E_5'))
    autoencoder.add(Dropout(dropout_rate, name='DO_5'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_5'))

if n_l_6 > 0:
    autoencoder.add(Dense(n_l_6, activation=activation, name='E_6'))
    autoencoder.add(Dropout(dropout_rate, name='DO_6'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_6'))

autoencoder.add(Dense(n_l_e, activation=activation, name=decoded_layer_name))
if activation is None:
    autoencoder.add(PReLU(name='PR_7'))

if n_l_6 > 0:
    autoencoder.add(Dense(n_l_6, activation=activation, name='D_1'))
    autoencoder.add(Dropout(dropout_rate, name='DO_8'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_8'))

if n_l_5 > 0:
    autoencoder.add(Dense(n_l_5, activation=activation, name='D_2'))
    autoencoder.add(Dropout(dropout_rate, name='DO_9'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_9'))

if n_l_4 > 0:
    autoencoder.add(Dense(n_l_4, activation=activation, name='D_3'))
    autoencoder.add(Dropout(dropout_rate, name='DO_10'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_10'))

if n_l_3 > 0:
    autoencoder.add(Dense(n_l_3, activation=activation, name='D_4'))
    autoencoder.add(Dropout(dropout_rate, name='DO_11'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_11'))

if n_l_2 > 0:
    autoencoder.add(Dense(n_l_2, activation=activation, name='D_5'))
    autoencoder.add(Dropout(dropout_rate, name='DO_12'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_12'))

if n_l_1 > 0:
    autoencoder.add(Dense(n_l_1, activation=activation, name='D_6'))
    autoencoder.add(Dropout(dropout_rate, name='DO_13'))
    if activation is None:
        autoencoder.add(PReLU(name='PR_13'))

autoencoder.add(Dense(n_wvl, activation='linear', name='recreated'))
autoencoder.summary()

# model file handling
out_model_file = 'model_weights.h5'
if os.path.isfile(out_model_file):
    autoencoder.load_weights(out_model_file, by_name=True)
else:
    autoencoder.compile(optimizer='adam', loss='mse')
    ann_fit_hist = autoencoder.fit(spectral_data, spectral_data,
                                   epochs=n_epoch,
                                   shuffle=True,
                                   batch_size=16384,
                                   validation_split=0.10,
                                   verbose=2)
    autoencoder.save_weights(out_model_file)

    plt.plot(ann_fit_hist.history['loss'], label='Train')
    plt.plot(ann_fit_hist.history['val_loss'], label='Validation')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.ylim(0, 0.005)
    plt.tight_layout()
    plt.legend()
    plt.savefig('ann_network_loss.png', dpi=250)
    plt.close()

print 'Predicting values'
processed_data = autoencoder.predict(spectral_data, verbose=2, batch_size=2048)

# denormalize data
print processed_data
print processed_data.shape

os.system('mkdir random')
os.chdir('random')
print ('Plotting random spectra')
for i_r in np.random.random(350)*n_obj:
    i_o = int(np.floor(i_r))
    plt.figure(figsize=(50, 5))
    plt.plot(1. - spectral_data[i_o, :], color='black', lw=0.8)
    plt.plot(1. - processed_data[i_o, :], color='blue', lw=0.8)
    plt.xlim(0, n_wvl)
    plt.ylim(0.3, 1.1)
    plt.tight_layout()
    plt.savefig('s_{:06.0f}.png'.format(i_r), dpi=250)
    plt.close()
os.chdir('..')

os.system('mkdir tsne_flag')
os.chdir('tsne_flag')
print ('Plotting tsne flaged spectra')
for s_id in np.random.choice(galah_tsne_flag['sobject_id'], 300, replace=False):
    i_o = np.where(galah_param['sobject_id'] == s_id)[0]
    if len(i_o) != 1:
        continue
    else:
        i_o = i_o[0]
    plt.figure(figsize=(50, 5))
    plt.plot(1. - spectral_data[i_o, :], color='black', lw=0.8)
    plt.plot(1. - processed_data[i_o, :], color='blue', lw=0.8)
    plt.xlim(0, n_wvl)
    plt.ylim(0.3, 1.1)
    plt.tight_layout()
    plt.savefig('s_{:06.0f}.png'.format(i_o), dpi=250)
    plt.close()
os.chdir('..')

os.system('mkdir large_dif')
os.chdir('large_dif')
print ('Plotting strange spectra')
spectrum_diff = np.nansum(spectral_data - processed_data, axis=1)
for i_o in np.argsort(spectrum_diff)[::-1][:300]:
    plt.figure(figsize=(50, 5))
    plt.plot(1. - spectral_data[i_o, :], color='black', lw=0.8)
    plt.plot(1. - processed_data[i_o, :], color='blue', lw=0.8)
    plt.xlim(0, n_wvl)
    plt.ylim(0.3, 1.1)
    plt.tight_layout()
    plt.savefig('s_{:06.0f}.png'.format(i_o), dpi=250)
    plt.close()
os.chdir('..')

# model that will output decode values
print 'Predicting encoded values'
autoencoder_encoded = Model(inputs=autoencoder.input,
                            outputs=autoencoder.get_layer(decoded_layer_name).output)
autoencoder_encoded.summary()

plt.plot(autoencoder.predict(np.full((1, n_wvl), 0.), verbose=2)[0])
plt.plot(autoencoder.predict(np.full((1, n_wvl), 0.1), verbose=2)[0])
plt.plot(autoencoder.predict(np.full((1, n_wvl), -0.1), verbose=2)[0])
plt.show()
plt.close()
# raise SystemExit

print autoencoder.get_layer(decoded_layer_name).get_weights()
print autoencoder_encoded.get_layer(decoded_layer_name).get_weights()

print ' -> Getting predictions from encoded layer'
decoded_spectra = autoencoder_encoded.predict(spectral_data, verbose=2)
decoded_spectra_all = autoencoder_encoded.predict(spectral_data_all, verbose=2)
print ' -> Saving reduced and encoded spectra'
out_file = 'encoded_spectra_ccd{:01.0f}_nf{:.0f}'.format(read_ccd+1, n_l_e)
joblib.dump(decoded_spectra_all, out_file+'.pkl')
# export as csv
csv = open(out_file+'.csv', 'w')
for i_l in range(len(galah_param)):
    line = str(galah_param[i_l]['sobject_id']) + ','
    line += ','.join([str(f) for f in decoded_spectra_all[i_l, :]])
    csv.write(line+'\n')
csv.close()

for i_f in range(decoded_spectra.shape[1]):
    title = 'Zeros: {:.1f}%'.format(100.*np.sum(decoded_spectra[:, i_f] == 0.)/decoded_spectra.shape[0])
    hist_range = (np.nanpercentile(decoded_spectra[:, i_f], 1), np.nanpercentile(decoded_spectra[:, i_f], 99))
    plt.hist(decoded_spectra[:, i_f], range=hist_range, bins=75)
    plt.xlim(hist_range)
    plt.title(title)
    plt.savefig('feature_{:02.0f}_train.png'.format(i_f), dpi=200)
    plt.close()

for i_f in range(decoded_spectra_all.shape[1]):
    title = 'Zeros: {:.1f}%'.format(100.*np.sum(decoded_spectra_all[:, i_f] == 0.)/decoded_spectra_all.shape[0])
    hist_range = (np.nanpercentile(decoded_spectra_all[:, i_f], 1), np.nanpercentile(decoded_spectra_all[:, i_f], 99))
    plt.hist(decoded_spectra_all[:, i_f], range=hist_range, bins=75)
    plt.xlim(hist_range)
    plt.title(title)
    plt.savefig('feature_{:02.0f}_all.png'.format(i_f), dpi=200)
    plt.close()

