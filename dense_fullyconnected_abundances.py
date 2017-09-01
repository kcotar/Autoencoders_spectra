import imp, os

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from astropy.table import Table, join
from socket import gethostname

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PC hostname
pc_name = gethostname()

# input data
if pc_name == 'gigli' or 'klemen-P5K-E':
    galah_data_input = '/home/klemen/GALAH_data/'
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
else:
    galah_data_input = '/data4/cotar/'
from helper_functions import move_to_dir


line_file = 'GALAH_Cannon_linelist.csv'
galah_param_file = 'sobject_iraf_52_reduced.fits'
abund_param_file = 'sobject_iraf_cannon2.1.7.fits'
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF.csv',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF.csv',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF.csv',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF.csv']

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------

# algorithm settings
save_models = True
output_results = True
output_plots = True
limited_rows = False
snr_cut = False
normalize_spectra = True
dropout_learning = True
n_dense_nodes = [3000, 2000, 800, 300, 1]

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------
galah_param = Table.read(galah_data_input + galah_param_file)
line_list = Table.read(galah_data_input + line_file, format='ascii.csv')

spectral_data = list([])
for i_band in [0, 1, 2, 3]:
    spectra_file = spectra_file_list[i_band]
    # determine what is to be read from the spectra
    print 'Defining cols to be read'
    abund_cols_read = list([])
    spectra_file_split = spectra_file.split('_')
    wvl_values = np.arange(float(spectra_file_split[3]), float(spectra_file_split[4]), float(spectra_file_split[6]))
    for line in line_list:
        idx_wvl = np.logical_and(wvl_values >= line['line_start'], wvl_values <= line['line_end'])
        if np.sum(idx_wvl) > 0:
            abund_cols_read.append(np.where(idx_wvl)[0])
    abund_cols_read = np.sort(np.hstack(abund_cols_read))
    print len(abund_cols_read)
    # do the actual reading of spectra
    print 'Reading spectra file: ' + spectra_file
    spectral_data.append(pd.read_csv(galah_data_input + spectra_file, sep=',', header=None,
                                     na_values='nan', skiprows=None, usecols=abund_cols_read).values)

spectral_data = np.hstack(spectral_data)
print spectral_data.shape
n_wvl_total = spectral_data.shape[1]

# somehow handle cols with nan values, delete cols or fill in data
idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
n_bad_spectra = len(idx_bad_spectra[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
    spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

# normalize data set if requested
if normalize_spectra:
    print 'Normalizing data'
    normalizer = StandardScaler()
    spectral_data = normalizer.fit_transform(spectral_data)

# prepare spectral data for the further use in the Keras library
# spectral_data = np.expand_dims(spectral_data, axis=2)  # not needed in this case, only convolution needs 3D arrays

move_to_dir('Abundance_determination')

# --------------------------------------------------------
# ---------------- Train ANN on a train set of abundances
# --------------------------------------------------------
abund_param = Table.read(galah_data_input + abund_param_file)
cannon_abundances_list = [col for col in abund_param.colnames if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col]
sme_abundances_list = [col for col in abund_param.colnames if '_abund_sme' in col and 'e_' not in col and 'flag_' not in col]
# select only the ones with some datapoints
sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]

# final set of parameters
galah_param_complete = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                            abund_param[list(np.hstack(('sobject_id', cannon_abundances_list, sme_abundances_list)))],
                            keys='sobject_id', join_type='outer')
# replace/fill strange masked (--) values with np.nan
for c_col in galah_param_complete.colnames[1:]:   # 1: to skis sobject_id column
    galah_param_complete[c_col] = galah_param_complete[c_col].filled(np.nan)
# select only rows with valid parameters and spectra data
galah_param_complete = galah_param_complete[np.isfinite(galah_param_complete['teff_guess'])]
print 'Size complete: ' + str(len(galah_param_complete))

for sme_abundance in sme_abundances_list:
    print 'Working on anundance: ' + sme_abundance
    element = sme_abundance.split('_')[0]
    output_col = element + '_abund_ann'

    # create a subset of spectra to be train on the sme values
    param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                        abund_param['sobject_id', sme_abundance][np.isfinite(abund_param[sme_abundance])],
                        keys='sobject_id', join_type='inner')
    abund_values_train = param_joined[sme_abundance].data
    idx_spectra_train = np.in1d(galah_param['sobject_id'], param_joined['sobject_id'])

    print 'Number of train objects: ' + str(np.sum(idx_spectra_train))
    spectral_data_train = spectral_data[idx_spectra_train]

    # ann network - fully connected layers
    ann_input = Input(shape=(n_wvl_total,), name='Input_'+sme_abundance)
    ann = ann_input
    # fully connected layers
    for n_nodes in n_dense_nodes:
        ann = Dense(n_nodes, activation=None, name='Dense_'+str(n_nodes))(ann)
        ann = PReLU(name='PReLU_'+str(n_nodes))(ann)
        if dropout_learning and n_nodes > 1:
            ann = Dropout(0.4, name='Dropout_'+str(n_nodes))(ann)

    abundance_ann = Model(ann_input, ann)
    abundance_ann.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    abundance_ann.summary()

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    # fit the NN model
    abundance_ann.fit(spectral_data_train, abund_values_train,
                      epochs=100,
                      batch_size=128,
                      shuffle=True,
                      callbacks=[earlystop],
                      validation_split=0.1,
                      verbose=1)

    # evaluate on all spectra
    print 'Predicting abundance values from spectra'
    abundance_predicted = abundance_ann.predict(spectral_data)

    # add it to the final table
    galah_param_complete[output_col] = abundance_predicted

    # scatter plot of results to the reference cannon and sme values
    print 'Plotting graphs'
    plt.scatter(galah_param_complete[element+'_abund_sme'], galah_param_complete[output_col],
                lw=0, s=0.2, alpha=0.1, c='black')
    plt.xlabel('SME reference value')
    plt.ylabel('ANN computed value')
    plt.savefig(element+'_ANN_sme', dpi=400)
    plt.close()
    plt.scatter(galah_param_complete[element + '_abund_cannon'], galah_param_complete[output_col],
                lw=0, s=0.2, alpha=0.1, c='black')
    plt.xlabel('CANNON reference value')
    plt.ylabel('ANN computed value')
    plt.savefig(element + '_ANN_cannon', dpi=400)
    plt.close()

