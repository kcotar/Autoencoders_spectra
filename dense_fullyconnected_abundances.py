import imp, os

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from astropy.table import Table, join
from socket import gethostname
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PC hostname
pc_name = gethostname()

# input data
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
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
# algorithm settings - inputs, outputs etc
save_models = True
output_results = True
output_plots = True
limited_rows = False
snr_cut = False

# data training and handling
train_multiple = True
n_train_multiple = 23
normalize_spectra = True
normalize_abund_values = True
use_all_nonnan_rows = True

# ann settings
dropout_learning = True
activation_function = None  # if set to none defaults to PReLu

# convolution layer 1
C_f_1 = 256  # number of filters
C_k_1 = 9  # size of convolution kernel
C_s_1 = 1  # strides value
P_s_1 = 4  # size of pooling operator
# convolution layer 2
C_f_2 = 128
C_k_2 = 5
C_s_2 = 1
P_s_2 = 4
# convolution layer 3
C_f_3 = 128
C_k_3 = 3
C_s_3 = 1
P_s_3 = 4
n_dense_nodes = [1800, 500, 1]  # the last layer is output, its size will be determined on the fly

# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
import keras.backend as K
import tensorflow as T


def custom_error_function(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    return K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=-1)


def read_spectra(spectra_file_list, line_list, get_elements=None, read_wvl_offset=0.1):  # in A
    if get_elements is not None:
        idx_list = np.in1d(line_list['Element'], get_elements, assume_unique=False)
        line_list_read = line_list[idx_list]
    else:
        line_list_read = Table(line_list)
    spectral_data = list([])
    for i_band in [0, 1, 2, 3]:
        spectra_file = spectra_file_list[i_band]
        # determine what is to be read from the spectra
        print 'Defining cols to be read'
        abund_cols_read = list([])
        spectra_file_split = spectra_file.split('_')
        wvl_values = np.arange(float(spectra_file_split[3]), float(spectra_file_split[4]), float(spectra_file_split[6]))
        for line in line_list_read:
            idx_wvl = np.logical_and(wvl_values >= line['line_start'] - read_wvl_offset,
                                     wvl_values <= line['line_end'] + read_wvl_offset)
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
    return spectral_data


def correct_spectra():
    pass

# --------------------------------------------------------
# ---------------- Data reading --------------------------
# --------------------------------------------------------
galah_param = Table.read(galah_data_input + galah_param_file)
line_list = Table.read(galah_data_input + line_file, format='ascii.csv')
abund_param = Table.read(galah_data_input + abund_param_file)
cannon_abundances_list = [col for col in abund_param.colnames if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col]
sme_abundances_list = [col for col in abund_param.colnames if '_abund_sme' in col and 'e_' not in col and 'flag_' not in col]
# select only the ones with some datapoints
sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]

read_elements = [elem.split('_')[0].capitalize() for elem in sme_abundances_list]
spectral_data = read_spectra(spectra_file_list, line_list, get_elements=read_elements)
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
spectral_data = np.expand_dims(spectral_data, axis=2)

output_dir = 'Abundance_determination'
if train_multiple:
    output_dir += '_multiple_'+str(n_train_multiple)
move_to_dir(output_dir)

# --------------------------------------------------------
# ---------------- Train ANN on a train set of abundances
# --------------------------------------------------------
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

if train_multiple:
    sme_abundances_list = list(combinations(sme_abundances_list, n_train_multiple))

additional_train_feat = ['teff_guess', 'feh_guess', 'logg_guess']
for sme_abundance in sme_abundances_list:
    if train_multiple:
        print 'Working on multiple abundance: ' + ' '.join(sme_abundance)
        elements = [sme.split('_')[0] for sme in sme_abundance]
        plot_suffix = '_'.join(elements)
        output_col = [elem+'_abund_ann' for elem in elements]
        if use_all_nonnan_rows:
            idx_abund_cols = np.isfinite(abund_param[sme_abundance].to_pandas().values).any(axis=1)
        else:
            idx_abund_cols = np.isfinite(abund_param[sme_abundance].to_pandas().values).all(axis=1)
    else:
        print 'Working on abundance: ' + sme_abundance
        element = sme_abundance.split('_')[0]
        plot_suffix = element
        output_col = [element + '_abund_ann']
        if use_all_nonnan_rows:
            # TODO: more elegant way to handle this
            idx_abund_cols = np.isfinite(abund_param[sme_abundance])
        else:
            idx_abund_cols = np.isfinite(abund_param[sme_abundance])
    n_dense_nodes[-1] = len(sme_abundance) + 3  # 3 outputs for stellar physical parameters

    if np.sum(idx_abund_cols) < 300:
        print ' Not enough train data to do this.'
        continue

    if use_all_nonnan_rows:
        plot_suffix += '_with_nan'

    # create a subset of spectra to be train on the sme values
    param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                        abund_param[list(np.hstack(('sobject_id', sme_abundance)))][idx_abund_cols],
                        keys='sobject_id', join_type='inner')
    idx_spectra_train = np.in1d(galah_param['sobject_id'], param_joined['sobject_id'])

    abund_values_train = param_joined[list(np.hstack((sme_abundance, additional_train_feat)))].to_pandas().values

    if normalize_abund_values:
        # normalizer_outptu = StandardScaler()
        # abund_values_train = normalizer_outptu.fit_transform(abund_values_train)
        # another version to deal with NaN/Inf values
        print 'Normalizing input train parameters'
        n_train_feat = abund_values_train.shape[1]
        train_feat_mean = np.zeros(n_train_feat)
        train_feat_std = np.zeros(n_train_feat)
        for i_f in range(n_train_feat):
            train_feat_mean[i_f] = np.nanmean(abund_values_train[:, i_f])
            train_feat_std[i_f] = np.nanstd(abund_values_train[:, i_f])
            abund_values_train[:, i_f] = (abund_values_train[:, i_f] - train_feat_mean[i_f])/train_feat_std[i_f]

    n_train_sme = np.sum(idx_spectra_train)
    print 'Number of train objects: ' + str(n_train_sme)
    spectral_data_train = spectral_data[idx_spectra_train]
    # spectral_data_train = np.expand_dims(spectral_data[idx_spectra_train], axis=2)

    # ann network - fully connected layers
    ann_input = Input(shape=(spectral_data_train.shape[1], 1), name='Input_'+plot_suffix)
    ann = ann_input

    ann = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_1', strides=C_s_1)(ann)
    ann = PReLU(name='R_1')(ann)
    ann = MaxPooling1D(P_s_1, padding='same', name='P_1')(ann)
    if C_f_2 > 0:
        ann = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_2', strides=C_s_2)(ann)
        ann = PReLU(name='R_2')(ann)
        ann = MaxPooling1D(P_s_2, padding='same', name='P_2')(ann)
    ann = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_3', strides=C_s_3)(ann)
    ann = PReLU(name='R_3')(ann)
    encoded_cae = MaxPooling1D(P_s_3, padding='same', name='P_3')(ann)

    # flatter output from convolutional network to the shape useful for fully-connected dense layers
    ann = Flatten(name='Conv_to_Dense')(ann)

    # fully connected layers
    for n_nodes in n_dense_nodes:
        ann = Dense(n_nodes, activation=activation_function, name='Dense_'+str(n_nodes))(ann)
        if dropout_learning and n_nodes > 25:
            # internal fully connected layers in ann network
            ann = Dropout(0.2, name='Dropout_'+str(n_nodes))(ann)
            if activation_function is None:
                ann = PReLU(name='PReLU_' + str(n_nodes))(ann)
        else:
            # output layer
            # ann = Activation('sigmoid')(ann)
            ann = PReLU(name='PReLU_' + str(n_nodes))(ann)

    abundance_ann = Model(ann_input, ann)
    if use_all_nonnan_rows:
        abundance_ann.compile(optimizer='adam', loss=custom_error_function, metrics=['accuracy'])
    else:
        abundance_ann.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    abundance_ann.summary()

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    # fit the NN model
    abundance_ann.fit(spectral_data_train, abund_values_train,
                      epochs=150,
                      batch_size=256,
                      shuffle=True,
                      callbacks=[earlystop],
                      validation_split=0.1,
                      verbose=1)

    # evaluate on all spectra
    print 'Predicting abundance values from spectra'
    abundance_predicted = abundance_ann.predict(spectral_data)
    if normalize_abund_values:
        print 'Denormalizing output values of features'
        # abundance_predicted = normalizer_outptu.inverse_transform(abundance_predicted)
        # another option
        for i_f in range(n_train_feat):
            abundance_predicted[:, i_f] = abundance_predicted[:, i_f] * train_feat_std[i_f] + train_feat_mean[i_f]

    # add it to the final table
    for i_o in range(len(output_col)):
        galah_param_complete[output_col[i_o]] = abundance_predicted[:, i_o]

    if activation_function is None:
        plot_suffix += '_prelu'
    else:
        plot_suffix += '_'+activation_function

    # scatter plot of results to the reference cannon and sme values
    if train_multiple:
        sme_abundances_plot = sme_abundance
    else:
        sme_abundances_plot = [sme_abundance]

    # sme_abundances_plot = np.hstack((sme_abundance, additional_train_feat))
    print 'Plotting graphs'
    for plot_abund in sme_abundances_plot:
        print ' plotting attribute - ' + plot_abund
        elem_plot = plot_abund.split('_')[0]
        graphs_title = elem_plot.capitalize() + ' - number of SME train objects is ' + str(n_train_sme)
        plot_range = (np.nanpercentile(abund_param[plot_abund], 1), np.nanpercentile(abund_param[plot_abund], 99))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[elem_plot+'_abund_sme'], galah_param_complete[elem_plot+'_abund_ann'],
                    lw=0, s=0.3, alpha=0.4, c='black')
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(elem_plot+'_ANN_sme_'+plot_suffix+'.png', dpi=400)
        plt.close()
        # second graph - cannon points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[elem_plot + '_abund_cannon'], galah_param_complete[elem_plot+'_abund_ann'],
                    lw=0, s=0.3, alpha=0.2, c='black')
        plt.title(graphs_title)
        plt.xlabel('CANNON reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(elem_plot + '_ANN_cannon_'+plot_suffix+'.png', dpi=400)
        plt.close()

