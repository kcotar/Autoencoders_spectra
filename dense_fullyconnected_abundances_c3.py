import imp, os

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import regularizers, optimizers
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from astropy.table import Table, join, unique
from socket import gethostname
from itertools import combinations, combinations_with_replacement

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
    imp.load_source('helper_functions', '../Carbon-Spectra/helper_functions.py')
    imp.load_source('spectra_collection_functions', '../Carbon-Spectra/spectra_collection_functions.py')
else:
    galah_data_input = '/data4/cotar/'
from helper_functions import *
from spectra_collection_functions import *


line_file = 'GALAH_Cannon_linelist.csv'
data_date = '20171111'
galah_param_file = 'sobject_iraf_52_reduced_'+data_date+'.fits'
# abund_param_file = 'sobject_iraf_cannon2.1.7.fits'
abund_param_file = 'Cannon3.0.1_Sp_SMEmasks_trainingset.fits'  # can have multiple lines with the same sobject_id - this is on purpose
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_'+data_date+'.pkl',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_'+data_date+'.pkl',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_'+data_date+'.pkl',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_'+data_date+'.pkl']

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------
# algorithm settings - inputs, outputs etc
save_models = True
output_results = True
output_plots = True
limited_rows = False
snr_cut = False
output_reference_plot = False
use_renormalized_spectra = False

# data training and handling
read_fe_lines = False
train_multiple = True
n_train_multiple = 29
normalize_abund_values = True

# data normalization and training set selection
use_cannon_stellar_param = True
normalize_spectra = True
global_normalization = False
zero_mean_only = False
use_all_nonnan_rows = True
squared_components = False

# ann settings
dropout_learning = False
dropout_rate = 0.2
use_regularizer = False
activation_function = 'linear'  # if set to none defaults to PReLu

# convolution layer 1
C_f_1 = 64  # number of filters
C_k_1 = 7  # size of convolution kernel
C_s_1 = 1  # strides value
P_s_1 = 4  # size of pooling operator
# convolution layer 2
C_f_2 = 64
C_k_2 = 7
C_s_2 = 1
P_s_2 = 3
# convolution layer 3
C_f_3 = 64
C_k_3 = 5
C_s_3 = 1
P_s_3 = 3
n_dense_nodes = [2500, 900, 1]  # the last layer is output, its size will be determined on the fly

# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
import keras.backend as K
import tensorflow as T


def custom_error_function(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    return K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=-1)


def read_spectra(spectra_file_list, line_list, get_elements=None, read_wvl_offset=0.2, add_fe_lines=False):  # in A
    if add_fe_lines:
        get_elements.append('Fe')
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
        spectral_data.append(read_pkl_spectra(galah_data_input + spectra_file, read_rows=None, read_cols=abund_cols_read))
        # spectral_data.append(pd.read_csv(galah_data_input + spectra_file, sep=',', header=None,
        #                                  na_values='nan', skiprows=None, usecols=abund_cols_read).values)

    spectral_data = np.hstack(spectral_data)
    print spectral_data.shape
    return spectral_data


def correct_spectra():
    pass


def rmse(f1, f2):
    diff = f1 - f2
    n_nonna = np.sum(np.isfinite(diff))
    if n_nonna == 0:
        return np.nan
    else:
        return np.sqrt(np.nansum(diff**2)/n_nonna)


# --------------------------------------------------------
# ---------------- Data reading --------------------------
# --------------------------------------------------------
galah_param = Table.read(galah_data_input + galah_param_file)
line_list = Table.read(galah_data_input + line_file, format='ascii.csv')
abund_param = Table.read(galah_data_input + abund_param_file)
# TODO: enable traning procedure with multiple sme values for one sobject_id
# TODO: temporary solution is to use onyl one line
abund_param = unique(abund_param, keys=['sobject_id'])
cannon_abundances_list = [col for col in abund_param.colnames if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col]
sme_abundances_list = [col for col in abund_param.colnames if '_abund_sme' in col and 'e_' not in col and 'flag_' not in col]
sme_params = ['Teff_sme', 'Feh_sme', 'Logg_sme']
# select only the ones with some datapoints
sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]
sme_abundances_list = [col for col in sme_abundances_list if len(col.split('_')[0])<=3]
print 'Abundances:', sme_abundances_list
print 'N sme abund:', len(sme_abundances_list)

# reference plot for sme cannon values
if output_reference_plot:
    move_to_dir('Abundance reference')
    for sme_abund in sme_abundances_list:
        print ' plotting reference data attribute - ' + sme_abund
        canon_abund = sme_abund.split('_')[0] + '_abund_cannon'
        # determine number of lines used for this element
        plot_range = (np.nanpercentile(abund_param[sme_abund], 1), np.nanpercentile(abund_param[sme_abund], 99))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(abund_param[sme_abund], abund_param[canon_abund], lw=0, s=0.4, c='black')
        plt.title('Reference abundance Cannon and SME plot - RMSE: '+str(rmse(abund_param[sme_abund], abund_param[canon_abund])))
        plt.xlabel('SME reference value')
        plt.ylabel('CANNON computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(sme_abund + canon_abund + '.png', dpi=400)
        plt.close()
    os.chdir('..')

# check if renormalised data were requested
if use_renormalized_spectra:
    for i_l in range(len(spectra_file_list)):
        spectra_file_list[i_l] = spectra_file_list[i_l][:-4] + '_renorm.csv'

# read spectral data for selected abundance absorption lines
read_elements = [elem.split('_')[0].capitalize() for elem in sme_abundances_list]
spectral_data = read_spectra(spectra_file_list, line_list, get_elements=read_elements, add_fe_lines=read_fe_lines)
n_wvl_total = spectral_data.shape[1]

# somehow handle cols with nan values, delete cols or fill in data
idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
n_bad_spectra = len(idx_bad_spectra[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
    spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

# normalize data (flux at every wavelength)
if global_normalization:
    # TODO: save and recover normalization parameters
    global_norm_param = [np.mean(spectral_data), np.std(spectral_data)]
    spectral_data = spectral_data - global_norm_param[0]
    if not zero_mean_only:
        spectral_data /= global_norm_param[1]
else:
    if zero_mean_only:
        normalizer = StandardScaler(with_mean=True, with_std=False)
    else:
        normalizer = StandardScaler(with_mean=True, with_std=True)
    normalizer.fit(spectral_data)
    spectral_data = normalizer.transform(spectral_data)

# prepare spectral data for the further use in the Keras library
spectral_data = np.expand_dims(spectral_data, axis=2)

output_dir = 'Cannon3.0_SME'
if train_multiple:
    output_dir += '_multiple_'+str(n_train_multiple)
if C_s_1 > 1:
    output_dir += '_stride'+str(C_s_1)
if use_regularizer:
    output_dir += '_regularizer'
move_to_dir(output_dir)


# --------------------------------------------------------
# ---------------- Train ANN on a train set of abundances
# --------------------------------------------------------
# final set of parameters
galah_param_complete = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                            abund_param[list(np.hstack(('sobject_id', cannon_abundances_list, sme_abundances_list, sme_params)))],
                            keys='sobject_id', join_type='outer')
# replace/fill strange masked (--) values with np.nan
for c_col in galah_param_complete.colnames[1:]:   # 1: to skis sobject_id column
    galah_param_complete[c_col] = galah_param_complete[c_col].filled(np.nan)
# select only rows with valid parameters and spectra data
galah_param_complete = galah_param_complete[np.isfinite(galah_param_complete['teff_guess'])]
print 'Size complete: ' + str(len(galah_param_complete))

if train_multiple:
    sme_abundances_list = list(combinations(sme_abundances_list, n_train_multiple))

if use_cannon_stellar_param:
    # additional_train_feat = ['teff_cannon', 'feh_cannon', 'logg_cannon']
    additional_train_feat = sme_params
else:
    additional_train_feat = ['teff_guess', 'feh_guess', 'logg_guess']

print 'SME list:', sme_abundances_list
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
    n_dense_nodes[-1] = len(sme_abundance) + len(additional_train_feat)  # 3 outputs for stellar physical parameters

    # create squared train components if requested to do so
    if squared_components:
        cols_to_square = np.hstack((sme_abundance, additional_train_feat))
        squared_train_feat = list([])
        for comb in combinations_with_replacement(cols_to_square, 2):
            new_label = str(comb[0])+'*'+str(comb[1])
            print ' Creating new combination: '+new_label
            new_label_values = abund_param[comb[0]] * abund_param[comb[1]]
            # check for number of nan values in this new training feature
            n_values = 1.*len(new_label_values)
            n_ok = np.sum(np.isfinite(new_label_values))
            n_nans = (n_values - n_ok)
            # the new feature must have sufficient number/percent of not nan values to be trained on
            if n_ok > 300:
                squared_train_feat.append(new_label)
                abund_param[new_label] = new_label_values
            else:
                print ' -> too many nan values (ok values: '+str(n_ok)+')'
        n_squared = len(squared_train_feat)
        print ' Total number of squared: '+str(n_squared)
        n_dense_nodes[-1] += n_squared

    if np.sum(idx_abund_cols) < 300:
        print ' Not enough train data to do this.'
        continue

    if use_all_nonnan_rows:
        plot_suffix += '_with_nan'

    # add some additional suffix describing processing parameters
    if global_normalization:
        plot_suffix += '_globalnorm'
    if zero_mean_only:
        plot_suffix += '_zeromean'
    if squared_components:
        plot_suffix += '_squared'
    if use_renormalized_spectra:
        plot_suffix += '_renorm'
    if use_cannon_stellar_param:
        plot_suffix += '_smeparam'
    if read_fe_lines:
        plot_suffix += '_withfe'
    # plot_suffix += '_f'+str(C_f_1)+'-'+str(C_f_2)+'-'+str(C_f_3)
    plot_suffix += '_optim'

    # create a subset of spectra to be train on the sme values
    if squared_components:
        param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                            abund_param[list(np.hstack(('sobject_id', sme_abundance, sme_params, squared_train_feat)))][idx_abund_cols],
                            keys='sobject_id', join_type='inner')
    else:
        param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                            abund_param[list(np.hstack(('sobject_id', sme_abundance, sme_params)))][idx_abund_cols],
                            keys='sobject_id', join_type='inner')
    idx_spectra_train = np.in1d(galah_param['sobject_id'], param_joined['sobject_id'])
    # Data consistency and repetition checks
    # print 'N param orig:', len(galah_param), 'uniqu', len(np.unique(galah_param['sobject_id']))
    # print 'N abund orig:', len(abund_param), 'uniqu', len(np.unique(abund_param['sobject_id']))
    # print 'N param:', len(param_joined), 'uniqu', len(np.unique(param_joined['sobject_id']))
    # print 'N spect:', np.sum(idx_spectra_train), 'uniqu', len(np.unique(galah_param['sobject_id'][idx_spectra_train]))

    if squared_components:
        abund_values_train = param_joined[list(np.hstack((sme_abundance, additional_train_feat, squared_train_feat)))].to_pandas().values
    else:
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

    # set up regularizer if needed
    if use_regularizer:
        # use a combination of l1 and/or l2 regularizer
        # w_reg = regularizers.l1_l2(1e-5, 1e-5)
        # a_reg = regularizers.l1_l2(1e-5, 1e-5)
        w_reg = regularizers.l2(1e-5)
        a_reg = regularizers.l2(1e-5)
    else:
        # default values for Conv1D and Dense layers
        w_reg = None
        a_reg = None

    # ann network - fully connected layers
    ann_input = Input(shape=(spectral_data_train.shape[1], 1), name='Input_'+plot_suffix)
    ann = ann_input

    ann = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_1', strides=C_s_1,
                 kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
    ann = PReLU(name='R_1')(ann)
    ann = MaxPooling1D(P_s_1, padding='same', name='P_1')(ann)
    if C_f_2 > 0:
        ann = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_2', strides=C_s_2,
                     kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        ann = PReLU(name='R_2')(ann)
        ann = MaxPooling1D(P_s_2, padding='same', name='P_2')(ann)
    ann = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_3', strides=C_s_3,
                 kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
    ann = PReLU(name='R_3')(ann)
    encoded_cae = MaxPooling1D(P_s_3, padding='same', name='P_3')(ann)

    # flatter output from convolutional network to the shape useful for fully-connected dense layers
    ann = Flatten(name='Conv_to_Dense')(ann)

    # fully connected layers
    for n_nodes in n_dense_nodes:
        ann = Dense(n_nodes, activation=activation_function, name='Dense_'+str(n_nodes),
                    kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        # add activation function to the layer
        if n_nodes > 25:
            # internal fully connected layers in ann network
            if dropout_learning:
                ann = Dropout(dropout_rate, name='Dropout_'+str(n_nodes))(ann)
            if activation_function is None:
                ann = PReLU(name='PReLU_' + str(n_nodes))(ann)
        else:
            # output layer
            ann = Activation('linear')(ann)
            # ann = PReLU(name='PReLU_' + str(n_nodes))(ann)

    abundance_ann = Model(ann_input, ann)
    selected_optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    if use_all_nonnan_rows:
        abundance_ann.compile(optimizer=selected_optimizer, loss=custom_error_function)
    else:
        abundance_ann.compile(optimizer=selected_optimizer, loss='mse')
    abundance_ann.summary()

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')
    # fit the NN model
    abundance_ann.fit(spectral_data_train, abund_values_train,
                      epochs=125,
                      batch_size=512,
                      shuffle=True,
                      callbacks=[earlystop],
                      validation_split=0.05,
                      verbose=2)

    # evaluate on all spectra
    print 'Predicting abundance values from spectra'
    abundance_predicted = abundance_ann.predict(spectral_data)
    if normalize_abund_values:
        print 'Denormalizing output values of features'
        # abundance_predicted = normalizer_outptu.inverse_transform(abundance_predicted)
        # another option
        for i_f in range(n_train_feat):
            abundance_predicted[:, i_f] = abundance_predicted[:, i_f] * train_feat_std[i_f] + train_feat_mean[i_f]

    # add parameters to the output
    for s_p in additional_train_feat:
        output_col.append(s_p.split('_')[0]+'_ann')
    # add abundance results to the final table
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

    # determine a feature to be used a colour of points
    c_data = np.int64(param_joined['teff_guess'].data)
    c_data_min = np.nanpercentile(c_data, 1)
    c_data_max = np.nanpercentile(c_data, 99)
    print c_data
    print c_data_min
    print c_data_max

    # sme_abundances_plot = np.hstack((sme_abundance, additional_train_feat))
    print 'Plotting graphs'
    for plot_abund in sme_abundances_plot:
        print ' plotting attribute - ' + plot_abund
        elem_plot = plot_abund.split('_')[0]
        # determine number of lines used for this element
        n_lines_element = np.sum(line_list['Element'] == elem_plot.capitalize())
        graphs_title = elem_plot.capitalize() + ' - SME train objects: ' + str(n_train_sme) + ' (lines: ' + str(n_lines_element) + ') - RMSE: '+str(rmse(galah_param_complete[elem_plot+'_abund_sme'], galah_param_complete[elem_plot+'_abund_ann']))
        plot_range = (np.nanpercentile(abund_param[plot_abund], 1), np.nanpercentile(abund_param[plot_abund], 99))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[elem_plot+'_abund_sme'], galah_param_complete[elem_plot+'_abund_ann'],
                    lw=0, s=0.4, c='black') #c=c_data, cmap='jet', vmin=c_data_min, vmax=c_data_max)
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(elem_plot+'_ANN_sme_'+plot_suffix+'.png', dpi=400)
        plt.close()

    for plot_param in sme_params:
        print ' plotting parameter - ' + plot_param
        param_plot = plot_param.split('_')[0]
        # determine number of lines used for this element
        graphs_title = 'SME parameter '+param_plot
        plot_range = (np.nanpercentile(abund_param[plot_param], 1), np.nanpercentile(abund_param[plot_param], 99))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[param_plot+'_sme'], galah_param_complete[param_plot+'_ann'],
                    lw=0, s=0.4, c='black') #c=c_data, cmap='jet', vmin=c_data_min, vmax=c_data_max)
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(param_plot+'_ANN_sme_'+plot_suffix+'.png', dpi=400)
        plt.close()

        # second graph - cannon points
"""
        graphs_title = elem_plot.capitalize() + ' - SME train objects: ' + str(n_train_sme) + ' (lines: ' + str(n_lines_element) + ') - RMSE: '+str(rmse(galah_param_complete[elem_plot+'_abund_cannon'], galah_param_complete[elem_plot+'_abund_ann']))
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[elem_plot + '_abund_cannon'], galah_param_complete[elem_plot+'_abund_ann'],
                    lw=0, s=0.4, c='black') #c=c_data, cmap='jet', vmin=c_data_min, vmax=c_data_max)
        plt.title(graphs_title)
        plt.xlabel('CANNON reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(elem_plot + '_ANN_cannon_'+plot_suffix+'.png', dpi=400)
        plt.close()
"""

# aslo save resuts at the end
fits_out = 'galah_abund_ANN_SME3.0.1.fits'
galah_param_complete.write(fits_out)


