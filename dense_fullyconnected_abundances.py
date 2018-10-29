import imp, os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import regularizers, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
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
galah_data_input = '/data4/cotar/'
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
    imp.load_source('spectra_collection_functions', '../Carbon-Spectra/spectra_collection_functions.py')

from helper_functions import move_to_dir
from spectra_collection_functions import read_pkl_spectra, save_pkl_spectra

date_string = '20180327'
line_file = 'GALAH_Cannon_linelist_newer.csv'
galah_param_file = 'sobject_iraf_53_reduced_'+date_string+'.fits'
# abund_param_file = 'sobject_iraf_cannon2.1.7.fits'
abund_param_file = 'GALAH_iDR3_ts_DR2.fits'  # can have multiple lines with the same sobject_id - this is on purpose
spectra_file_list = ['galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd2_5640_5880_wvlstep_0.050_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd4_7700_7895_wvlstep_0.070_ext4_'+date_string+'.pkl']
#spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.020_lin_renorm_'+date_string+'.pkl',
#                     'galah_dr52_ccd2_5640_5880_wvlstep_0.025_lin_renorm_'+date_string+'.pkl',
#                     'galah_dr52_ccd3_6475_6745_wvlstep_0.030_lin_renorm_'+date_string+'.pkl',
#                     'galah_dr52_ccd4_7700_7895_wvlstep_0.035_lin_renorm_'+date_string+'.pkl']
# spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.020_ext0_renorm_'+date_string+'.pkl',
#                      'galah_dr52_ccd2_5640_5880_wvlstep_0.025_ext0_renorm_'+date_string+'.pkl',
#                      'galah_dr52_ccd3_6475_6745_wvlstep_0.030_ext0_renorm_'+date_string+'.pkl',
#                      'galah_dr52_ccd4_7700_7895_wvlstep_0.035_ext0_renorm_'+date_string+'.pkl']

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
read_complete_spectrum = True
read_fe_lines = True
train_multiple = True
n_train_multiple = 23
normalize_abund_values = True

# data normalization and training set selection
use_cannon_stellar_param = True
global_normalization = True
zero_mean_only = True
use_all_nonnan_rows = True
squared_components = False

# ann settings
dropout_learning = False
dropout_rate = 0.2
use_regularizer = False
activation_function = None#'linear'  # if set to none defaults to PReLu


n_dense_nodes = [2500, 900, 1]  # the last layer is output, its size will be determined on the fly

# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
import keras.backend as K
import tensorflow as T


def custom_error_function(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    return K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=-1)


def read_spectra(spectra_file_list, line_list, complete_spectrum=False, get_elements=None, read_wvl_offset=0.2, add_fe_lines=False):  # in A
    if add_fe_lines:
        get_elements.append('Fe')  # RISKY should be replaced with deep copy of the object
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
        spectra_file_split = spectra_file.split('_')
        wvl_values = np.arange(float(spectra_file_split[3]), float(spectra_file_split[4]), float(spectra_file_split[6]))
        if not complete_spectrum:
            abund_cols_read = list([])
            for line in line_list_read:
                idx_wvl = np.logical_and(wvl_values >= line['line_start'] - read_wvl_offset,
                                         wvl_values <= line['line_end'] + read_wvl_offset)
                if np.sum(idx_wvl) > 0:
                    abund_cols_read.append(np.where(idx_wvl)[0])
            abund_cols_read = np.unique(np.hstack(abund_cols_read))  # unique instead of sort to remove duplicated wvl pixels
        else:
            abund_cols_read = np.where(wvl_values > 0.)[0]
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
# cannon_abundances_list = [col for col in abund_param.colnames if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col]
# sme_abundances_list = [col for col in abund_param.colnames if '_abund_sme' in col and 'e_' not in col and 'flag_' not in col]
sme_abundances_list = [col for col in abund_param.colnames if '_fe' in col and len(col.split('_')) == 2 and len(col.split('_')[0]) <= 2]
sme_params = ['teff', 'fe_h', 'logg', 'vbroad']

# select only the ones with some datapoints
sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]
sme_abundances_list = [col for col in sme_abundances_list if len(col.split('_')[0])<=3]
print 'SME Abundances:', sme_abundances_list
print 'Number Abund:', len(sme_abundances_list)


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
spectral_data = read_spectra(spectra_file_list, line_list, complete_spectrum=read_complete_spectrum, get_elements=read_elements, add_fe_lines=read_fe_lines)
n_wvl_total = spectral_data.shape[1]

# somehow handle cols with nan values, delete cols or fill in data
idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
n_bad_spectra = len(idx_bad_spectra[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
    spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

# normalize data (flux at every wavelength)
if global_normalization:
    # shift normalized spectra from the 1...0 range to 0...1, where 0 if flux level
    spectral_data = 1. - spectral_data
    # # TODO: save and recover normalization parameters
    # global_norm_param = [np.mean(spectral_data), np.std(spectral_data)]
    # spectral_data = spectral_data - global_norm_param[0]
    # if not zero_mean_only:
    #     spectral_data /= global_norm_param[1]
else:
    if zero_mean_only:
        normalizer = StandardScaler(with_mean=True, with_std=False)
    else:
        normalizer = StandardScaler(with_mean=True, with_std=True)
    normalizer.fit(spectral_data)
    spectral_data = normalizer.transform(spectral_data)

# prepare spectral data for the further use in the Keras library
spectral_data = np.expand_dims(spectral_data, axis=2)

output_dir = '/data4/cotar/Cannon3.0_SME_fullyconnected_'+date_string
if train_multiple:
    output_dir += '_multiple_'+str(n_train_multiple)
# if C_s_1 > 1:
#     output_dir += '_stride'+str(C_s_1)
if use_regularizer:
    output_dir += '_regularizer'
output_dir += '_prelu_globflux'
move_to_dir(output_dir)


# --------------------------------------------------------
# ---------------- Train ANN on a train set of abundances
# --------------------------------------------------------
# final set of parameters
galah_param_complete = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                            abund_param[list(np.hstack(('sobject_id', sme_abundances_list, sme_params)))],
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
        elements = sme_abundance.split('_')[0]
        plot_suffix = elements
        output_col = [elements + '_abund_ann']
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
    # plot_suffix += '_f'+str(C_f_1)+'-'+str(C_f_2)+'-'+str(C_f_3)
    if use_cannon_stellar_param:
        plot_suffix += '_cannon'
    if read_fe_lines:
        plot_suffix += '_withfe'

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

    # ------------------------------ Convolution part---------------------------------------------
    # ann = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_1', strides=C_s_1,
    #              kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
    # ann = PReLU(name='R_1')(ann)
    # ann = MaxPooling1D(P_s_1, padding='same', name='P_1')(ann)
    # if C_f_2 > 0:
    #     ann = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_2', strides=C_s_2,
    #                  kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
    #     ann = PReLU(name='R_2')(ann)
    #     ann = MaxPooling1D(P_s_2, padding='same', name='P_2')(ann)
    # ann = Conv1D(C_f_3, C_k_3, activation=None, padding='same', name='C_3', strides=C_s_3,
    #              kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
    # ann = PReLU(name='R_3')(ann)
    # encoded_cae = MaxPooling1D(P_s_3, padding='same', name='P_3')(ann)
    #
    # # flatter output from convolutional network to the shape useful for fully-connected dense layers
    # ann = Flatten(name='Conv_to_Dense')(ann)
    # -----------------------------------------------------------------------------------------------

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
        output_col.append(s_p + '_ann')
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
        graphs_title = elem_plot.capitalize() + ' - SME train objects: ' + str(n_train_sme) + ' (lines: ' + str(
            n_lines_element) + ') - RMSE: ' + str(
            rmse(galah_param_complete[plot_abund], galah_param_complete[elem_plot + '_abund_ann']))
        plot_range = (np.nanpercentile(abund_param[plot_abund], 1), np.nanpercentile(abund_param[plot_abund], 99))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[plot_abund], galah_param_complete[elem_plot + '_abund_ann'],
                    lw=0, s=0.4, c='black')  # c=c_data, cmap='jet', vmin=c_data_min, vmax=c_data_max)
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(elem_plot + '_ANN_sme_' + plot_suffix + '.png', dpi=400)
        plt.close()

    for plot_param in sme_params:
        print ' plotting parameter - ' + plot_param
        param_plot = plot_param  # .split('_')[0]
        # determine number of lines used for this element
        graphs_title = 'SME parameter ' + param_plot
        plot_range = (np.nanpercentile(abund_param[plot_param], 0.1), np.nanpercentile(abund_param[plot_param], 99.9))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[param_plot], galah_param_complete[param_plot + '_ann'],
                    lw=0, s=0.4, c='black')  # c=c_data, cmap='jet', vmin=c_data_min, vmax=c_data_max)
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('ANN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(param_plot + '_ANN_sme_' + plot_suffix + '.png', dpi=400)
        plt.close()

# aslo save resuts at the end
fits_out = 'galah_abund_ANN.fits'
galah_param_complete.write(fits_out)

