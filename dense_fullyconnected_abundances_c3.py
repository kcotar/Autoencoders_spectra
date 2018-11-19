import imp, os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import regularizers, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.callbacks import EarlyStopping, ModelCheckpoint
from astropy.table import Table, join, unique, vstack
from socket import gethostname
from itertools import combinations, combinations_with_replacement
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras import backend as K
import tensorflow as tf
tf_config = tf.ConfigProto(intra_op_parallelism_threads=72,
                           inter_op_parallelism_threads=72,
                           allow_soft_placement=True)
session = tf.Session(config=tf_config)
K.set_session(session)

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
read_complete_spectrum = True  # read and use all spectrum pixels
read_H_lines = False  # TODO: to be implemented
read_fe_lines = True
train_multiple = True
n_train_multiple = 30  # 30 elements in total
normalize_abund_values = True
n_multiple_runs = 11

# data normalization and training set selection
use_cannon_stellar_param = True
global_normalization = True
zero_mean_only = True
use_all_nonnan_rows = True
squared_components = False

# ann settings
dropout_learning = True
dropout_rate = 0.2
dropout_learning_c = False
dropout_rate_c = 0.2
use_regularizer = False
activation_function = None  # dense layers - if set to None defaults to PReLu
activation_function_c = None  # 'relu'  # None  # convolution layers - if set to None defaults to PReLu

# convolution layer 1
C_f_1 = 32  # number of filters
C_k_1 = 11  # size of convolution kernel
C_s_1 = 1  # strides value
P_s_1 = 3  # size of pooling operator
# convolution layer 2
C_f_2 = 64
C_k_2 = 7
C_s_2 = 1
P_s_2 = 3
# convolution layer 3
C_f_3 = 0
C_k_3 = 3
C_s_3 = 1
P_s_3 = 3
n_dense_nodes = [3000, 1500, 700, 300, 1]  # the last layer is output, its size will be determined on the fly

# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
import keras.backend as K
import tensorflow as T


def custom_error_function(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    mse = K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=-1)
    return K.sum(mse)


def custom_error_function_2(y_true, y_pred):
    # VERSION1 - not sure if indexing and axis value are correct in this way
    # NOTE: boolean_mask reduces the dimensionality of the matrix, therefore the loss is prevailed by parameters with more observations
    # bool_finite = T.is_finite(y_true)
    # mse = K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=0)
    # return K.sum(mse)
    # VERSION2 - same thing, but using more understandable, but probably a bit slower for loop
    mse_final = 0
    for i1 in range(K.int_shape(y_pred)[1]):
        v1 = y_pred[:, i1]
        v2 = y_true[:, i1]
        bool_finite = T.is_finite(v2)
        mse_final += K.mean(K.square(T.boolean_mask(v1, bool_finite) - T.boolean_mask(v2, bool_finite)))
    return mse_final


def custom_error_function_3(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    mae = K.mean(K.abs(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=0)
    return K.sum(mae)


def custom_error_function_4(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    log_cosh = K.mean(K.log(K.cosh(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite))), axis=0)
    return K.sum(log_cosh)


def custom_error_function_5(y_true, y_pred):
    bool_finite = T.is_finite(y_true)
    mse = K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=0)
    mse_final = K.sum(mse)
    for i1 in range(K.int_shape(y_pred)[1]):
        for i2 in range(i1+1, K.int_shape(y_pred)[1]):
            v1 = y_pred[:, i1] * y_pred[:, i2]
            v2 = y_true[:, i1] * y_true[:, i2]
            bool_finite = T.is_finite(v2)
            mse_final += K.mean(K.square(T.boolean_mask(v1, bool_finite) - T.boolean_mask(v2, bool_finite)))
    return mse_final


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


def bias(f1, f2):
    diff = f1 - f2
    if np.sum(np.isfinite(diff)) == 0:
        return np.nan
    else:
        return np.nanmedian(diff)


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

# validation cluster data
openc_param = Table.read(galah_data_input + 'GALAH_iDR3_OpenClusters.fits')
globc_param = Table.read(galah_data_input + 'GALAH_iDR3_GlobularClusters.fits')
cluster_param = vstack([openc_param, globc_param])
cluster_param = unique(cluster_param, keys=['sobject_id'])  # sort the data among other thing

# TEST stact all data together as train
abund_param = unique(vstack([cluster_param, abund_param]), keys=['sobject_id'])

# select only the ones with some datapoints
sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]
sme_abundances_list = [col for col in sme_abundances_list if len(col.split('_')[0]) <= 3]
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
        plt.title('Reference abundance Cannon and SME plot - BIAS: {:.2f}    RMSE: {:.2f}'.format(bias(abund_param[sme_abund], abund_param[canon_abund]), rmse(abund_param[sme_abund], abund_param[canon_abund])))
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
    print ' Shifting flux levels into PRelu area'
    spectral_data = 1. - spectral_data
    # # TODO: save and recover normalization parameters
    # global_norm_param = [np.mean(spectral_data), np.std(spectral_data)]
    # spectral_data = spectral_data - global_norm_param[0]
    # if not zero_mean_only:
    #     spectral_data /= global_norm_param[1]
else:
    normalizer_file = 'normalizer_spectra_'+str(n_wvl_total)+'.pkl'
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

# prepare spectral data for the further use in the Keras library
spectral_data = np.expand_dims(spectral_data, axis=2)

output_dir = '/data4/cotar/Cannon3.0_SME_'+date_string
if train_multiple:
    output_dir += '_multiple_'+str(n_train_multiple)
if C_s_1 > 1:
    output_dir += '_stride'+str(C_s_1)
if use_regularizer:
    output_dir += '_regularizer'
if dropout_learning:
    output_dir += '_dropout{:.1f}'.format(dropout_rate)
if read_complete_spectrum:
    output_dir += '_allspectrum'
elif read_fe_lines:
    output_dir += '_alllines'
if activation_function is not None:
    output_dir += '_'+activation_function

output_dir += '_C-{:.0f}-{:.0f}-{:.0f}_F-{:.0f}-{:.0f}-{:.0f}_Adam_completetrain'.format(C_k_1, C_k_2, C_k_3, C_f_1, C_f_2, C_f_3)
move_to_dir(output_dir)


# --------------------------------------------------------
# ---------------- Train ANN on a train set of abundances
# --------------------------------------------------------
# final set of parameters
galah_param_complete = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                            abund_param[list(np.hstack(('sobject_id', sme_abundances_list, sme_params)))],
                            keys='sobject_id', join_type='left')
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
for i_run in np.arange(n_multiple_runs)+1:
    print 'ANN run number:', i_run
    for sme_abundance in sme_abundances_list:
        if train_multiple:
            print 'Working on multiple abundance: ' + ' '.join(sme_abundance)
            elements = [sme.split('_')[0] for sme in sme_abundance]
            plot_suffix = '_'.join(elements)
            output_col = [elem+'_abund_ann' for elem in elements]
            if use_all_nonnan_rows:
                idx_abund_rows = np.isfinite(abund_param[sme_abundance].to_pandas().values).any(axis=1)
                idx_abund_rows_valid = np.isfinite(cluster_param[sme_abundance].to_pandas().values).any(axis=1)
            else:
                idx_abund_rows = np.isfinite(abund_param[sme_abundance].to_pandas().values).all(axis=1)
                idx_abund_rows_valid = np.isfinite(cluster_param[sme_abundance].to_pandas().values).all(axis=1)
        else:
            print 'Working on abundance: ' + sme_abundance
            elements = sme_abundance.split('_')[0]
            move_to_dir(elements)
            plot_suffix = elements
            output_col = [elements + '_abund_ann']
            if use_all_nonnan_rows:
                # TODO: more elegant way to handle this
                idx_abund_rows = np.isfinite(abund_param[sme_abundance])
                idx_abund_rows_valid = np.isfinite(cluster_param[sme_abundance])
            else:
                idx_abund_rows = np.isfinite(abund_param[sme_abundance])
                idx_abund_rows_valid = np.isfinite(cluster_param[sme_abundance])
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

        if np.sum(idx_abund_rows) < 300:
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
        # plot_suffix += '_optim'

        # create a subset of spectra to be train on the sme values
        if squared_components:
            param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                                abund_param[list(np.hstack(('sobject_id', sme_abundance, sme_params, squared_train_feat)))][idx_abund_rows],
                                keys='sobject_id', join_type='inner')
        else:
            param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                                abund_param[list(np.hstack(('sobject_id', sme_abundance, sme_params)))][idx_abund_rows],
                                keys='sobject_id', join_type='inner')
            param_joined_valid = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                                      cluster_param[list(np.hstack(('sobject_id', sme_abundance, sme_params)))][idx_abund_rows_valid],
                                      keys='sobject_id', join_type='inner')
        idx_spectra_train = np.in1d(galah_param['sobject_id'], param_joined['sobject_id'])
        idx_spectra_valid = np.in1d(galah_param['sobject_id'], param_joined_valid['sobject_id'])
        # Data consistency and repetition checks
        # print 'N param orig:', len(galah_param), 'uniqu', len(np.unique(galah_param['sobject_id']))
        # print 'N abund orig:', len(abund_param), 'uniqu', len(np.unique(abund_param['sobject_id']))
        # print 'N param:', len(param_joined), 'uniqu', len(np.unique(param_joined['sobject_id']))
        # print 'N spect:', np.sum(idx_spectra_train), 'uniqu', len(np.unique(galah_param['sobject_id'][idx_spectra_train]))

        if squared_components:
            abund_values_train = param_joined[list(np.hstack((sme_abundance, additional_train_feat, squared_train_feat)))].to_pandas().values
        else:
            abund_values_train = param_joined[list(np.hstack((sme_abundance, additional_train_feat)))].to_pandas().values
            abund_values_valid = param_joined_valid[list(np.hstack((sme_abundance, additional_train_feat)))].to_pandas().values

        if normalize_abund_values:
            abund_normalizer_file = 'normalizer_abund_'+'_'.join(elements)+'.pkl'
            n_train_feat = abund_values_train.shape[1]
            if os.path.isfile(abund_normalizer_file):
                print 'Reading normalization parameters'
                train_feat_mean, train_feat_std = joblib.load(abund_normalizer_file)
            else:
                print 'Normalizing input train parameters'
                train_feat_mean = np.zeros(n_train_feat)
                train_feat_std = np.zeros(n_train_feat)
                for i_f in range(n_train_feat):
                    train_feat_mean[i_f] = np.nanmedian(abund_values_train[:, i_f])
                    train_feat_std[i_f] = np.nanstd(abund_values_train[:, i_f])
                joblib.dump([train_feat_mean, train_feat_std], abund_normalizer_file)

            for i_f in range(n_train_feat):
                abund_values_train[:, i_f] = (abund_values_train[:, i_f] - train_feat_mean[i_f]) / train_feat_std[i_f]
                abund_values_valid[:, i_f] = (abund_values_valid[:, i_f] - train_feat_mean[i_f]) / train_feat_std[i_f]
                p_par = list(np.hstack((sme_abundance, additional_train_feat)))
                p_val = abund_values_train[:, i_f]
                plt.hist(p_val, range=(-2.5, 2.5), bins=50)
                plt.savefig('train_norm_'+p_par[i_f]+'.png',dpi=200)
                plt.close()

        n_train_sme = np.sum(idx_spectra_train)
        n_valid_cluster = np.sum(idx_spectra_valid)
        print 'Number of train objects: ' + str(n_train_sme)
        print 'Number of valid objects: ' + str(n_valid_cluster)
        spectral_data_train = spectral_data[idx_spectra_train]
        spectral_data_valid = spectral_data[idx_spectra_valid]

        # set up regularizer if needed
        if use_regularizer:
            # use a combination of l1 and/or l2 regularizer
            # w_reg = regularizers.l1_l2(1e-5, 1e-5)
            # a_reg = regularizers.l1_l2(1e-5, 1e-5)
            w_reg = regularizers.l1(1e-5)
            a_reg = regularizers.l1(1e-5)
        else:
            # default values for Conv1D and Dense layers
            w_reg = None
            a_reg = None

        # ann network - fully connected layers
        ann_input = Input(shape=(spectral_data_train.shape[1], 1), name='Input_'+plot_suffix)
        ann = ann_input
        # first cnn feature extraction layer
        ann = Conv1D(C_f_1, C_k_1, activation=activation_function_c, padding='same', name='C_1', strides=C_s_1,
                     kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        if activation_function_c is None:
            ann = PReLU(name='R_1')(ann)
        ann = MaxPooling1D(P_s_1, padding='same', name='P_1')(ann)
        if dropout_learning_c:
            ann = Dropout(dropout_rate_c, name='D_1')(ann)
        # second cnn feature extraction layer
        if C_f_2 > 0:
            ann = Conv1D(C_f_2, C_k_2, activation=activation_function_c, padding='same', name='C_2', strides=C_s_2,
                         kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
            if activation_function_c is None:
                ann = PReLU(name='R_2')(ann)
            ann = MaxPooling1D(P_s_2, padding='same', name='P_2')(ann)
            if dropout_learning_c:
                ann = Dropout(dropout_rate_c, name='D_2')(ann)
        # third cnn feature extraction layer
        if C_f_3 > 0:
            ann = Conv1D(C_f_3, C_k_3, activation=activation_function_c, padding='same', name='C_3', strides=C_s_3,
                         kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
            if activation_function_c is None:
                ann = PReLU(name='R_3')(ann)
            ann = MaxPooling1D(P_s_3, padding='same', name='P_3')(ann)
            if dropout_learning_c:
                ann = Dropout(dropout_rate_c, name='D_3')(ann)

        # flatter output from convolutional network to the shape useful for fully-connected dense layers
        ann = Flatten(name='Conv_to_Dense')(ann)

        # fully connected layers
        for n_nodes in n_dense_nodes:
            ann = Dense(n_nodes, activation=activation_function, name='Dense_'+str(n_nodes),
                        kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
            # add activation function to the layer
            if n_nodes > 50:
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
        selected_optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        # selected_optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
        # selected_optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, decay=0.0)
        if use_all_nonnan_rows:
            abundance_ann.compile(optimizer=selected_optimizer, loss=custom_error_function_2)
        else:
            abundance_ann.compile(optimizer=selected_optimizer, loss='mse')
        abundance_ann.summary()

        metwork_weights_file = 'ann_network_run{:02.0f}_last.h5'.format(i_run)
        if os.path.isfile(metwork_weights_file):
            save_fits = False
            print 'Reading NN weighs - CAE'
            abundance_ann.load_weights(metwork_weights_file, by_name=True)
        else:
            # define early stopping callback
            earlystop = EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='auto')
            checkpoint = ModelCheckpoint('ann_network_run{:02.0f}'.format(i_run)+'_{epoch:02d}-{loss:.3f}-{val_loss:.3f}.h5',
                                         monitor='val_loss', verbose=0, save_best_only=False,
                                         save_weights_only=True, mode='auto', period=1)
            # fit the NN model
            ann_fit_hist = abundance_ann.fit(spectral_data_train, abund_values_train,
                                             epochs=350,
                                             batch_size=512,
                                             shuffle=True,
                                             callbacks=[earlystop, checkpoint],
                                             validation_split=0.04,
                                             #validation_data=(spectral_data_valid, abund_values_valid),
                                             verbose=2)

            i_best = np.argmin(ann_fit_hist.history['val_loss'])
            plt.plot(ann_fit_hist.history['loss'], label='Train')
            plt.plot(ann_fit_hist.history['val_loss'], label='Validation')
            plt.axvline(i_best, color='black', ls='--', alpha=0.5, label='Best val_loss')
            plt.title('Model accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Loss value')
            plt.ylim(0., 20)
            plt.tight_layout()
            plt.legend()
            plt.savefig('ann_network_loss_run{:02.0f}.png'.format(i_run), dpi=250)
            plt.close()

            last_loss = ann_fit_hist.history['loss'][-1]
            if last_loss > 15:
                # something went wrong, do not evaluate this case
                print 'Final loss was quite large:', last_loss
                save_fits = True
                save_model_last = False
            else:
                save_fits = True
                save_model_last = False

            if save_model_last:
                print 'Saving NN weighs - CAE'
                abundance_ann.save_weights(metwork_weights_file, overwrite=True)

            # recover weights of the best model and compute predictions
            h5_weight_files = glob('ann_network_run{:02.0f}_{:02.0f}-*.h5'.format(i_run, i_best+1))
            if len(h5_weight_files) == 1:
                print 'Restoring epoch {:.0f} with the lowest validation loss ({:.3f}).'.format(i_best + 1, ann_fit_hist.history['val_loss'][i_best])
                abundance_ann.load_weights(h5_weight_files[0], by_name=True)
            else:
                print 'The last model will be used to compute predictions.'
                print 'Glob weights search results:', h5_weight_files

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

        plot_suffix += '_run{:02.0f}'.format(i_run)

        # scatter plot of results to the reference cannon and sme values
        if train_multiple:
            sme_abundances_plot = sme_abundance
        else:
            sme_abundances_plot = [sme_abundance]

        # determine a feature to be used a colour of points
        c_data = np.int64(param_joined['teff_guess'].data)
        c_data_min = np.nanpercentile(c_data, 1)
        c_data_max = np.nanpercentile(c_data, 99)
        # print c_data
        # print c_data_min
        # print c_data_max

        # sme_abundances_plot = np.hstack((sme_abundance, additional_train_feat))
        print 'Plotting graphs'
        for plot_abund in sme_abundances_plot:
            print ' plotting attribute - ' + plot_abund
            elem_plot = plot_abund.split('_')[0]
            # determine number of lines used for this element
            n_lines_element = np.sum(line_list['Element'] == elem_plot.capitalize())
            graphs_title = elem_plot.capitalize() + ' - SME train objects: ' + str(np.sum(np.isfinite(abund_param[plot_abund]))) + ' (lines: ' + str(n_lines_element) + ') - BIAS: {:.2f}   RMSE: {:.2f}'.format(bias(galah_param_complete[plot_abund], galah_param_complete[elem_plot+'_abund_ann']), rmse(galah_param_complete[plot_abund], galah_param_complete[elem_plot+'_abund_ann']))
            plot_range = (np.nanpercentile(abund_param[plot_abund], 1), np.nanpercentile(abund_param[plot_abund], 99))
            # first scatter graph - train points
            plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
            plt.scatter(galah_param_complete[plot_abund], galah_param_complete[elem_plot+'_abund_ann'],
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
            param_plot = plot_param  # .split('_')[0]
            # determine number of lines used for this element
            graphs_title = param_plot.capitalize() + ' - SME train objects: ' + str(np.sum(np.isfinite(abund_param[plot_abund]))) + ' - BIAS: {:.2f}   RMSE: {:.2f}'.format(bias(galah_param_complete[param_plot], galah_param_complete[param_plot+'_ann']), rmse(galah_param_complete[param_plot], galah_param_complete[param_plot+'_ann']))
            plot_range = (np.nanpercentile(abund_param[plot_param], 0.1), np.nanpercentile(abund_param[plot_param], 99.9))
            # first scatter graph - train points
            plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
            plt.scatter(galah_param_complete[param_plot], galah_param_complete[param_plot+'_ann'],
                        lw=0, s=0.4, c='black') #c=c_data, cmap='jet', vmin=c_data_min, vmax=c_data_max)
            plt.title(graphs_title)
            plt.xlabel('SME reference value')
            plt.ylabel('ANN computed value')
            plt.xlim(plot_range)
            plt.ylim(plot_range)
            plt.savefig(param_plot+'_ANN_sme_'+plot_suffix+'.png', dpi=400)
            plt.close()

        if not train_multiple:
            os.chdir('..')

    # also save results (predictions) at the end of every run
    if save_fits:
        fits_out = 'galah_abund_ANN_SME3.0.1_run{:02.0f}.fits'.format(i_run)
        galah_param_complete.write(fits_out)   
