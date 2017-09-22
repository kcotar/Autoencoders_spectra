import imp, os

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten
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
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
    galah_data_input = '/home/klemen/GALAH_data/'
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
else:
    galah_data_input = '/data4/cotar/'
from helper_functions import move_to_dir


tsne_problematic_file ='tsne_class_1_0.csv'
galah_param_file = 'sobject_iraf_52_reduced.fits'
abund_param_file = 'sobject_iraf_cannon2.1.7.fits'
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF.csv',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF.csv',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF.csv',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF.csv']

# reading settings
spectra_get_cols = [2000, 4000, 2000, 2016]

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
normalize_abund_values = True
dropout_learning = True
activation_function = None  # if set to none defaults to PReLu

# convolution layer 1
C_f_1 = 32  # number of filters
C_k_1 = 9  # size of convolution kernel
C_s_1 = 2  # strides value
P_s_1 = 4  # size of pooling operator
# convolution layer 2
C_f_2 = 16
C_k_2 = 7
C_s_2 = 1
P_s_2 = 4
n_dense_nodes = [1000, 350]

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------
galah_param = Table.read(galah_data_input + galah_param_file)
tsne_problematic = Table.read(galah_data_input + tsne_problematic_file)

spectral_data = list([])
for i_band in [0, 2]:
    spectra_file = spectra_file_list[i_band]
    # data availability check
    if not os.path.isfile(galah_data_input + spectra_file):
        print 'Spectral file not found: ' + spectra_file
        continue

    print 'Working on ' + spectra_file
    # determine cols to be read from csv file
    spectra_file_split = spectra_file.split('_')
    wlv_begin = float(spectra_file_split[3])
    wlv_end = float(spectra_file_split[4])
    wlv_step = float(spectra_file_split[6])
    wvl_range = np.arange(wlv_begin, wlv_end, wlv_step)
    n_wvl = len(wvl_range)
    # select the middle portion of the spectra, that should be free of nan values
    col_start = int(n_wvl / 2. - spectra_get_cols[i_band] / 2.)
    col_end = int(col_start + spectra_get_cols[i_band])

    suffix = ''
    if snr_cut and not limited_rows:
        snr_percentile = 5.
        snr_col = 'snr_c' + str(i_band + 1) + '_guess'  # as ccd numbering starts with 1
        print 'Cutting off {:.1f}% of spectra with low snr value ('.format(snr_percentile) + snr_col + ').'
        snr_percentile_value = np.percentile(galah_param[snr_col], snr_percentile)
        skip_rows = np.where(galah_param[snr_col] < snr_percentile_value)[0]
        suffix += '_snrcut'
    elif limited_rows:
        n_first_lines = 7500
        print 'Only limited number (' + str(n_first_lines) + ') of spectra rows will be read'
        skip_rows = np.arange(n_first_lines, len(galah_param))
        suffix += '_subset'
    else:
        skip_rows = None

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectra file: ' + spectra_file
    spectral_data.append(pd.read_csv(galah_data_input + spectra_file, sep=',', header=None,
                                     na_values='nan', usecols=range(col_start, col_end), skiprows=skip_rows).values)

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
spectral_data = np.expand_dims(spectral_data, axis=2)  # not needed in this case, only convolution needs 3D arrays

move_to_dir('Classifier_problematic')

# --------------------------------------------------------
# ---------------- Train ANN on a train set of abundances
# --------------------------------------------------------
# determine training set
prob_classes = ['binary']
for prob_class in prob_classes:
    print 'Working on problematic class: ' + prob_class

    sobject_prob = tsne_problematic[tsne_problematic['published_reduced_class_proj1'] == prob_class]['sobject_id']
    n_prob = len(sobject_prob)
    print 'Problematic in train dataset: '+str(n_prob)

    # the next best thing to kind of non-problematic spectra
    sobject_ok = galah_param[np.in1d(galah_param['sobject_id'], tsne_problematic['sobject_id'], invert=True)]['sobject_id']
    n_ok = len(sobject_ok)
    # select random spectra from the ok set
    # number of the is equal to problematic spectra
    if n_ok > n_prob:
        sobject_ok = sobject_ok[np.int64(np.random.random(n_prob)*n_ok)]

    # sort and get unique spectra at the same time
    sobject_train = np.unique(np.hstack((sobject_prob, sobject_ok)))

    output_col = prob_class+'_ann'

    # TODO
    # TODO
    # TODO

    # create a subset of spectra to be train on the sme values
    param_joined = join(galah_param['sobject_id', 'teff_guess', 'feh_guess', 'logg_guess'],
                        abund_param['sobject_id', sme_abundance][np.isfinite(abund_param[sme_abundance])],
                        keys='sobject_id', join_type='inner')
    idx_spectra_train = np.in1d(galah_param['sobject_id'], param_joined['sobject_id'])

    abund_values_train = param_joined[sme_abundance, 'teff_guess', 'feh_guess', 'logg_guess'].to_pandas().values

    if normalize_abund_values:
        normalizer_outptu = StandardScaler()
        abund_values_train = normalizer_outptu.fit_transform(abund_values_train)

    n_train_sme = np.sum(idx_spectra_train)
    print 'Number of train objects: ' + str(n_train_sme)
    spectral_data_train = spectral_data[idx_spectra_train]

    # ann network - fully connected layers
    ann_input = Input(shape=(spectral_data_train.shape[1], 1), name='Input_'+sme_abundance)
    ann = ann_input

    ann = Conv1D(C_f_1, C_k_1, activation=None, padding='same', name='C_1', strides=C_s_1)(ann)
    ann = PReLU(name='R_1')(ann)
    ann = MaxPooling1D(P_s_1, padding='same', name='P_1')(ann)
    if C_f_2 > 0:
        ann = Conv1D(C_f_2, C_k_2, activation=None, padding='same', name='C_2', strides=C_s_2)(ann)
        ann = PReLU(name='R_2')(ann)
        ann = MaxPooling1D(P_s_2, padding='same', name='P_2')(ann)

    # flatter output from convolutional network to the shape useful for fully-connected dense layers
    ann = Flatten(name='Conv_to_Dense')(ann)

    # fully connected layers
    for n_nodes in n_dense_nodes:
        ann = Dense(n_nodes, activation=activation_function, name='Dense_'+str(n_nodes))(ann)
        if activation_function is None:
            ann = PReLU(name='PReLU_'+str(n_nodes))(ann)
        if dropout_learning and n_nodes > 1:
            ann = Dropout(0.2, name='Dropout_'+str(n_nodes))(ann)

    abundance_ann = Model(ann_input, ann)
    abundance_ann.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    abundance_ann.summary()

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    # fit the NN model
    abundance_ann.fit(spectral_data_train, abund_values_train,
                      epochs=125,
                      batch_size=128,
                      shuffle=True,
                      callbacks=[earlystop],
                      validation_split=0.1,
                      verbose=1)

    # evaluate on all spectra
    print 'Predicting abundance values from spectra'
    abundance_predicted = abundance_ann.predict(spectral_data)
    if normalize_abund_values:
        abundance_predicted = normalizer_outptu.inverse_transform(abundance_predicted)

    # add it to the final table
    galah_param_complete[output_col] = abundance_predicted[:, 0]

    if activation_function is None:
        plot_suffix = 'prelu'
    else:
        plot_suffix = activation_function

    # # scatter plot of results to the reference cannon and sme values
    # print 'Plotting graphs'
    # graphs_title = element.capitalize() + ' - number of SME learning objects is ' + str(n_train_sme)
    # plot_range = (np.nanmin(abund_param[sme_abundance]), np.nanmax(abund_param[sme_abundance]))
    # # first scatter graph - train points
    # plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
    # plt.scatter(galah_param_complete[element+'_abund_sme'], galah_param_complete[output_col],
    #             lw=0, s=0.1, alpha=0.4, c='black')
    # plt.title(graphs_title)
    # plt.xlabel('SME reference value')
    # plt.ylabel('ANN computed value')
    # plt.xlim(plot_range)
    # plt.ylim(plot_range)
    # plt.savefig(element+'_ANN_sme_'+plot_suffix+'.png', dpi=400)
    # plt.close()
    # # second graph - cannon points
    # plt.scatter(galah_param_complete[element + '_abund_cannon'], galah_param_complete[output_col],
    #             lw=0, s=0.1, alpha=0.2, c='black')
    # plt.title(graphs_title)
    # plt.xlabel('CANNON reference value')
    # plt.ylabel('ANN computed value')
    # plt.xlim(plot_range)
    # plt.ylim(plot_range)
    # plt.savefig(element + '_ANN_cannon_'+plot_suffix+'.png', dpi=400)
    # plt.close()

