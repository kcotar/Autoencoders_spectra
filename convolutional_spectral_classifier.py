import imp, os

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.externals import joblib
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
    galah_data_input = '/home/klemen/data4_mount/'
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
    imp.load_source('spectra_collection_functions', '../Carbon-Spectra/spectra_collection_functions.py')
else:
    galah_data_input = '/data4/cotar/'
from helper_functions import move_to_dir
from spectra_collection_functions import read_pkl_spectra, save_pkl_spectra

# tsne_problematic_file ='tsne_class_1_0.csv'  # dr51 classes
# tsne_prob_col = 'published_reduced_class_proj1'
tsne_problematic_file ='dr52_class_joined.csv'  # larger set of dr52 stars
tsne_prob_col = 'dr52_class_reduced'

date_string = '20171111'
galah_param_file = 'sobject_iraf_52_reduced_'+date_string+'.fits'
# cannon3_file = 'sobject_iraf_iDR2_171103_cannon.fits'
cannon1_file = 'sobject_iraf_cannon_1.2.fits'

# # renormed and oversampled set of spectra
# spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.020_lin_renorm_'+date_string+'.pkl',
#                      'galah_dr52_ccd2_5640_5880_wvlstep_0.025_lin_renorm_'+date_string+'.pkl',
#                      'galah_dr52_ccd3_6475_6745_wvlstep_0.030_lin_renorm_'+date_string+'.pkl',
#                      'galah_dr52_ccd4_7700_7895_wvlstep_0.035_lin_renorm_'+date_string+'.pkl']
# original and resampled set of spectra
# spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_'+date_string+'.pkl',
#                      'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_'+date_string+'.pkl',
#                      'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_'+date_string+'.pkl',
#                      'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_'+date_string+'.pkl']
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.020_ext0_renorm_'+date_string+'.pkl',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.025_ext0_renorm_'+date_string+'.pkl',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.030_ext0_renorm_'+date_string+'.pkl',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.035_ext0_renorm_'+date_string+'.pkl']

# reading settings
# spectra_get_cols = [3500, 3500, 3500, 2000]  # normal sampling
spectra_get_cols = [5500, 5500, 5500, 3000]  # resampled sampling

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
dropout_rate = 0.1
activation_function = None  # if set to none defaults to PReLu

# convolution layer 1
C_f_1 = 256  # number of filters
C_k_1 = 11  # size of convolution kernel
C_s_1 = 2  # strides value
P_s_1 = 6  # size of pooling operator
# convolution layer 2
C_f_2 = 128
C_k_2 = 9
C_s_2 = 2
P_s_2 = 6
# convolution layer 3
C_f_3 = 128
C_k_3 = 5
C_s_3 = 2
P_s_3 = 6
n_dense_nodes = [1200, 300, 1]

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------
galah_param = Table.read(galah_data_input + galah_param_file)
tsne_problematic = Table.read(galah_data_input + tsne_problematic_file)
cannon_param = Table.read(galah_data_input + cannon1_file)
tsne_problematic.filled(0)

# determine bad sobjects from parameters
idx_ok = galah_param['red_flag'] == 0
idx_ok = np.logical_and(idx_ok, galah_param['flag_guess'] == 0)
idx_ok = np.logical_and(idx_ok, galah_param['sobject_id'] > 140301000000000)
idx_ok = np.logical_and(idx_ok, galah_param['snr_c2_iraf'] > 30)
sobject_ok_param = galah_param['sobject_id'][idx_ok]

spectral_data = list([])
for i_band in [0, 1, 2, 3]:
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

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectra file: ' + spectra_file
    spectral_data.append(read_pkl_spectra(galah_data_input + spectra_file, read_rows=None, read_cols=range(col_start, col_end)))

spectral_data = np.hstack(spectral_data)
print spectral_data.shape
n_wvl_total = spectral_data.shape[1]

# somehow handle cols with nan values, delete cols or fill in data
idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
n_bad_spectra = len(idx_bad_spectra[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
    spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value
idx_bad_spectra = None

# normalize data set if requested
if normalize_spectra:
    print 'Normalizing data'
    # version 1 - takes too much RAM
    # normalizer = StandardScaler()
    # spectral_data = normalizer.fit_transform(spectral_data)
    # version 2 - consumes less RAM, but it might takes longer time
    for i_c in range(n_wvl_total):
        wvl_col_data = spectral_data[:, i_c]
        spectral_data[:, i_c] = (spectral_data[:, i_c] - np.nanmean(wvl_col_data))/np.nanstd(wvl_col_data)

# prepare spectral data for the further use in the Keras library
spectral_data = np.expand_dims(spectral_data, axis=2)  # not needed in this case, only convolution needs 3D arrays

move_to_dir('Classifier_problematic_dr52')

# --------------------------------------------------------
# ---------------- Train ANN on a train (sub)set of Gregor's published classes
# --------------------------------------------------------
# determine training set
# prob_classes = ['binary']
prob_classes_str = np.unique(tsne_problematic[tsne_prob_col])  # get all unique classes
prob_classes_str = [c for c in prob_classes_str if c != '0']
prob_classes_num = np.int16(np.arange(0, len(prob_classes_str))+1)  # set a numerical value for every wanted class
n_dense_nodes[-1] = len(prob_classes_str) + 1  # +1 for "non-problematic" data

# construct and fill arrays that will be used as a trainig set
idx_spectra_class_train = np.ndarray(len(galah_param), dtype=np.bool)
idx_spectra_class_train.fill(False)  # set that none of the spectra is used for trainig purpose
spectra_class_train = np.ndarray(len(galah_param), dtype=np.int16)
spectra_class_train.fill(0)  # set all spectra to OK == class 0

N_MAX_PER_CLASS = 5000
for i_c in range(len(prob_classes_str)):
    print 'Working on problematic class:', prob_classes_str[i_c]
    sobject_prob = tsne_problematic[tsne_problematic['dr52_class_reduced'] == prob_classes_str[i_c]]['sobject_id']
    idx_spectra_prob = np.in1d(galah_param['sobject_id'], sobject_prob)
    n_prob = np.sum(idx_spectra_prob)
    if n_prob > N_MAX_PER_CLASS:
        # repeat selection with a subset of data
        sobject_prob = sobject_prob[np.int64(np.random.rand(N_MAX_PER_CLASS)*n_prob)]
        idx_spectra_prob = np.in1d(galah_param['sobject_id'], sobject_prob)
    n_prob = np.sum(idx_spectra_prob)
    print ' In train dataset: '+str(n_prob)

    # mark selection into train arrays
    idx_spectra_class_train = np.logical_or(idx_spectra_class_train, idx_spectra_prob)
    spectra_class_train[idx_spectra_prob] = prob_classes_num[i_c]


n_prob_train_total = np.sum(idx_spectra_class_train)
# the next best thing to select kind of non-problematic spectra
# they are not in tnse class and are from dr5.1 release (as we do not have complete dr5.2 classification info yet)
sobject_ok_dr51 = cannon_param[np.in1d(cannon_param['sobject_id'], tsne_problematic['sobject_id'], invert=True)]['sobject_id']
sobject_ok_dr52 = galah_param[np.in1d(galah_param['sobject_id'], sobject_ok_dr51)]['sobject_id']
# also remove spectra with reduction problems, low snr, pilot survey itd
sobject_ok_dr52 = sobject_ok_dr52[np.in1d(sobject_ok_dr52, sobject_ok_param)]
n_ok = len(sobject_ok_dr52)
# select a subset of ok spectra, so they match in number wit problematic spectra
if n_ok > n_prob_train_total:
    sobject_ok = sobject_ok_dr52[np.int64(np.random.random(n_prob_train_total)*n_ok)]
# mark ok selection in train array
idx_subject_ok = np.in1d(galah_param['sobject_id'], sobject_ok)
idx_spectra_class_train = np.logical_or(idx_spectra_class_train, idx_subject_ok)


# encode class values as integers
print 'Encoding labels'
encoder = LabelEncoder()
spectra_class_train_encoded = encoder.fit_transform(spectra_class_train[idx_spectra_class_train])
# convert integers to dummy variables (i.e. one hot encoded)
spectra_class_train_encoded = np_utils.to_categorical(spectra_class_train_encoded)
print spectra_class_train_encoded

n_train_class = np.sum(idx_spectra_class_train)
print 'Number of train objects: ' + str(n_train_class)
spectral_data_train = spectral_data[idx_spectra_class_train]

for f_width in range(1, 6, 1):
    # determine odd sized filter
    C_k_1 = 4*f_width + 1
    C_k_2 = 3*f_width + 1
    C_k_3 = 2*f_width + 1

    # ann network - fully connected layers
    ann_input = Input(shape=(spectral_data_train.shape[1], 1), name='Input_spectra')
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
        if n_nodes > 30:
            ann = Dense(n_nodes, activation=activation_function, name='Dense_'+str(n_nodes))(ann)
            if dropout_learning:
                ann = Dropout(dropout_rate, name='Dropout_' + str(n_nodes))(ann)
            if activation_function is None:
                ann = PReLU(name='PReLU_'+str(n_nodes))(ann)
        else:
            # final layer should produce classifcation: sigmoid(binary) or softmax(multiclass)
            # NOTE ON FINAL ACTIVATION FUNCTION SELECTION !!!!!!!!!!!!!!
            # For a multi-class problem, where you predict 1 of many classes, you use Softmax output.
            # However, in both binary and multi-label classification problems, where multiple classes
            # might be 1 in the output, you use a sigmoid output
            ann = Dense(n_nodes, activation='sigmoid', name='Dense_' + str(n_nodes))(ann)

    abundance_ann = Model(ann_input, ann)
    abundance_ann.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    abundance_ann.summary()

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    # fit the NN model
    abundance_ann.fit(spectral_data_train, spectra_class_train_encoded,
                      epochs=7,
                      batch_size=256,
                      shuffle=True,
                      callbacks=[earlystop],
                      validation_split=0.15,  # percent of the data at the end of the data-set
                      verbose=2)

    # evaluate on all spectra
    print 'Predicting class values from all spectra'
    class_predicted_prob = abundance_ann.predict(spectral_data)

    print 'Classes:', 'OK', prob_classes_str
    joblib.dump(class_predicted_prob, 'multiclass_prob_array_withok_'+str(f_width)+'.pkl')
    # save results of classification

    print class_predicted_prob
    prob_class_final = np.argmax(class_predicted_prob, axis=1)

