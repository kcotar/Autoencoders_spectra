import os, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'

import matplotlib
matplotlib.use('Agg')

#import tensorflow as tf
#tf_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=36,
#                                     inter_op_parallelism_threads=4,
#                                     allow_soft_placement=True)
#session = tf.compat.v1.Session(config=tf_config)
#tf.compat.v1.keras.backend.set_session(session)

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.utils import plot_model
import numpy as np
from astropy.table import Table, vstack, hstack
from sklearn.externals import joblib
from glob import glob
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader
SourceFileLoader('s_collection', '../Carbon-Spectra/spectra_collection_functions.py').load_module()
from s_collection import CollectionParameters, read_pkl_spectra, save_pkl_spectra

# get arguments if they were added to the procedure
in_args = sys.argv
n_l_e = 10
n_epoch = 20
optimizer = 'RMSprop'
if len(in_args) >= 2:
    n_l_e = int(in_args[1])
if len(in_args) >= 3:
    n_epoch = int(in_args[2])
if len(in_args) >= 4:
    optimizer = str(in_args[3])

print('Used input parameters:', n_l_e, n_epoch, optimizer)

print('Reading data sets')
galah_data_input = '/shared/ebla/cotar/'
galah_spectra_input = '/shared/data-camelot/cotar/'
date_string = '20190801'
line_file = 'GALAH_Cannon_linelist_newer.csv'
galah_param_file = 'sobject_iraf_53_reduced_'+date_string+'.fits'
galah_param = Table.read(galah_data_input + galah_param_file)
sme_param_file = 'GALAH_iDR3_v1_181221.fits'
sme_param = Table.read(galah_data_input + sme_param_file)
galah_tsne_flag_old = Table.read(galah_data_input + 'tsne_class_1_0.csv')  # older classification
galah_tsne_flag_new = Table.read(galah_data_input + 'tsne_classification_dr52_2018_04_09.csv')  # never classification with re-reduced spectra and omitted ccd4
tsne_class_string = 'tsne_class'
galah_tsne_flag_old['published_reduced_class_proj1'].name = tsne_class_string

galah_tsne_flag = vstack([galah_tsne_flag_old['sobject_id', tsne_class_string], galah_tsne_flag_new['sobject_id', tsne_class_string]])
print('t-SNE objects:', len(np.unique(galah_tsne_flag_old['sobject_id'])), len(np.unique(galah_tsne_flag_new['sobject_id'])), len(np.unique(galah_tsne_flag['sobject_id'])))

galah_tsne_flag = galah_tsne_flag.filled()

# remove peculiar flagged stars that are actually wanted in our case
for no_use_class in ['cool metal-poor giants', 'hot stars', 'mol. absorption bands', 'MAB', 'CMP', 'HOT', 'HFR', 'NTR']:
    galah_tsne_flag = galah_tsne_flag[galah_tsne_flag[tsne_class_string] != no_use_class]

idx_get_spectra = galah_param['snr_c2_guess'] > 20.
idx_get_spectra = np.logical_and(idx_get_spectra,
                                 np.in1d(galah_param['sobject_id'],
                                         galah_tsne_flag['sobject_id'], invert=True))
idx_get_spectra = np.logical_and(idx_get_spectra,
                                 np.in1d(galah_param['sobject_id'],
                                         sme_param[sme_param['flag_sp'] < 16]['sobject_id'], invert=False))
idx_get_spectra = np.logical_and(idx_get_spectra,
                                 galah_param['red_flag'] == 0)
idx_get_spectra = np.where(idx_get_spectra)[0]

# omit some classes for plotting purposes
for no_use_class in ['N/A', 'problematic', 'SPI', 'TAB', 'TEM']:
    galah_tsne_flag = galah_tsne_flag[galah_tsne_flag[tsne_class_string] != no_use_class]

#min_wvl = np.array([4725, 5665, 6485, 7700])
#max_wvl = np.array([4895, 5865, 6725, 7875])

min_wvl = np.array([4710, 5640, 6475, 7700])
max_wvl = np.array([4910, 5880, 6745, 7895])

spectra_file_list = ['galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd2_5640_5880_wvlstep_0.050_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd4_7700_7895_wvlstep_0.070_ext4_'+date_string+'.pkl']

for read_ccd in [2, 0]:  #:range(len(spectra_file_list)):
    # parse resampling settings from filename
    read_pkl_file = galah_spectra_input + spectra_file_list[read_ccd]
    csv_param = CollectionParameters(read_pkl_file)
    wvl_values = csv_param.get_wvl_values()
    wvl_limits = csv_param.get_wvl_range()
    ccd_number = int(csv_param.get_ccd())

    idx_read = np.where(np.logical_and(wvl_values >= min_wvl[read_ccd], wvl_values <= max_wvl[read_ccd]))[0]

    suffix = '_{:.0f}D_{:.0f}epoch_4layer'.format(n_l_e, n_epoch) + '_' + optimizer
    out_dir = galah_spectra_input+'Autoencoder_dense_test_complex_ccd{:01.0f}_prelu'.format(read_ccd+1)+suffix
    os.system('mkdir '+out_dir)
    os.chdir(out_dir)

    # print(idx_read
    wvl_read = wvl_values[idx_read]
    n_wvl = len(wvl_read)
    
    n_obj = len(idx_get_spectra)
    print('Reading pkl spectra with:', n_wvl, n_obj)
    spectral_data_all = read_pkl_spectra(read_pkl_file, read_cols=idx_read)
    
    idx_bad = np.logical_not(np.isfinite(spectral_data_all))
    if np.sum(idx_bad) > 0:
        print('Correcting bad values:', np.sum(idx_bad))
        spectral_data_all[idx_bad] = 1.
    
    # idx_bad = spectral_data_all > 1.2
    # if np.sum(idx_bad) > 0:
    #     print('spectral_data_all large/low values:', np.sum(idx_bad))
    #     spectral_data_all[idx_bad] = 1.2
    
    idx_bad = spectral_data_all < 0
    if np.sum(idx_bad) > 0:
        print('spectral_data_all large/low values:', np.sum(idx_bad))
        spectral_data_all[idx_bad] = 0
    
    # normalize data
    spectral_data_all = 1. - spectral_data_all
    
    spectral_data = spectral_data_all[idx_get_spectra, :]
    print(spectral_data.shape)  # trainnig set of spectra
    print(spectral_data_all.shape)  # complete set of observed and reduced spectra

    activation = None  # PReLU if set to None
    dropout_rate = 0  # from 0 to 1
    decoded_layer_name = 'encoded'
    n_wvl = spectral_data.shape[1]

    # compute number of nodes in every connected layer
    n_l_1 = int(n_wvl * 0.75)
    n_l_2 = int(n_wvl * 0.50)
    n_l_3 = 0  # int(n_wvl * 0.40)
    n_l_4 = 0  # int(n_wvl * 0.20)
    n_l_5 = int(n_wvl * 0.25)
    n_l_6 = int(n_wvl * 0.10)

    # create ann model
    autoencoder = Sequential()

    if n_l_1 > 0:
        autoencoder.add(Dense(n_l_1, input_shape=(n_wvl,), activation=activation, name='E_1'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_1'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_1'))

    if n_l_2 > 0:
        autoencoder.add(Dense(n_l_2, activation=activation, name='E_2'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_2'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_2'))

    if n_l_3 > 0:
        autoencoder.add(Dense(n_l_3, activation=activation, name='E_3'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_3'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_3'))

    if n_l_4 > 0:
        autoencoder.add(Dense(n_l_4, activation=activation, name='E_4'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_4'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_4'))

    if n_l_5 > 0:
        autoencoder.add(Dense(n_l_5, activation=activation, name='E_5'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_5'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_5'))

    if n_l_6 > 0:
        autoencoder.add(Dense(n_l_6, activation=activation, name='E_6'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_6'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_6'))

    autoencoder.add(Dense(n_l_e, activation=activation, name=decoded_layer_name))
    if activation is None:
        autoencoder.add(PReLU(name='PR_7'))

    if n_l_6 > 0:
        autoencoder.add(Dense(n_l_6, activation=activation, name='D_1'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_8'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_8'))

    if n_l_5 > 0:
        autoencoder.add(Dense(n_l_5, activation=activation, name='D_2'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_9'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_9'))

    if n_l_4 > 0:
        autoencoder.add(Dense(n_l_4, activation=activation, name='D_3'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_10'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_10'))

    if n_l_3 > 0:
        autoencoder.add(Dense(n_l_3, activation=activation, name='D_4'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_11'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_11'))

    if n_l_2 > 0:
        autoencoder.add(Dense(n_l_2, activation=activation, name='D_5'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_12'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_12'))

    if n_l_1 > 0:
        autoencoder.add(Dense(n_l_1, activation=activation, name='D_6'))
        if dropout_rate > 0:
            autoencoder.add(Dropout(dropout_rate, name='DO_13'))
        if activation is None:
            autoencoder.add(PReLU(name='PR_13'))

    autoencoder.add(Dense(n_wvl, activation='linear', name='recreated'))
    autoencoder.summary()

    # Visualize network architecture and save the visualization as a file
    plot_model(autoencoder, show_layer_names=True, show_shapes=True, to_file='ann_network_structure_a.pdf')
    plot_model(autoencoder, show_layer_names=True, show_shapes=True, to_file='ann_network_structure_a.png', dpi=300)
    # plot_model(autoencoder, show_layer_names=True, show_shapes=False, to_file='ann_network_structure_b.pdf')
    # plot_model(autoencoder, show_layer_names=False, show_shapes=False, to_file='ann_network_structure_c.pdf')

    # model file handling
    out_model_file = 'model_weights.h5'
    if os.path.isfile(out_model_file):
        autoencoder.load_weights(out_model_file, by_name=True)
    else:
        autoencoder.compile(optimizer=optimizer, loss='mse')
        checkpoint = ModelCheckpoint('ann_model_run_{epoch:03d}-{loss:.4f}-{val_loss:.4f}.h5',
                                     monitor='val_loss', verbose=0, save_best_only=False,
                                     save_weights_only=True, mode='auto', period=1)
        ann_fit_hist = autoencoder.fit(spectral_data, spectral_data,
                                       epochs=n_epoch,
                                       callbacks=[checkpoint],
                                       shuffle=True,
                                       batch_size=20000,
                                       validation_split=0.05,
                                       verbose=2)

        i_best = np.argmin(ann_fit_hist.history['loss'])
        # i_best = np.argmin(ann_fit_hist.history['val_loss'])
        plt.plot(ann_fit_hist.history['loss'], label='Train')
        plt.plot(ann_fit_hist.history['val_loss'], label='Validation')
        plt.axvline(np.arange(n_epoch)[i_best], ls='--', color='black', alpha=0.5)
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.ylim(0, 0.005)
        plt.tight_layout()
        plt.legend()
        plt.savefig('ann_network_loss.png', dpi=250)
        plt.close()

        # save loss and val_loss in textual format
        loss_combined = np.vstack((ann_fit_hist.history['loss'], ann_fit_hist.history['val_loss'])).T
        np.savetxt('ann_network_loss.txt', loss_combined)

        # recover weights of the best model and compute predictions
        h5_weight_files = glob('ann_model_run_{:03.0f}-*-*.h5'.format(i_best+1))
        if len(h5_weight_files) == 1:
            print('Restoring epoch {:.0f} with the lowest loss ({:.4f}).'.format(i_best + 1, ann_fit_hist.history['val_loss'][i_best]))
            autoencoder.load_weights(h5_weight_files[0], by_name=True)
            # delete all other h5 files that were not used and may occupy a lot of hdd space
            for h5_file in glob('ann_model_run_*-*-*.h5'):
                os.system('rm '+h5_file)

        print('Saving final selected/best model/weights')
        autoencoder.save_weights(out_model_file)

    print('Predicting values')
    processed_data_all = autoencoder.predict(spectral_data_all, verbose=2, batch_size=32768)

    print('Saving ANN median like spectra')

    pkl_out_file = spectra_file_list[read_ccd][:-4] + '_ann_median.pkl'
    save_pkl_spectra(1. - processed_data_all, galah_spectra_input + pkl_out_file)
    
    # denormalize data
    print(processed_data_all)
    print(processed_data_all.shape)

    def plot_selected_spectrum(i_r):
        i_o = int(np.floor(i_r))
        s_id = galah_param['sobject_id'][i_o]
        idx_star = np.where(sme_param['sobject_id'] == s_id)[0]
        sme_star = sme_param[idx_star]
        plt.figure(figsize=(20, 5))
        plt.plot(wvl_values[idx_read], 1. - spectral_data_all[i_o, :], color='black', lw=0.8)
        plt.plot(wvl_values[idx_read], 1. - processed_data_all[i_o, :], color='blue', lw=0.8)
        plt.xlim(wvl_values[idx_read][0], wvl_values[idx_read][-1])
        plt.ylim(0.3, 1.1)
        if len(sme_star) == 1:
            plot_title = 'red_f: {:.0f}, sme_f: {:.0f}, teff: {:.0f}, feh: {:.2f}, logg: {:.2f}, vbroad: {:.1f}'.format(np.float(sme_star['red_flag']), np.float(sme_star['flag_sp']), np.float(sme_star['teff']), np.float(sme_star['fe_h']), np.float(sme_star['logg']), np.float(sme_star['vbroad']))
            plt.title(plot_title)
        plt.tight_layout()
        plt.savefig('s_{:06.0f}.png'.format(i_r), dpi=300)
        plt.close()
    
    os.system('mkdir random')
    os.chdir('random')
    print('Plotting random spectra')
    for i_r in np.random.random(250)*n_obj:
        plot_selected_spectrum(i_r)
    os.chdir('..')
    
    os.system('mkdir tsne_flag')
    os.chdir('tsne_flag')
    print('Plotting tsne flaged spectra')
    for s_id in np.random.choice(galah_tsne_flag['sobject_id'], 250, replace=False):
        i_o = np.where(galah_param['sobject_id'] == s_id)[0]
        if len(i_o) != 1:
            continue
        else:
            i_o = i_o[0]
        plot_selected_spectrum(i_o)
    os.chdir('..')

    os.system('mkdir tsne_HaHb')
    os.chdir('tsne_HaHb')
    print('Plotting potential HaHb emitters')
    for s_id in np.random.choice(galah_tsne_flag['sobject_id'][np.logical_or(galah_tsne_flag[tsne_class_string] == 'HaHb emission',galah_tsne_flag[tsne_class_string] == 'HAE_HBE')], 250, replace=False):
        i_o = np.where(galah_param['sobject_id'] == s_id)[0]
        if len(i_o) != 1:
            continue
        else:
            i_o = i_o[0]
        plot_selected_spectrum(i_o)
    os.chdir('..')

    
    os.system('mkdir large_dif')
    os.chdir('large_dif')
    print('Plotting strange spectra')
    spectrum_diff = np.nansum(spectral_data_all - processed_data_all, axis=1)
    for i_o in np.argsort(spectrum_diff)[::-1][:250]:
        plot_selected_spectrum(i_o)
    os.chdir('..')

    os.system('mkdir fast_rot')
    os.chdir('fast_rot')
    print('Plotting fast rotators')
    for s_id in np.random.choice(sme_param[sme_param['vbroad'] >= 22]['sobject_id'], 250, replace=False):
        i_o = np.where(galah_param['sobject_id'] == s_id)[0]
        if len(i_o) != 1:
            continue
        else:
            i_o = i_o[0]
        plot_selected_spectrum(i_o)
    os.chdir('..')

    os.system('mkdir hot_stars')
    os.chdir('hot_stars')
    print('Plotting hot stars')
    for s_id in np.random.choice(sme_param[sme_param['teff'] >= 6800]['sobject_id'], 250, replace=False):
        i_o = np.where(galah_param['sobject_id'] == s_id)[0]
        if len(i_o) != 1:
            continue
        else:
            i_o = i_o[0]
        plot_selected_spectrum(i_o)
    os.chdir('..')



    # model that will output decode values
    print('Predicting encoded values')
    autoencoder_encoded = Model(inputs=autoencoder.input,
                                outputs=autoencoder.get_layer(decoded_layer_name).output)
    autoencoder_encoded.summary()

    plt.plot(autoencoder.predict(np.full((1, n_wvl), 0.), verbose=2)[0])
    plt.plot(autoencoder.predict(np.full((1, n_wvl), 0.1), verbose=2)[0])
    plt.plot(autoencoder.predict(np.full((1, n_wvl), -0.1), verbose=2)[0])
    plt.show()
    plt.close()
    # raise SystemExit

    print(autoencoder.get_layer(decoded_layer_name).get_weights())
    print(autoencoder_encoded.get_layer(decoded_layer_name).get_weights())

    print(' -> Getting predictions from encoded layer')
    decoded_spectra = autoencoder_encoded.predict(spectral_data, verbose=2, batch_size=32768)
    decoded_spectra_all = autoencoder_encoded.predict(spectral_data_all, verbose=2, batch_size=32768)
    print(' -> Saving reduced and encoded spectra')
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
    
    os.chdir('..')
