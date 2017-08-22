import imp, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table, Column
from sklearn.preprocessing import StandardScaler

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

imp.load_source('spectra', '../Carbon-Spectra/helper_functions.py')
from spectra import get_spectra_dr52

# settings
normalize_data = True
plot_spectra = True
use_bands = [0, 1, 2]

# input data
spectra_dir_2 = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_data_input = '/home/klemen/GALAH_data/'
galah_param_file = 'sobject_iraf_52_reduced.fits'
reduced_spectra_files = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_25_encoded.csv',
                         'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_25_encoded.csv',
                         'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_25_encoded.csv',
                         'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_250_25_encoded.csv']

# read objects parameters
galah_param = Table.read(galah_data_input + galah_param_file)

# read spectral data sets
reduced_data = list([])
for csv_file in np.array(reduced_spectra_files)[use_bands]:
    print 'Reading reduced spectra file: ' + csv_file
    reduced_data.append(pd.read_csv(csv_file, sep=',', header=None, na_values='nan').values)

# merge spectral data sets
reduced_data = np.hstack(reduced_data)
print reduced_data.shape

# normalize data set if requested
if normalize_data:
    print 'Normalizing data'
    normalizer = StandardScaler()
    reduced_data = normalizer.fit_transform(reduced_data)

# find spectra that are most simillar to so randomlly selected spectra
idx_random = np.int64(np.random.random(50)*len(galah_param))
for idx in idx_random:
    ref_data = reduced_data[idx, :]
    similarity = np.sqrt(np.sum((reduced_data - ref_data)**2, axis=1))
    idx_similar = np.argsort(similarity)#[1:]
    s_ids = galah_param['sobject_id'][idx_similar[:20]]
    print ','.join([str(s) for s in s_ids])
    if plot_spectra:
        for s_id in s_ids:
            s2, w2 = get_spectra_dr52(str(s_id), bands=[3], root=spectra_dir_2, individual=False, extension=4)
            if len(s2) == 0:
                continue
            plt.plot(w2[0], s2[0])
        plt.ylim((0, 1.2))
        plt.show()
