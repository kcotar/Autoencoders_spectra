import imp, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table, Column
from sklearn.preprocessing import StandardScaler


imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

imp.load_source('tsne', '../tSNE_test/tsne_functions.py')
from tsne import *

# settings
normalize_data = False
perp = 50
theta = 0.3
seed = 35

# input data
galah_data_input = '/home/klemen/GALAH_data/'
galah_param_file = 'sobject_iraf_52_reduced.fits'
reduced_spectra_files = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_40_encoded.csv',
                         'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_40_encoded.csv',
                         'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_40_encoded.csv',
                         'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_250_40_encoded.csv']

# read objects parameters
galah_param = Table.read(galah_data_input + galah_param_file)

# read spectral data sets
reduced_data = list([])
for csv_file in reduced_spectra_files:
    print 'Reading reduced spectra file: ' + csv_file
    reduced_data.append(pd.read_csv(csv_file, sep=',', header=None, na_values='nan').values)

# merge spectral data sets
reduced_data = np.hstack(reduced_data)
print reduced_data.shape

if normalize_data:
    normalizer = StandardScaler()
    reduced_data = normalizer.fit_transform(reduced_data)

# run tSNE
tsne_result = bh_tsne(reduced_data, no_dims=2, perplexity=perp, theta=theta, randseed=seed, verbose=True,
                      distance='euclidean', path='/home/klemen/tSNE_test/')
tsne_ax1, tsne_ax2 = tsne_results_to_columns(tsne_result)

# save results in csv format
csv_out_filename = 'tsne_results_perp'+str(perp)+'_theta'+str(theta)+'_CAE_16_5_4_16_5_4_8_3_2_AE_500_40'
if normalize_data:
    csv_out_filename += '_norm'
csv_out_filename += '.csv'

if os.path.isfile(csv_out_filename):
    tsne_data = Table.read(csv_out_filename, format='ascii.csv')
else:
    tsne_data = galah_param['sobject_id', 'galah_id']
    tsne_data.add_column(Column(name='tsne_axis_1', data=tsne_ax1, dtype=np.float64))
    tsne_data.add_column(Column(name='tsne_axis_2', data=tsne_ax2, dtype=np.float64))
    tsne_data.write(csv_out_filename, format='ascii.csv')

# plot tsne results

