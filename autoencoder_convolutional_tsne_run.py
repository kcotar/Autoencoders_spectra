import imp, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table, Column
from sklearn.preprocessing import StandardScaler


imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir, plot_tsne_results, plot_star_cluster

imp.load_source('tsne', '../tSNE_test/tsne_functions.py')
from tsne import *

# settings
normalize_data = True
remove_flagged = True
perp = 50
theta = 0.3
seed = 35
n_middle = 40
use_bands = [0, 1, 2]
suffix_out = '_noccd4'

# input data
galah_data_input = '/home/klemen/GALAH_data/'
galah_param_file = 'sobject_iraf_52_reduced.fits'
galah_tsne_1_0 = 'tsne_class_1_0.csv'

reduced_spectra_files = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_'+str(n_middle)+'_encoded.csv',
                         'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_'+str(n_middle)+'_encoded.csv',
                         'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_500_'+str(n_middle)+'_encoded.csv',
                         'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF_CAE_16_5_4_16_5_4_8_3_2_AE_250_'+str(n_middle)+'_encoded.csv']

# save results in csv format
csv_out_filename = 'tsne_results_perp'+str(perp)+'_theta'+str(theta)+'_CAE_16_5_4_16_5_4_8_3_2_AE_500_'+str(n_middle)+suffix_out
if remove_flagged:
    csv_out_filename += '_redflagok'
if normalize_data:
    csv_out_filename += '_norm'
csv_out_filename += '.csv'

# read objects parameters
galah_param = Table.read(galah_data_input + galah_param_file)
tsne_class_old = Table.read(galah_data_input + galah_tsne_1_0)3

if remove_flagged:
    idx_ok_lines = galah_param['red_flag'] == 0
    galah_param = galah_param[idx_ok_lines]
    csv_spectra_skip_rows = np.where(np.logical_not(idx_ok_lines))[0]
    print 'Objects removing flagged data: '+str(np.sum(idx_ok_lines))
else:
    csv_spectra_skip_rows = None

if os.path.isfile(csv_out_filename):
    print 'Reading precomputed tSNE results'
    tsne_data = Table.read(csv_out_filename, format='ascii.csv')
else:
    # read spectral data sets
    reduced_data = list([])
    for csv_file in np.array(reduced_spectra_files)[use_bands]:
        print 'Reading reduced spectra file: ' + csv_file
        reduced_data.append(pd.read_csv(csv_file, sep=',', header=None,
                                        na_values='nan', skiprows=csv_spectra_skip_rows).values)

    # merge spectral data sets
    reduced_data = np.hstack(reduced_data)
    print reduced_data.shape

    # normalize data set if requested
    if normalize_data:
        print 'Normalizing data'
        normalizer = StandardScaler()
        reduced_data = normalizer.fit_transform(reduced_data)

    # run tSNE
    tsne_result = bh_tsne(reduced_data, no_dims=2, perplexity=perp, theta=theta, randseed=seed, verbose=True,
                          distance='euclidean', path='/home/klemen/tSNE_test/')
    tsne_ax1, tsne_ax2 = tsne_results_to_columns(tsne_result)

    tsne_data = galah_param['sobject_id', 'galah_id']
    tsne_data.add_column(Column(name='tsne_axis_1', data=tsne_ax1, dtype=np.float64))
    tsne_data.add_column(Column(name='tsne_axis_2', data=tsne_ax2, dtype=np.float64))
    tsne_data.write(csv_out_filename, format='ascii.csv')

# plot tSNE results
move_to_dir(csv_out_filename[:-4])

print 'Plotting final tSNE results'
plot_tsne_results(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  galah_param, ['teff_guess', 'logg_guess', 'feh_guess', 'rv_guess'],
                  suffix='', prefix='', ps=0.2)

# print np.unique(galah_param['red_flag'], return_counts=True)
idx_flagged = galah_param['red_flag'] > 0
plot_tsne_results(tsne_data['tsne_axis_1'][idx_flagged], tsne_data['tsne_axis_2'][idx_flagged],
                  galah_param[idx_flagged], ['red_flag'],
                  suffix='_all', prefix='', ps=0.2)
idx_flagged = np.logical_and(galah_param['red_flag'] > 0, galah_param['red_flag'] < 16)
plot_tsne_results(tsne_data['tsne_axis_1'][idx_flagged], tsne_data['tsne_axis_2'][idx_flagged],
                  galah_param[idx_flagged], ['red_flag'],
                  suffix='_badwvl', prefix='', ps=0.2)
idx_flagged = np.logical_and(galah_param['red_flag'] >= 16, galah_param['red_flag'] < 64)
plot_tsne_results(tsne_data['tsne_axis_1'][idx_flagged], tsne_data['tsne_axis_2'][idx_flagged],
                  galah_param[idx_flagged], ['red_flag'],
                  suffix='_molecfit', prefix='', ps=0.2)

idx_tsne_old = np.in1d(galah_param['sobject_id'], tsne_class_old['sobject_id'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic.png',
                  title=None, ps=0.2)
idx_tsne_old = np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['published_reduced_class_proj1']=='binary'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_binary.png',
                  title=None, ps=0.2)
idx_tsne_old = np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['published_reduced_class_proj1']=='mol. absorption bands'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_molabs.png',
                  title=None, ps=0.2)
idx_tsne_old = np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['published_reduced_class_proj1']=='cool metal-poor giants'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_coolpoor.png',
                  title=None, ps=0.2)
idx_tsne_old =np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['published_reduced_class_proj1']=='hot stars'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_hot.png',
                  title=None, ps=0.2)
idx_tsne_old = np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['published_reduced_class_proj1']=='HaHb emission'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_HaHb.png',
                  title=None, ps=0.2)

idx_tsne_old = np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['unreduced_flag_proj1']=='emission spikes - ccd3 ccd4'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_ccd3ccd4e.png',
                  title=None, ps=0.2)
idx_tsne_old = np.in1d(galah_param['sobject_id'],
                      tsne_class_old['sobject_id'][tsne_class_old['unreduced_flag_proj1']=='emission spike(s) - ccd4'])
plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  tsne_data['tsne_axis_1'][idx_tsne_old], tsne_data['tsne_axis_2'][idx_tsne_old],
                  filename='old_tsne_problematic_ccd4e.png',
                  title=None, ps=0.2)

plot_star_cluster(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                  None, None,
                  filename='all.png',
                  title=None, ps=0.2)
