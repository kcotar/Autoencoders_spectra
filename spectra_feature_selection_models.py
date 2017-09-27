import os, socket, sys

from astropy.table import Table, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# feature selection imports
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

pc_name = socket.gethostname()

# input data
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
    galah_data_input = '/home/klemen/GALAH_data/'
else:
    galah_data_input = '/data4/cotar/'

galah_param_file = 'sobject_iraf_52_reduced.fits'
galah_param_file_old = 'sobject_iraf_general_1.1.fits'
galah_problematic_file = 'tsne_class_1_0.csv'
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF.csv',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF.csv',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF.csv',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF.csv']

# --------------------------------------------------------
# ---------------- FUNCTIONS USED ------------------------
# --------------------------------------------------------


def plot_selected_ranges(wvl_selected, wvl_range, wvl_step, png_name):
    for wvl in wvl_selected:
        plt.axvspan(wvl-wvl_step/2., wvl+wvl_step/2., color='blue', alpha=0.75)
    plt.xlim(wvl_range)
    plt.ylim((0.4, 1.2))
    plt.tight_layout()
    plt.savefig(png_name, dpi=400)
    plt.close()

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------

# algorithm settings
output_results = True
output_plots = True
snr_cut = False  # do not use in this configuration

# reading settings
spectra_get_cols = [4000, 4000, 200, 2016]

# --------------------------------------------------------
# ---------------- REFERENCE DATA ------------------------
# --------------------------------------------------------
print 'Reading parameter files'
galah_param = Table.read(galah_data_input + galah_param_file)
galah_param_old = Table.read(galah_data_input + galah_param_file_old)
galah_prob = Table.read(galah_data_input + galah_problematic_file)
galah_prob.filled(0)

# determine cross-section between old and new dataset (dr51 and dr52)
galah_common_sobject = join(galah_param, galah_param_old, keys='sobject_id', join_type='inner')['sobject_id']

# reassign problematic flags from strings to integers/numbers
use_classes = ['hot stars', 'cool metal-poor giants', 'binary', 'mol. absorption bands', 'HaHb emission']
remove_classes = ['0', 'problematic']
# first remove problematic and unknown classes from the list
print 'Removing unknown/problematic objects'
for i_r in range(len(remove_classes)):
    sobj_rem = galah_prob[galah_prob['published_reduced_class_proj1'] == remove_classes[i_r]]['sobject_id']
    if len(sobj_rem) > 0:
        idx_use = np.in1d(galah_common_sobject, sobj_rem, invert=True)
        galah_common_sobject = galah_common_sobject[idx_use]
# assign classes to the spectra of objects
print 'Determining numeric classes'
galah_prob_flag = np.zeros(len(galah_common_sobject))
for i_u in range(len(use_classes)):
    sobj_class = galah_prob[galah_prob['published_reduced_class_proj1'] == use_classes[i_u]]['sobject_id']
    if len(sobj_class) > 0:
        idx_class = np.in1d(galah_common_sobject, sobj_class, invert=False)
        galah_prob_flag[idx_class] = i_u+1

# determine spectral data rows to be read
skip_rows = np.where(np.in1d(galah_param['sobject_id'], galah_common_sobject, invert=True))[0]
print 'Rows to be read: '+str(len(galah_param)-len(skip_rows))

# --------------------------------------------------------
# ---------------- READ SPECTRA --------------------------
# --------------------------------------------------------

input_arguments = sys.argv
if len(input_arguments) > 1:
    read_galah_bands = np.int8(input_arguments[1].split(','))
    print 'Manual bands: '+str(read_galah_bands)
else:
    read_galah_bands = [0, 1, 2, 3]

spectral_data = list([])
for i_band in read_galah_bands:
    suffix = ''

    spectra_file = spectra_file_list[i_band]
    # data availability check
    if not os.path.isfile(galah_data_input + spectra_file):
        print 'Spectral file not found: '+spectra_file
        continue

    print 'Working on '+spectra_file
    # determine cols to be read from csv file
    spectra_file_split = spectra_file.split('_')
    wlv_begin = float(spectra_file_split[3])
    wlv_end = float(spectra_file_split[4])
    wlv_step = float(spectra_file_split[6])
    wvl_range = np.arange(wlv_begin, wlv_end, wlv_step)
    n_wvl = len(wvl_range)
    # select the middle portion of the spectra, that should be free of nan values
    col_start = int(n_wvl/2. - spectra_get_cols[i_band]/2.)
    col_end = int(col_start + spectra_get_cols[i_band])

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectral data from {:07.2f} to {:07.2f}'.format(wvl_range[col_start], wvl_range[col_end])
    spectral_data.append(pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan', dtype=np.float16,  # float16 is more than enough
                                     usecols=range(col_start, col_end), skiprows=skip_rows).values)

spectral_data = np.hstack(spectral_data)
print spectral_data.shape
n_wvl_total = spectral_data.shape[1]

# somehow handle cols with nan values, delete cols or fill in data
idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
n_bad_spectra = len(idx_bad_spectra[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
    spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

# --------------------------------------------------------
# ---------------- FEATURE SELECTION PROCESS -------------
# --------------------------------------------------------

# Create the RFE object and rank flux at each pixel
n_features_out = 100
print 'Performing RFE feature selection'
svc_classifier = SVC(kernel="linear", C=1)
rfe_selection = RFE(estimator=svc_classifier, n_features_to_select=n_features_out, step=0.05, verbose=1)
rfe_selection.fit(spectral_data, galah_prob_flag)
wvl_ranking_rfe = rfe_selection.ranking_  # selected features are marked with 1, others that were eliminated are increasing in number
print wvl_ranking_rfe
wvl_selected_rfe = wvl_range[wvl_ranking_rfe == 1]
plot_selected_ranges(wvl_selected_rfe, (wlv_begin, wlv_end), wlv_step, 'rfe_selected.png')

print 'Performing KBest feature selection.'
kbest_selection = SelectKBest(chi2, k=n_features_out)
kbest_selection.fit(spectral_data, galah_prob_flag)
wvl_scores_kbest = rfe_selection.scores_  # higher score -> better feture
print wvl_scores_kbest
wvl_selected_kbest = wvl_range[np.argsort(wvl_scores_kbest)[-n_features_out:]]  # last because they are ordered in increasing order
plot_selected_ranges(wvl_selected_kbest, (wlv_begin, wlv_end), wlv_step, 'kbest_selected.png')

print 'Performing Feature Importance selection.'
etrees_selection = ExtraTreesClassifier()
etrees_selection.fit(spectral_data, galah_prob_flag)
wvl_importance_etrees = etrees_selection.feature_importances_  # higher importance -> better feture
print wvl_importance_etrees
wvl_selected_etrees = wvl_range[np.argsort(wvl_importance_etrees)[-n_features_out:]]
plot_selected_ranges(wvl_selected_etrees, (wlv_begin, wlv_end), wlv_step, 'etrees_selected.png')


