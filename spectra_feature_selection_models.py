import imp, os, socket, sys
from skfeature.function.statistical_based import CFS, f_score, t_score

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
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
else:
    galah_data_input = '/data4/cotar/'
from helper_functions import move_to_dir

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


def plot_selected_ranges(wvl_selected, wvl_range, wvl_step, png_name,
                         spectra_f=None, spectra_w=None):
    if spectra_f is not None and spectra_w is not None:
        plt.plot(spectra_w, spectra_f, color='black', lw=0.1)
    for wvl in wvl_selected:
        plt.axvspan(wvl-wvl_step/2., wvl+wvl_step/2., facecolor='blue', alpha=0.5, lw=0)
    plt.xlim(wvl_range)
    plt.ylim((0.4, 1.2))
    plt.tight_layout()
    plt.show()
    plt.savefig(png_name, dpi=500)
    plt.close()


def get_best_wvl(wvl_scores, wvl, n_out):
    best_wvl = wvl[np.argsort(wvl_scores)[-n_out:]]  # last because they are ordered in increasing order
    return best_wvl


def final_best_wvl(in_best, wvl_scores, n_out, union=False):
    best_wvl = np.argsort(wvl_scores)[-n_out:]
    best_array = np.ndarray(len(in_best))
    best_array.fill(False)
    best_array[best_wvl] = True
    if union:
        return np.logical_or(in_best, best_array)
    else:
        return np.logical_and(in_best, best_array)


def final_best_wvl_range(in_best_range, wvl, wvl_scores, n_out, range=0.3, union=False):  # range in Ang
    best_range = np.ndarray(len(in_best_range))
    best_range.fill(False)
    best_wvl = get_best_wvl(wvl_scores, wvl, n_out)
    for b_w in best_wvl:
        idx_mark = np.logical_and(wvl > b_w - range,
                                  wvl < b_w + range)
        best_range[idx_mark] = True
    if union:
        return np.logical_or(in_best_range, best_range)
    else:
        return np.logical_and(in_best_range, best_range)


def generate_ranges_outputs(wvl_selected, wvl, txt_path):
    selection = False
    txt_out = open(txt_path, 'w')
    for i_wvl in range(len(wvl)):
        if selection:
            if not wvl_selected[i_wvl]:
                selection = False
                txt_out.write(' '+str(wvl[i_wvl])+'\n')
        else:
            if wvl_selected[i_wvl]:
                selection = True
                txt_out.write(str(wvl[i_wvl-1]))
    # end of the wvl range check
    if selection:
        txt_out.write(' ' + str(wvl[-1]) + '\n')
    txt_out.close()


# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------

# algorithm settings
output_results = True
output_plots = True
snr_cut = False  # do not use in this configuration

# reading settings
spectra_get_cols = [4000, 4000, 4000, 2016]

input_arguments = sys.argv
if len(input_arguments) > 1:
    read_galah_bands = np.int8(input_arguments[1].split(','))
    print 'Manual bands: '+str(read_galah_bands)
else:
    read_galah_bands = [0, 1, 2, 3]
if len(input_arguments) > 2:
    select_one_class = input_arguments[2]
    print 'Manual class: '+str(select_one_class)
else:
    select_one_class = None

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
print 'Dr51 and dr52 crossection size: '+str(len(galah_common_sobject))

# reassign problematic flags from strings to integers/numbers
use_classes = ['binary', 'hot stars', 'cool metal-poor giants',  'mol. absorption bands', 'HaHb emission']
remove_classes = ['0', 'problematic']
if select_one_class is not None:
    for use_class in use_classes:
        if select_one_class != use_class:
            remove_classes.append(use_class)
    use_classes = [select_one_class]

# first remove problematic and unknown classes from the list
print 'Removing unknown/problematic objects'
for i_r in range(len(remove_classes)):
    sobj_rem = galah_prob[galah_prob['published_reduced_class_proj1'] == remove_classes[i_r]]['sobject_id']
    if len(sobj_rem) > 0:
        print ' Class "'+remove_classes[i_r]+'" has objects: '+str(len(sobj_rem))
        idx_use = np.in1d(galah_common_sobject, sobj_rem, invert=True)
        galah_common_sobject = galah_common_sobject[idx_use]

# assign classes to the spectra of objects
print 'Determining numeric classes'
galah_prob_flag = np.zeros(len(galah_common_sobject))
for i_u in range(len(use_classes)):
    sobj_class = galah_prob[galah_prob['published_reduced_class_proj1'] == use_classes[i_u]]['sobject_id']
    if len(sobj_class) > 0:
        print ' Class "' + use_classes[i_u] + '" has objects: ' + str(len(sobj_class))
        idx_class = np.in1d(galah_common_sobject, sobj_class, invert=False)
        galah_prob_flag[idx_class] = i_u+1

# create class statistics
flag_uniq, flag_counts = np.unique(galah_prob_flag, return_counts=True)
n_ok_obj = np.sum(flag_counts[flag_uniq == 0])
print 'Non-problematic objects: '+str(n_ok_obj)
n_bad_obj = np.sum(flag_counts[flag_uniq != 0])
print 'Problematic objects: '+str(n_bad_obj)
# create more balanced dataset for classification problem
if n_ok_obj > n_bad_obj*2:
    print 'Balancing number of objects in classes'
    sobj_ok = galah_common_sobject[galah_prob_flag == 0]
    # select random subset of ok objects
    sobj_ok_sel = sobj_ok[np.int64(np.random.rand(n_bad_obj*2)*n_ok_obj)]
    sobj_sel = np.hstack((sobj_ok_sel, galah_common_sobject[galah_prob_flag > 0]))
    idx_sel = np.in1d(galah_common_sobject, sobj_sel)
    galah_common_sobject = galah_common_sobject[idx_sel]
    galah_prob_flag = galah_prob_flag[idx_sel]

# determine spectral data rows to be read
skip_rows = np.where(np.in1d(galah_param['sobject_id'], galah_common_sobject, invert=True))[0]
print 'Rows to be read: '+str(len(galah_param)-len(skip_rows))

# --------------------------------------------------------
# ---------------- READ SPECTRA --------------------------
# --------------------------------------------------------

if select_one_class is None:
    move_to_dir('Features_all-classes')
else:
    move_to_dir('Features_'+select_one_class)

# spectral_data = list([])
for i_band in read_galah_bands:
    suffix = 'b'+str(i_band+1)+'_'
    suffix += ''

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
    get_cols = range(col_start, col_end)
    wvl_read = wvl_range[get_cols]

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectral data from {:07.2f} to {:07.2f}'.format(wvl_range[col_start], wvl_range[col_end])
    spectral_data = pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan', dtype=np.float16,  # float16 is more than enough
                                     usecols=get_cols, skiprows=skip_rows).values

    # spectral_data = np.hstack(spectral_data)
    print spectral_data.shape
    n_wvl_total = spectral_data.shape[1]

    med_spectra = np.median(spectral_data, axis=0)

    # somehow handle cols with nan values, delete cols or fill in data
    idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
    n_bad_spectra = len(idx_bad_spectra[0])
    if n_bad_spectra > 0:
        print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
        spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

    # handle extreme values in data
    val_large = 2.5
    idx_large = np.where(spectral_data > val_large)
    n_bad_spectra = len(idx_large[0])
    if n_bad_spectra > 0:
        print 'Correcting '+str(n_bad_spectra)+' large flux values.'
        spectral_data[idx_large] = val_large  # remove nan values with theoretical continuum flux value

    val_negative = 0.01
    idx_negative = np.where(spectral_data < val_negative)
    n_bad_spectra = len(idx_negative[0])
    if n_bad_spectra > 0:
        print 'Correcting '+str(n_bad_spectra)+' negative flux values.'
        spectral_data[idx_negative] = val_negative  # remove nan values with theoretical continuum flux value


    # --------------------------------------------------------
    # ---------------- FEATURE SELECTION PROCESS -------------
    # --------------------------------------------------------

    # crossection
    wvl_best = np.ndarray(len(wvl_read))
    wvl_best.fill(True)
    wvl_best_range = np.array(wvl_best)
    # union
    wvl_best_union = np.ndarray(len(wvl_read))
    wvl_best_union.fill(False)
    wvl_best_range_union = np.array(wvl_best_union)

    # Create the RFE object and rank flux at each pixel
    n_features_out = 200

    print 'Performing KBest feature selection 1.'
    kbest_selection = SelectKBest(chi2, k='all')
    kbest_selection.fit(spectral_data, galah_prob_flag)
    wvl_scores_kbest = kbest_selection.scores_  # higher score -> better feature
    print wvl_scores_kbest
    plot_selected_ranges(get_best_wvl(wvl_scores_kbest, wvl_read, n_features_out),
                         (wlv_begin, wlv_end), wlv_step, suffix+'kbest_selected_chi2.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    wvl_best = final_best_wvl(wvl_best, wvl_scores_kbest, n_features_out)
    wvl_best_range = final_best_wvl_range(wvl_best_range, wvl_read, wvl_scores_kbest, n_features_out)
    wvl_best_union = final_best_wvl(wvl_best_union, wvl_scores_kbest, n_features_out, union=True)
    wvl_best_range_union = final_best_wvl_range(wvl_best_range_union, wvl_read, wvl_scores_kbest, n_features_out, union=True)


    print 'Performing Feature Importance selection.'
    etrees_selection = ExtraTreesClassifier()
    etrees_selection.fit(spectral_data, galah_prob_flag)
    wvl_importance_etrees = etrees_selection.feature_importances_  # higher importance -> better feature
    print wvl_importance_etrees
    plot_selected_ranges(get_best_wvl(wvl_importance_etrees, wvl_read, n_features_out),
                         (wlv_begin, wlv_end), wlv_step, suffix+'etrees_selected.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    wvl_best = final_best_wvl(wvl_best, wvl_importance_etrees, n_features_out)
    wvl_best_range = final_best_wvl_range(wvl_best_range, wvl_read, wvl_importance_etrees, n_features_out)
    wvl_best_union = final_best_wvl(wvl_best_union, wvl_importance_etrees, n_features_out, union=True)
    wvl_best_range_union = final_best_wvl_range(wvl_best_range_union, wvl_read, wvl_importance_etrees, n_features_out, union=True)

    from skfeature.function.similarity_based import fisher_score, reliefF
    print 'Fisher score'
    wvl_scores_fisher = fisher_score.fisher_score(spectral_data, galah_prob_flag)
    print wvl_scores_fisher
    plot_selected_ranges(get_best_wvl(wvl_scores_fisher, wvl_read, n_features_out),
                         (wlv_begin, wlv_end), wlv_step, suffix+'fisher_selected.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    wvl_best = final_best_wvl(wvl_best, wvl_scores_fisher, n_features_out)
    wvl_best_range = final_best_wvl_range(wvl_best_range, wvl_read, wvl_scores_fisher, n_features_out)
    wvl_best_union = final_best_wvl(wvl_best_union, wvl_scores_fisher, n_features_out, union=True)
    wvl_best_range_union = final_best_wvl_range(wvl_best_range_union, wvl_read, wvl_scores_fisher, n_features_out,
                                                union=True)

    print 'RelierF score'
    wvl_scores_reliefF = reliefF.reliefF(spectral_data, galah_prob_flag, k=10)
    print wvl_scores_reliefF
    plot_selected_ranges(get_best_wvl(wvl_scores_reliefF, wvl_read, n_features_out),
                         (wlv_begin, wlv_end), wlv_step, suffix+'reliefF_selected.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    wvl_best = final_best_wvl(wvl_best, wvl_scores_reliefF, n_features_out)
    wvl_best_range = final_best_wvl_range(wvl_best_range, wvl_read, wvl_scores_reliefF, n_features_out)
    wvl_best_union = final_best_wvl(wvl_best_union, wvl_scores_reliefF, n_features_out, union=True)
    wvl_best_range_union = final_best_wvl_range(wvl_best_range_union, wvl_read, wvl_scores_reliefF, n_features_out,
                                                union=True)

    # if select_one_class is not None:
    #     # TAKES A LONG TIME TO COMPLETE
    #     print 'CFS statistical score'
    #     wvl_scores_CFS = CFS.cfs(spectral_data, galah_prob_flag)
    #     print wvl_scores_CFS
    #     plot_selected_ranges(wvl_read[wvl_scores_CFS],
    #                          (wlv_begin, wlv_end), wlv_step, suffix+'cfs_selected.png',
    #                          spectra_f=med_spectra, spectra_w=wvl_read)

    print 'F-score statistical score'
    wvl_scores_fscore = f_score.f_score(spectral_data, galah_prob_flag)
    print wvl_scores_fscore
    plot_selected_ranges(get_best_wvl(wvl_scores_fscore, wvl_read, n_features_out),
                         (wlv_begin, wlv_end), wlv_step, suffix+'fscore_selected.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    wvl_best = final_best_wvl(wvl_best, wvl_scores_fscore, n_features_out)
    wvl_best_range = final_best_wvl_range(wvl_best_range, wvl_read, wvl_scores_fscore, n_features_out)
    wvl_best_union = final_best_wvl(wvl_best_union, wvl_scores_fscore, n_features_out, union=True)
    wvl_best_range_union = final_best_wvl_range(wvl_best_range_union, wvl_read, wvl_scores_fscore, n_features_out,
                                                union=True)

    if len(np.unique(galah_prob_flag)) == 2:
        # NOTE: y should be guaranteed to a binary class vector
        print 'T-score statistical score'
        wvl_scores_tscore = t_score.t_score(spectral_data, galah_prob_flag)
        print wvl_scores_tscore
        plot_selected_ranges(get_best_wvl(wvl_scores_tscore, wvl_read, n_features_out),
                             (wlv_begin, wlv_end), wlv_step, suffix+'tscore_selected.png',
                             spectra_f=med_spectra, spectra_w=wvl_read)
        wvl_best = final_best_wvl(wvl_best, wvl_scores_tscore, n_features_out)
        wvl_best_range = final_best_wvl_range(wvl_best_range, wvl_read, wvl_scores_tscore, n_features_out)
        wvl_best_union = final_best_wvl(wvl_best, wvl_best_union, n_features_out, union=True)
        wvl_best_range_union = final_best_wvl_range(wvl_best_range_union, wvl_read, wvl_scores_tscore, n_features_out,
                                                    union=True)

        # if select_one_class is not None:
    #     # MAY TAKE TOO LONG
    #     print 'Performing RFE feature selection'
    #     svc_classifier = SVC(kernel="linear", C=1)
    #     rfe_selection = RFE(estimator=svc_classifier, n_features_to_select=n_features_out, step=0.1, verbose=1)
    #     rfe_selection.fit(spectral_data, galah_prob_flag)
    #     wvl_ranking_rfe = rfe_selection.ranking_  # selected features are marked with 1, others that were eliminated are increasing in number
    #     print wvl_ranking_rfe
    #     print wvl_ranking_rfe.shape  # TODO: investigate strange shape of this array
    #     print wvl_range.shape
    #     wvl_selected_rfe = wvl_range[wvl_ranking_rfe == 1]
    #     plot_selected_ranges(wvl_selected_rfe, (wlv_begin, wlv_end), wlv_step, suffix+'rfe_selected.png',
    #                          spectra_f=med_spectra, spectra_w=wvl_read)

    plot_selected_ranges(wvl_read[wvl_best],
                         (wlv_begin, wlv_end), wlv_step, suffix + 'final.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    plot_selected_ranges(wvl_read[wvl_best_range],
                         (wlv_begin, wlv_end), wlv_step, suffix + 'final_ranges.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    plot_selected_ranges(wvl_read[wvl_best_union],
                         (wlv_begin, wlv_end), wlv_step, suffix + 'final_union.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)
    plot_selected_ranges(wvl_read[wvl_best_range_union],
                         (wlv_begin, wlv_end), wlv_step, suffix + 'final_union_ranges.png',
                         spectra_f=med_spectra, spectra_w=wvl_read)

    # output start and end points of the selected ranges
    # generate csv file with outputs

    generate_ranges_outputs(wvl_best, wvl_read, suffix + 'final.txt')
    generate_ranges_outputs(wvl_best, wvl_best_range, suffix + 'final_ranges.txt')
    generate_ranges_outputs(wvl_best, wvl_best_union, suffix + 'final_union.txt')
    generate_ranges_outputs(wvl_best, wvl_best_range_union, suffix + 'final_union_ranges.txt')
