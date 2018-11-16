import imp
from astropy.table import Table, join, hstack, vstack
import numpy as np
from glob import glob
from os import chdir, system, path
import matplotlib.pyplot as plt

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import *
imp.load_source('helper2', '../tSNE_test/cannon3_functions.py')
from helper2 import *


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


data_dir = '/data4/cotar/'

cannon_data = Table.read(data_dir + 'GALAH_iDR3_ts_DR2.fits')
openc_data = Table.read(data_dir + 'GALAH_iDR3_OpenClusters.fits')
globc_data = Table.read(data_dir + 'GALAH_iDR3_GlobularClusters.fits')
cluster_data = vstack([openc_data, globc_data, cannon_data])
# print cannon_data.colnames

sub_dir = 'Cannon3.0_SME_20180327_multiple_30_dropout0.3_allspectrum_prelu_C-11-5-3_F-32-64-64_Adam_completetrain/'

fits_orig = glob(data_dir + sub_dir + 'galah_*_run*.fits')

chdir(data_dir + sub_dir)
system('mkdir combined')
chdir('combined')

final_ann_fits = 'GALAH_iDR3_ts_DR2_abund_ANN.fits'

if not path.isfile(final_ann_fits):
    data_raw_list = list([])
    for fits in fits_orig:
        fits_name = fits.split('/')[-1][:-5]
        print fits_name
        abund_ann = Table.read(fits)
        ann_cols = [c for c in abund_ann.colnames if '_ann' in c]
        remove_cols = [c for c in abund_ann.colnames if c not in ann_cols and 'sobject_id' not in c]
        # print abund_ann.colnames
        data_raw = abund_ann[ann_cols].to_pandas().values
        data_raw_list.append(data_raw)
        # print data_raw.shape

        # H-R diagnostics plot
        plt.scatter(abund_ann['teff_ann'], abund_ann['logg_ann'], s=0.4, alpha=0.1, lw=0, c='black', label='ANN')
        plt.scatter(cannon_data['teff'], cannon_data['logg'], s=0.6, alpha=1., lw=0, c='red', label='SME')
        plt.xlabel('Teff')
        plt.ylabel('logg')
        plt.xlim(7600, 3300)
        plt.ylim(5.5, 0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fits_name+'_kiel.png', dpi=250)
        plt.close()

        # H-R diagnostics plot
        idx_train_val = np.isfinite(abund_ann['teff'])
        plt.scatter(abund_ann['teff_ann'][idx_train_val], abund_ann['logg_ann'][idx_train_val], s=0.6, alpha=1., lw=0, c='black', label='ANN - train')
        plt.scatter(cannon_data['teff'], cannon_data['logg'], s=0.6, alpha=1., lw=0, c='red', label='SME')
        plt.xlabel('Teff')
        plt.ylabel('logg')
        plt.xlim(7600, 3300)
        plt.ylim(5.5, 0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fits_name + '_kiel_trainset.png', dpi=250)
        plt.close()

        abund_ann.remove_columns(remove_cols)

    data_raw_all = np.stack(data_raw_list)
    data_raw_median = np.median(data_raw_all, axis=0)
    data_raw_std = np.std(data_raw_all, axis=0)

    for i_c in range(len(ann_cols)):
        print ann_cols[i_c]
        # write results to final array and compute std for every abundance measurement
        abund_ann[ann_cols[i_c]] = data_raw_median[:, i_c]
        abund_ann['e_'+ann_cols[i_c]] = data_raw_std[:, i_c]
        std_abund = np.nanstd(abund_ann[ann_cols[i_c]])
        # filter outliers by setting them to nan
        # idx_std_out = np.abs(abund_ann[abund_cols[i_c]] - np.nanmean(abund_ann[abund_cols[i_c]])) >= 5.*std_abund
        # abund_ann[abund_cols[i_c]][idx_std_out] = np.nan
        # abund_ann['e_'+abund_cols[i_c]][idx_std_out] = np.nan
        #
        # print ' STD before:{:.2f} after:{:.2f}'.format(std_abund, np.nanstd(abund_ann[abund_cols[i_c]]))
        # print '  - masked:', np.sum(idx_std_out)

    abund_ann.write(final_ann_fits, overwrite=True)
else:
    abund_ann = Table.read(final_ann_fits)

# abund_ann = join(abund_ann, cannon_data, keys='sobject_id', join_type='left')
abund_ann = join(abund_ann, cluster_data, keys='sobject_id', join_type='left')
suffix = '_clusters'
print abund_ann.colnames
print abund_ann

for plot_abund in [c for c in abund_ann.colnames if 'abund' in c and len(c.split('_')) == 3]:
    print ' plotting attribute - ' + plot_abund
    elem_plot = plot_abund.split('_')[0]
    # determine number of lines used for this element
    ann_vals = abund_ann[elem_plot + '_abund_ann']
    sme_vals = abund_ann[elem_plot + '_fe']
    idx_oc = np.in1d(abund_ann['sobject_id'], openc_data['sobject_id'])
    idx_gc = np.in1d(abund_ann['sobject_id'], globc_data['sobject_id'])
    graphs_title = elem_plot.capitalize() + ' - trained on SME values - BIAS: {:.2f}   RMSE: {:.2f}'.format(
        bias(ann_vals, sme_vals), rmse(ann_vals, sme_vals))
    plot_range = (np.nanpercentile(cannon_data[elem_plot + '_fe'], 0.5), np.nanpercentile(cannon_data[elem_plot + '_fe'], 99.5))
    # first scatter graph - train points
    plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
    plt.scatter(sme_vals, ann_vals, lw=0, s=0.4, c='black', label='')
    plt.scatter(sme_vals[idx_oc], ann_vals[idx_oc], lw=0, s=1, c='green', label='OC')
    plt.scatter(sme_vals[idx_gc], ann_vals[idx_gc], lw=0, s=1, c='red', label='GC')
    plt.title(graphs_title)
    plt.xlabel('SME reference value')
    plt.ylabel('ANN computed value')
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.legend()
    plt.savefig('final_' + elem_plot + '_ANN'+suffix+'.png', dpi=400)
    plt.close()

    plt.scatter(abund_ann['fe_h_ann'], ann_vals, lw=0, s=0.5, alpha=0.2, c='black', label='')
    plt.xlabel('Fe/H value')
    plt.ylabel('ANN computed value')
    plt.xlim([-2, 1])
    plt.ylim([-2, 2])
    plt.legend()
    plt.savefig('final_' + elem_plot + '_ANN'+suffix+'_feh.png', dpi=400)
    plt.close()

for plot_abund in ['teff', 'logg', 'fe_h', 'vbroad']:
    print ' plotting attribute - ' + plot_abund
    # determine number of lines used for this element
    ann_vals = abund_ann[plot_abund + '_ann']
    sme_vals = abund_ann[plot_abund + '']
    idx_oc = np.in1d(abund_ann['sobject_id'], openc_data['sobject_id'])
    idx_gc = np.in1d(abund_ann['sobject_id'], globc_data['sobject_id'])
    graphs_title = plot_abund + ' - trained on SME values - BIAS: {:.2f}   RMSE: {:.2f}'.format(
        bias(ann_vals, sme_vals), rmse(ann_vals, sme_vals))
    plot_range = (np.nanpercentile(cannon_data[plot_abund], 0.5), np.nanpercentile(cannon_data[plot_abund], 99.5))
    # first scatter graph - train points
    plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
    plt.scatter(sme_vals, ann_vals, lw=0, s=0.4, c='black')
    plt.scatter(sme_vals[idx_oc], ann_vals[idx_oc], lw=0, s=1, c='green', label='OC')
    plt.scatter(sme_vals[idx_gc], ann_vals[idx_gc], lw=0, s=1, c='red', label='GC')
    plt.title(graphs_title)
    plt.xlabel('SME reference value')
    plt.ylabel('ANN computed value')
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.legend()
    plt.savefig('final_' + plot_abund + '_ANN'+suffix+'.png', dpi=400)
    plt.close()

# H-R diagnostics plot
plt.scatter(abund_ann['teff_ann'], abund_ann['logg_ann'], s=0.4, alpha=0.1, lw=0, c='black', label='ANN')
plt.scatter(abund_ann['teff'], abund_ann['logg'], s=0.6, alpha=1., lw=0, c='red', label='SME')
plt.xlabel('Teff')
plt.ylabel('logg')
plt.xlim(7600, 3300)
plt.ylim(5.5, 0)
plt.legend()
plt.tight_layout()
plt.savefig('final_kiel_ANN'+suffix+'.png', dpi=250)
plt.close()
