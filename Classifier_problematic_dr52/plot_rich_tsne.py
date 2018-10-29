import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from itertools import combinations
from sklearn.externals import joblib
from Inspect_GALAH_class import *
from os import chdir

# data = Table.read('Cannon3.0.1_Sp_SMEmasks_trainingset.fits')
tsne = Table.read('tsne_result.csv')
tsne_class = Table.read('dr52_class_joined.csv')
tsne_class = tsne_class.filled(0)
ann_class = joblib.load('multiclass_prob_array_withok_1.pkl')
date_string = '20171111'
galah_param_file = 'sobject_iraf_52_reduced_'+date_string+'_pos.fits'
galah = Table.read('/home/klemen/data4_mount/'+galah_param_file)

print ann_class[galah['sobject_id']==171003002601218]

raise SystemExit

ann_class = ann_class > 0.75

ann_class_n = np.sum(ann_class, axis=1)
print np.min(ann_class_n), np.max(ann_class_n)
print np.unique(ann_class_n, return_counts=True)

# plot some random spectra for selected 
help_lines=[
    (r'$\mathrm{H_\alpha}$',6562.7970),
    (r'$\mathrm{H_\beta}$' ,4861.3230),
    (r'$\mathrm{Li}$'      ,6707.7635)
    ]
spectra_dir = '/media/storage/HERMES_REDUCED/dr5.2/'
n_plots_rand = 250
for a_c in [2, 3]:
	save_fig_dir = 'spectra_class_'+str(a_c)
	c_sid = galah[ann_class[:,a_c]]['sobject_id']
	rand_sid = c_sid[np.unique(np.int64(np.random.rand(n_plots_rand) * len(c_sid)))]
	print a_c, len(c_sid)
	for sid in rand_sid:
		print sid
		fits = fits_class(sobject_id=sid, directory=spectra_dir)
		fits.plot_norm_spectrum_on4axes(help_lines=help_lines, savefig=save_fig_dir)


raise SystemExit

for a_c in range(ann_class.shape[1]):
	c_sid = galah[ann_class[:,a_c]]['sobject_id']
	idx_tsne_sel = np.in1d(tsne['sobject_id'], c_sid)
	plt.scatter(tsne['tsne_axis1'],tsne['tsne_axis2'], s=1, lw=0, c='black')
	plt.scatter(tsne['tsne_axis1'][idx_tsne_sel],tsne['tsne_axis2'][idx_tsne_sel], s=1, lw=0, c='red')
	plt.savefig('ann_'+str(a_c)+'_multi_wok3.png', dpi=300)
	plt.close()

raise SystemExit

for prob_class in np.unique(tsne_class['dr52_class_reduced']):
	print 'Plot class', prob_class
	binary_sid = tsne_class[tsne_class['dr52_class_reduced']==prob_class]['sobject_id']
	idx_tsne_sel = np.in1d(tsne['sobject_id'], binary_sid)
	plt.scatter(tsne['tsne_axis1'],tsne['tsne_axis2'], s=1, lw=0, c='black')
	plt.scatter(tsne['tsne_axis1'][idx_tsne_sel],tsne['tsne_axis2'][idx_tsne_sel], s=1, lw=0, c='red')
	plt.savefig('tsne_'+prob_class+'.png', dpi=300)
	plt.close()

raise SystemExit

ann_class = Table.read('class_problematic_ann.fits')
for a_c in np.unique(ann_class['class']):
	print 'Plot class', a_c
	c_sid = ann_class[ann_class['class']==a_c]['sobject_id']
	idx_tsne_sel = np.in1d(tsne['sobject_id'], c_sid)
	plt.scatter(tsne['tsne_axis1'],tsne['tsne_axis2'], s=1, lw=0, c='black')
	plt.scatter(tsne['tsne_axis1'][idx_tsne_sel],tsne['tsne_axis2'][idx_tsne_sel], s=1, lw=0, c='red')
	plt.savefig('tsne_'+str(a_c)+'.png', dpi=300)
	plt.close()

raise SystemExit

for prob_class in np.unique(tsne_class['dr52_class']):
	print 'Plot class', prob_class
	dr52class_sid = tsne_class[tsne_class['dr52_class']==prob_class]['sobject_id']
	# random selection
	n_sid = len(dr52class_sid)
	random_sel = dr52class_sid[np.int64(np.random.rand(25)*n_sid)]
	print ','.join([str(id) for id in random_sel])



abund_col = [col for col in data.colnames if 'abund_sme' in col and 'e_' not in col and len(col.split('_')[0])<3]
print abund_col

rich_sid = data[np.logical_and(data['Y_abund_sme']>0.5,data['Ba_abund_sme']>0.5)]['sobject_id']

for abund in abund_col:
	print 'Plot rich', abund
	rich_sid = data[data[abund]>0.5]['sobject_id']
	elem = abund.split('_')[0]
	idx_tsne_sel = np.in1d(tsne['sobject_id'], rich_sid)
	plt.scatter(tsne['tsne_axis1'],tsne['tsne_axis2'], s=1, lw=0, c='black')
	plt.scatter(tsne['tsne_axis1'][idx_tsne_sel],tsne['tsne_axis2'][idx_tsne_sel], s=1, lw=0, c='red')
	plt.savefig('rich/tsne_'+elem+'_rich.png', dpi=300)
	plt.close()
