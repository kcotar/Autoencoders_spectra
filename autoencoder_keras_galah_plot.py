import matplotlib
matplotlib.use('Agg')

from os import chdir
import numpy as np
from astropy.table import Table
from sklearn.externals import joblib
import matplotlib.pyplot as plt

print 'Reading data sets'
galah_data_input = '/shared/ebla/cotar/'
date_string = '20180327'
galah_param_file = 'sobject_iraf_53_reduced_'+date_string+'.fits'
tsne_file = 'tsne_class_1_0.csv'

galah_data = Table.read(galah_data_input + galah_param_file)
tsne_data = Table.read(galah_data_input + tsne_file)
tsne_data = tsne_data.filled()

proj_dir = '/shared/ebla/cotar/Autoencoder_dense_test_complex_ccd3_prelu_2D_4layers_relu/'
proj_coords = joblib.load(proj_dir + 'encoded_spectra.pkl')

chdir(proj_dir)

x_range = np.percentile(proj_coords[:,0], [1.5,98.5])
y_range = np.percentile(proj_coords[:,1], [1.5,98.5])

for c in np.unique(tsne_data['published_reduced_class_proj1']):
	if c == 'N/A':
		continue
	idx_mark = np.in1d(galah_data['sobject_id'], tsne_data[tsne_data['published_reduced_class_proj1'] == c]['sobject_id'])
	plt.scatter(proj_coords[:,0], proj_coords[:,1], lw=0, s=0.5, c='black', alpha=0.3)
	plt.scatter(proj_coords[idx_mark,0], proj_coords[idx_mark,1], lw=0, s=1, c='red', alpha=1)
	plt.title(c)
	plt.xlim(x_range)
	plt.ylim(y_range)
	plt.tight_layout()
	plt.savefig('proj_'+c+'.png', dpi=300)
	plt.close()
