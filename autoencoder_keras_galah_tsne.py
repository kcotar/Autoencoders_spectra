import matplotlib.pyplot as plt
import numpy as np
#import umap

from os import system, chdir
from astropy.table import Table
from sklearn.externals import joblib
from MulticoreTSNE import MulticoreTSNE as TSNE_multi

# input data
galah_data_input = '/shared/ebla/cotar/'
data_output = '/shared/data-camelot/cotar/'

date_string = '20190801'
galah_data = Table.read(galah_data_input + 'sobject_iraf_53_reduced_'+date_string+'.fits')
cannon_data = Table.read(galah_data_input + 'sobject_iraf_iDR2_180325_cannon.fits')
ts_data = Table.read(galah_data_input + 'tsne_class_1_0.csv')
ts_data = ts_data.filled()

# spectrum_data = list([])
# for i_b in [1, 2, 3, 4]:  # [1, 2, 3, 4]:
#     print 'Reading decoded spectrum', i_b
#     spectrum_data.append(joblib.load(galah_data_input + 'Autoencoder_dense_test_complex_ccd'+str(i_b)+'_prelu_30D_4layer/encoded_spectra.pkl'))
# spectrum_data = np.hstack(spectrum_data)

spectrum_data = joblib.load(galah_data_input + 'Autoencoder_dense_test_complex_ccd1234_relu_100D_4layer/encoded_spectra_ccd5_nf100.pkl')

# use only spectra will completly valid rows
idx_valid = np.isfinite(spectrum_data).all(axis=1)
spectrum_data = spectrum_data[idx_valid, :]
galah_data = galah_data[idx_valid]

out_dir = galah_data_input + 'Autoencoder_dense_test_complex_ccd1234_relu_100D_4layer_tsne'
system('mkdir '+out_dir)
chdir(out_dir)

# --------------------------------------------
# --------------------------------------------
# run tSNE on reduced data
print 'Running multi-core tSNE projection'
perp = 75
theta = 0.4
tsne_class = TSNE_multi(n_components=2, perplexity=perp, n_iter=1200, n_iter_without_progress=350,
                        init='random', verbose=1, method='barnes_hut', angle=theta, n_jobs=65)
tsne_res = tsne_class.fit_transform(spectrum_data)

plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, s=1, alpha=0.2, c='black')
plt.tight_layout()
plt.savefig('tsne.png', dpi=300)
plt.close()

for u_c in np.unique(ts_data['published_reduced_class_proj1']):
    if u_c == 'N/A':
        continue
    idx_mark = np.in1d(galah_data['sobject_id'], ts_data[ts_data['published_reduced_class_proj1'] == u_c]['sobject_id'])
    print u_c, np.sum(idx_mark)
    plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, s=1, alpha=0.2, c='black')
    plt.scatter(tsne_res[idx_mark, 0], tsne_res[idx_mark, 1], lw=0, s=1, alpha=1., c='red')
    plt.tight_layout()
    plt.savefig('tsne_'+u_c+'.png', dpi=300)
    plt.close()

raise SystemExit
# --------------------------------------------
# --------------------------------------------
# run UMAP on reduced data
neighb = 75
dist = 0.7
spread = 1.0
metric = 'euclidean'

print ' Running UMAP'
umap_embed = umap.UMAP(n_neighbors=neighb,
                       min_dist=dist,
                       spread=spread,
                       metric=metric,
                       init='spectral',  # spectral or random
                       local_connectivity=1,
                       set_op_mix_ratio=1.,  # 0. - 1.
                       n_components=2,
                       transform_seed=42,
                       n_epochs=1000,
                       verbose=True).fit_transform(spectrum_data)

plt.scatter(umap_embed[:, 0], umap_embed[:, 1], lw=0, s=1, alpha=0.2, c='black')
plt.tight_layout()
plt.savefig('umap.png', dpi=300)
plt.close()

for u_c in np.unique(ts_data['published_reduced_class_proj1']):
    idx_mark = np.in1d(galah_data['sobject_id'], ts_data[ts_data['published_reduced_class_proj1'] == u_c]['sobject_id'])
    plt.scatter(umap_embed[:, 0], umap_embed[:, 1], lw=0, s=1, alpha=0.2, c='black')
    plt.scatter(umap_embed[idx_mark, 0], umap_embed[idx_mark, 1], lw=0, s=1., alpha=1, c='red')
    plt.tight_layout()
    plt.savefig('umap_'+u_c+'.png', dpi=300)
    plt.close()

