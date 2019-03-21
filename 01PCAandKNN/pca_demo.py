import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
print pca.explained_variance_ratio_
print pca.singular_values_

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,svd_solver='full', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,svd_solver='arpack', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
