import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from torch import float32

from utils import *
import torch
import math
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from collections import Counter
import random
import faiss


class K_clustering(object):  # GMM

    def __init__(self, dataset, know_num, k, pca=512, random_state=42):
        self.know_num = know_num
        self.k = k  # K
        self.dataset = dataset
        self.pca = pca  # number of dimensions projected to in PCA
        self.random_state = random_state

    def clustering(self, method="Faiss.Kmeans"):
        document_representations = np.array(self.dataset["represent"]).astype('float32')
        if self.pca > 0:
            mat = faiss.PCAMatrix(document_representations.shape[1], self.pca)
            mat.train(document_representations)
            assert mat.is_trained
            document_representations = mat.apply_py(document_representations)

        if method == "GMM":
            gmm = GaussianMixture(n_components=self.k, covariance_type='tied', n_init=10, random_state=self.random_state)
            gmm.fit(document_representations)
            documents_to_class = gmm.predict(document_representations)
            distribute = -gmm.predict_proba(document_representations).max(axis=1) + 1
            return documents_to_class.tolist()#gmm.means_
        elif method == "Kmeans":
            kmeans = KMeans(n_clusters=self.k, n_init=1 + 100 // self.know_num, max_iter=20, random_state=self.random_state)
            kmeans.fit(document_representations)
            documents_to_class = kmeans.predict(document_representations)
            return documents_to_class.tolist()
        elif method == "Faiss.Kmeans":  # fast
            ncentroids = self.k
            niter = 20
            nredo = 5
            d = document_representations.shape[1]
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, nredo=nredo, seed=self.random_state)
            kmeans.train(document_representations)
            D, I = kmeans.index.search(document_representations, 1)
            return I.reshape(-1).tolist() #kmeans.centroids