import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from random import shuffle
from skimage.io import imread
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import rand_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.mixture import GaussianMixture
from pandas.plotting import scatter_matrix
import pandas as pd
import tp2_aux as tp2_aux
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def getData():
	images = tp2_aux.images_as_matrix()
	images = np.array(images)

	return images

def getLabels(images):
	labels = np.loadtxt('labels.txt', delimiter = ',')
	images = images[labels[:,1]>0]
	labels = labels[labels[:,1]>0][:,1]

	return labels, images

# feature extraction
def PCA_FE(images):
	pca = PCA(n_components=6)
	pca.fit(images)
	data = pca.transform(images)
	return data

def TSNE_FE(images):
	tsne = TSNE(n_components=6, method='exact', n_jobs=-1)
	data = tsne.fit_transform(images)
	return data

def Isomap_FE(images):
	isomap = Isomap(n_components=6, n_jobs=-1)
	data = isomap.fit_transform(images)
	return data

def plotElbow(dist_4):
    fig, ax = plt.subplots()
    ax.set_ylabel('4-distance')
    ax.set_xlabel('points')
    ax.set_title('Cotovelo')
    ax.plot(range(0, 563), dist_4, 'o')
    # plt.show()
    fig.savefig("valley.png")

def create_elbow_plot(images):
	zeros = np.zeros(563)
	neigh = KNeighborsClassifier()
	neigh.fit(images, zeros)
	neigh_dist, neigh_ind = neigh.kneighbors(n_neighbors = 4, return_distance = True)
	dist_4 =  neigh_dist[:,3]
	dist_4.sort()
	dist_4 = dist_4[::-1]
	plotElbow(dist_4)

def selectBestFeatures(images, labels, k):
	kb = SelectKBest(f_classif, k=k)
	best_feats = kb.fit_transform(images, labels)
	choosen_features = kb.get_support(list(range(0, images.shape[1])))
	return best_feats, choosen_features

def precision_recall_f1_score(true_lables, pred_labels):
	confusion_matrix = pair_confusion_matrix(true_lables, pred_labels)

	tp = confusion_matrix[1, 1]
	fn = confusion_matrix[1, 0]
	tn = confusion_matrix[0, 0]
	fp = confusion_matrix[0, 1]

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = 2 * precision * recall / (precision + recall)

	return precision, recall, f1

class Stats():
	def __init__(self, name):
		self.RI = 0
		self.ARI = 0
		self.silhouette = 0
		self.precision = 0
		self.recall = 0
		self.f1 = 0
		self.name = name
		self.stats = []
	
	def setBestStats(self, RI, ARI, silhouette, precision, recall, f1):
		self.stats.append([RI, ARI, silhouette, precision, recall, f1])
		if(precision > self.precision):
			self.RI = RI
			self.ARI = ARI
			self.silhouette = silhouette
			self.precision = precision
			self.recall = recall
			self.f1 = f1

			return True
		return False
	
	@staticmethod
	def computeStats(external_data_labels, method_labeled_imgs, features, method_all_images):
		RI = rand_score(external_data_labels, method_labeled_imgs)
		ARI = adjusted_rand_score(external_data_labels, method_labeled_imgs)
		
		if(len(np.unique(method_all_images)) > 1):
			silhouette = silhouette_score(features, method_all_images)
		else:
			silhouette = 0

		precision, recall, f1 = precision_recall_f1_score(external_data_labels, method_labeled_imgs)
		return RI, ARI, silhouette, precision, recall, f1

	def create_plot(self, var_range, label):
		fig, ax = plt.subplots()
		ax.set_xlabel(label)
		ax.set_title(self.name)
		self.stats = np.array(self.stats)
		var_range = np.array(var_range)
		ax.plot(var_range, self.stats[:,0], '-', linewidth=3, label="RI")
		ax.plot(var_range, self.stats[:,1], '-', linewidth=3, label="ARI")
		ax.plot(var_range, self.stats[:,2], '-', linewidth=3, label="Silhoutte")
		ax.plot(var_range, self.stats[:,3], '-', linewidth=3, label="Precision")
		ax.plot(var_range, self.stats[:,4], '-', linewidth=3, label="Recall")
		ax.plot(var_range, self.stats[:,5], '-', linewidth=3, label="F1")
		lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		fig.tight_layout()
		fig.savefig("results_" + self.name + ".png")

	def print(self):
		print(self.name)
		print("\tRand index: ", self.RI)
		print("\tAdjusted Rand index: ", self.ARI)
		print("\tSilhouette score: ", self.silhouette)
		print("\tPrecision: {0}, Recall: {1}, F1: {2}".format(self.precision, self.recall, self.f1))


def problem(K_BEST_FEATURES):

	if(os.path.exists("features.npy")):
		features = np.load("features.npy")
	else:
		images = getData()
		features = []

		PCA_features = PCA_FE(images)
		features = PCA_features


		TSNE_features = TSNE_FE(images)
		features = np.append(features, TSNE_features, axis=1)

		Isomap_features = Isomap_FE(images)
		features = np.append(features, Isomap_features, axis=1)
		
		np.save("features", features)

	data_labels, labeled_images = getLabels(features)

	# pd_data = pd.DataFrame(data = features)
	# scatter_matrix(pd_data, alpha = 0.5, figsize =(15,10), diagonal = 'kde')

	labeled_images, choosen_features = selectBestFeatures(labeled_images, data_labels, K_BEST_FEATURES)
	features = features[:, choosen_features]

	create_elbow_plot(features)

	# pd_data = pd.DataFrame(data = features)
	# scatter_matrix(pd_data, alpha = 0.5, figsize =(15,10), diagonal = 'kde')
	
	dbscan_stats = Stats("DBSCAN")
	kmeans_stats = Stats("K - Means")
	gmixture_stats = Stats("Gaussian Mixture")
	
	best_eps = 0
	eps_range = range(1000, 2000, 10)
	for eps in eps_range:
		dbscan_labeled_imgs = DBSCAN(eps=eps).fit_predict(labeled_images)
		dbscan_all_imgs = DBSCAN(eps=eps).fit_predict(features)
		
		RI, ARI, silhouette, precision, recall, f1 = Stats.computeStats(data_labels, dbscan_labeled_imgs, features, dbscan_all_imgs)
		if(dbscan_stats.setBestStats(RI, ARI, silhouette, precision, recall, f1)):
			best_eps = eps

	best_k_kmeans = 0
	k_range = range(2, 12, 1)
	for k in k_range:
		kmeans = KMeans(n_clusters=k).fit(features)
		kmeans_labeled_imgs = kmeans.predict(labeled_images)
		kmeans_all_imgs = kmeans.predict(features)
		RI, ARI, silhouette, precision, recall, f1 = Stats.computeStats(data_labels, kmeans_labeled_imgs, features, kmeans_all_imgs)
		if(kmeans_stats.setBestStats(RI, ARI, silhouette, precision, recall, f1)):
			best_k_kmeans = k

	best_k_gmixture = 0
	for k in k_range:
		gmixture = GaussianMixture(n_components=k).fit(features)
		gmixture_labeled_imgs = gmixture.predict(labeled_images)
		gmixture_all_imgs = gmixture.predict(features)
		RI, ARI, silhouette, precision, recall, f1 = Stats.computeStats(data_labels, gmixture_labeled_imgs, features, gmixture_all_imgs)
		if(gmixture_stats.setBestStats(RI, ARI, silhouette, precision, recall, f1)):
			best_k_gmixture = k

	print("Using {0} features in total\n".format(K_BEST_FEATURES))
	dbscan_stats.print()
	print("\tBest Epsilon: ", best_eps)
	print()
	kmeans_stats.print()
	print("\tBest K: ", best_k_kmeans)
	print()
	gmixture_stats.print()
	print("\tBest K: ", best_k_gmixture)

	ids = np.array(list(range (0, 563)))

	dbscan_labels = DBSCAN(eps=best_eps).fit_predict(features)
	kmeans_labels = KMeans(n_clusters=best_k_kmeans).fit_predict(features)
	gaussianMixedRace = GaussianMixture(n_components=best_k_gmixture).fit_predict(features)

	tp2_aux.report_clusters(ids, dbscan_labels, 'report_DBSCAN.html')
	tp2_aux.report_clusters(ids, kmeans_labels, 'report_K - Means.html')
	tp2_aux.report_clusters(ids, gaussianMixedRace, 'report_Gaussian Mixture.html')

	dbscan_stats.create_plot(eps_range, "Epsilon")
	kmeans_stats.create_plot(k_range, "Number of clusters")
	gmixture_stats.create_plot(k_range, "Number of Components")

	return dbscan_stats, kmeans_stats, gmixture_stats


problem(8) # running with the best K_BEST_FEATURES
plt.show()

# stats = []
# feats_range = range(1, 19)
# for k in feats_range:
# 	dbscan_stats, kmeans_stats, gmixture_stats = problem(k)
# 	stats.append([dbscan_stats.precision, kmeans_stats.precision, gmixture_stats.precision])

# stats = np.array(stats)

# plt.show()
# fig, ax = plt.subplots()
# ax.plot(feats_range, stats[:, 0], label = "DBSCAN")
# ax.plot(feats_range, stats[:, 1], label = "K - Means")
# ax.plot(feats_range, stats[:, 2], label = "Gaussian Mixture")
# ax.set_ylabel("Precision")
# ax.set_xlabel("Number of features")
# ax.set_title("Precision vs Number of Features")
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig.tight_layout()
# fig.savefig("Precision_vs_K_Best_Features.png")

