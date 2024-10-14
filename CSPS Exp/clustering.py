import numpy as np
import hdbscan as hd
import sklearn.cluster as cluster
import sklearn.mixture as mixture
from utils import clustering_metrics

for name in ["mnist_map", "svhn_map", "cifar10_map", "cifar100_map", "mini_imagenet_vtb"]:
    for data in ["in", "out"]:
        if name in ["cifar100_map", "mini_imagenet_vtb"]:
            classes = 100
        else:
            classes = 10

        x = np.load(f"data_npy/{name}_net_{data}.npy")
        x = np.mean(x, axis=1)
        x = np.reshape(x, (x.shape[0], -1))
        y_true = np.load(f"data_npy/{name}_true_cls.npy")
        print(f"Loaded {name}+{data} data of shape: {x.shape} with classes: {classes}")

        """ Kmeans Clustering """
        kmeans = cluster.KMeans(n_clusters=classes, random_state=0, n_init='auto').fit(x)
        y_pred = np.asarray(kmeans.labels_)
        clustering_metrics(y_true, y_pred, "Kmeans Clustering")

        """ Gaussian Mixture Clustering """
        gm = mixture.GaussianMixture(n_components=classes, random_state=0).fit(x)
        y_pred = np.asarray(gm.predict(x))
        clustering_metrics(y_true, y_pred, "Gaussian Mixture Clustering")

        """ Birch Clustering """
        birch = cluster.Birch(n_clusters=classes).fit(x)
        y_pred = np.asarray(birch.predict(x))
        clustering_metrics(y_true, y_pred, "Birch Clustering")

        """ HDBScan Clustering """
        hdbscan = hd.HDBSCAN(min_cluster_size=50, min_samples=1).fit(x)
        y_pred = hdbscan.labels_
        clustering_metrics(y_true, y_pred, "HDBScan Clustering")
