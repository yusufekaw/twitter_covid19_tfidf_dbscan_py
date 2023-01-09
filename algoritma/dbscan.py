import numpy as np
import pandas as pd
from array import *
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

centers = [[1, 1], [-1, -1], [1, -1]]


#menentukan dataset dbscan
def dbscan_data(n,x,y):
    data = []
    for i in range(n):
        data.insert(i, [x[i],y[i]])
    return pd.DataFrame(data, columns=['jumlah_tf','jumlah_tfidf'])

def dbscan_data_dari_csv():
    df = pd.read_csv('./dataset/klastering.csv')    
    return df

def data_scale(data):
    return StandardScaler().fit_transform(data)
    
def ploting_data_mentah(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def dbscan_cluster(data, eps, min_samples):
    return DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(data)

def dbscan_labels(dbscan_cluster):
    return dbscan_cluster.labels_

def jumlah_cluster(dbscan_labels):
    return len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

def jumlah_noise(dbscan_labels):
    return list(dbscan_labels).count(-1)

def silhouette_coefficient(data,dbscan_labels):
    return metrics.silhouette_score(data, dbscan_labels)

def plot_cluster(dbscan_labels,dbscan_data,dbscan_cluster,jumlah_cluster):
    unique_labels = set(dbscan_labels)
    core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
    core_samples_mask[dbscan_cluster.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = dbscan_labels == k

        xy = dbscan_data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

        xy = dbscan_data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    plt.title(f"Estimasi jumlah_cluster : {jumlah_cluster}")
    plt.show()