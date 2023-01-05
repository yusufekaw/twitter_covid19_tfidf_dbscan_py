import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from data import *

centers = [[1, 1], [-1, -1], [1, -1]]
X = kata_dasar()

X = StandardScaler().fit_transform(X)

print(X)