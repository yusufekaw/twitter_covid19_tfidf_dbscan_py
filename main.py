from data import *
from algoritma.tfidf import *
from algoritma.dbscan import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler
from array import *

data_mentah = csv_ke_array() #mengambil data mentahan
dataset = kata_dasar() #mengambil dataset
jumlah_data = len(data_mentah)



print("\t====================================Dataset============================================")
#menampilkan data mentahan
print("\t***data mentah / data asli***")
for i,dokumen in enumerate(data_mentah):
	print("D",i, dokumen)

#menampilkan dataset
print("\n\t***data hasil komversi sastrawi / mengubah kata kerja menjadi kata dasar***")
for i,dokumen in enumerate(dataset):
	print("D",i, dokumen)

#implementasi dataframe
dataframe = pd.DataFrame({'docs': dataset})
df = df(dataframe)
print("\n\t***daftar kata-kata yang membentuk tokenisasi setiap dokumen dalam korpus***")
print(df[['tokens']]) #menampilkan dataframe

print("\n\n\t=====================================TFIDF=============================================")
#implementasi TF
tf = tf(df)
print("\n\t***TF***")
print(tf) #menampilkan hasil tf

#implementasi idf
idf = idf(tf,df)
print("\n\t***IDF***")
print(idf) #menampilkan hasil idf

#implementasi tfidf
tfidf = tf_idf(tf,idf)
print("\n\t***TFIDF***")
print(tfidf) #menampilkan hasil tfidf

#implementasi normalisasi tfidf
tfidf = normalisasi_tfidf(tfidf)
print("\n\t***Normalisasi TFIDF***")
print(tfidf) #menampilkan hasil normalisasi tfidf

print("\n\n\t==============================DBSCAN Clustering======================================")

#set dataset dbscan
x = np.sum(tf, axis = 1)
y = np.sum(tfidf, axis = 1)

dbscan_data = dbscan_data(jumlah_data,x,y)
print("\n\t***Dataset DBSCAN***")
print(dbscan_data)

#Normalisasi dataset dbscan
dbscan_data_scale = data_scale(dbscan_data)
print("\n\n\t===========================Preprocessing Dataset DBSCAN===================================")
print(dbscan_data_scale)

#ploting data
ploting_data_mentah(dbscan_data_scale)

#pengujian dengan epsilon dan sample secara acak berdasarkan input user
while(True):
	eps = input("epsilon\t\t: ")
	min_samples = input("Min Sample\t: ")

	#proses klasterisasi
	dbscan_cluster = dbscan_cluster(dbscan_data_scale,eps,min_samples)
	dbscan_labels = dbscan_labels(dbscan_cluster)

	#jumlah cluster
	jumlah_cluster = jumlah_cluster(dbscan_labels)
	print("Estimated jumlah cluster : %d" % jumlah_cluster)

	#jumlah noise dalam klaster
	jumlah_noise = jumlah_noise(dbscan_labels)
	print("Estimated jumlah noise : %d" % jumlah_noise)

	#ploting klaster
	plot_cluster(dbscan_labels,dbscan_data_scale,dbscan_cluster,jumlah_cluster)

	#pengujian silhouette coefficient
	silhouette_coefficient=silhouette_coefficient(dbscan_data_scale,dbscan_labels)
	print(f"Silhouette Coefficient: {silhouette_coefficient:.3f}")
