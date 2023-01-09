from algoritma.dbscan import *

data_dbscan = dbscan_data_dari_csv()
print(data_dbscan)

dbscan_data_scale = data_scale(data_dbscan)
print("\n\n\t===========================Preprocessing Dataset DBSCAN===================================")
print(dbscan_data_scale)

ploting_data_mentah(dbscan_data_scale)

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