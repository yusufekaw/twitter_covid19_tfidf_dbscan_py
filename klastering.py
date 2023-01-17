from algoritma.dbscan import *
from openpyxl import load_workbook

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
#pelabelan kelas
label_kelas = dbscan_cluster.fit_predict(dbscan_data_scale)

#kombinasi dataset dengan kelas
df_dbscan_data_scale = pd.DataFrame(dbscan_data_scale, columns=['X','Y'])
df_label_kelas = pd.DataFrame({'Kelas':label_kelas})
hasil_klastering = pd.concat([df_dbscan_data_scale, df_label_kelas], axis=1)
print(hasil_klastering)

#hasil_klastering.to_csv("dataset/hasil/klastering_eps",eps,"_minsample",min_samples,".csv",index=False)
##with pd.ExcelWriter("dataset/hasil/klastering_eps",eps,"_minsample",min_samples,".xlsx") as writer:  
##    df1.to_excel(writer, sheet_name='Sheet_name_1')
    
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

#konversi hasil uji sbg dictionary
hasil_uji = {
    'Jumlah Klaster':[jumlah_cluster],
    'Jumlah Noise':[jumlah_noise],
    'Silhouette Coefficient': [round(silhouette_coefficient,3)]
    }

#konversi hasil uji ke dataframe
df_hasil_uji = pd.DataFrame(hasil_uji, index = [0])
#transpose dataframe
df_hasil_uji_transpose = df_hasil_uji.transpose()

#simpan ke excel untuk dokumentasi
path = "dataset/hasil/klastering.xlsx"
nama_sheet = "eps " + str(eps) + " min_samples " + str(min_samples)
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = "openpyxl")
writer.book = book
hasil_klastering.to_excel(writer, sheet_name = nama_sheet)
df_hasil_uji_transpose.to_excel(writer, sheet_name = nama_sheet, startrow=0, startcol=7)
writer.close()