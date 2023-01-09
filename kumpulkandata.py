from data import *

data_mentah = csv_ke_array() #mengambil data mentahan
jumlah_data = len(data_mentah)

print("\t====================================Dataset============================================")
#menampilkan data mentahan
print("\t***data mentah / data asli***")
for i,dokumen in enumerate(data_mentah):
	print("D"+str(i)+"\n",dokumen)
print("\ntunggu . . . konversi sastrawi\n")
dataset = kata_dasar() #mengambil dataset
no_index = []
print("\n\t***data hasil komversi sastrawi / mengubah kata kerja menjadi kata dasar***")
for i,dokumen in enumerate(dataset):
    no_index.append(i)
    print("D"+str(i)+"\n",dokumen)

#simpan data ke csv untuk diolah kembali 
dokumen_csv = pd.DataFrame({'Text':dataset})
dokumen_csv.to_csv("dataset/sastrawi.csv",index=False)