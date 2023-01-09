from data import *
from algoritma.tfidf import *

datasastrawi = data_sastrawi() #mendapatakan data sastrawi

#implementasi dataframe
dataframe = pd.DataFrame({'docs': datasastrawi})
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
normalisasitfidf = normalisasi_tfidf(tfidf)
print("\n\t***Normalisasi TFIDF***")
print(normalisasitfidf) #menampilkan hasil normalisasi tfidf

#simpan ke excel untuk dokumentasi
df_tf = pd.DataFrame(tf)
df_idf = pd.DataFrame(idf)
df_tfidf = pd.DataFrame(tfidf)
df_tfidf = pd.DataFrame(tfidf)
df_normalisasi_tfidf = pd.DataFrame(normalisasitfidf)
with pd.ExcelWriter("dataset/preprosesing.xlsx") as writer:
    df['docs'].to_excel(writer, sheet_name="sastrawi")
    df['tokens'].to_excel(writer, sheet_name="token")
    df_tf.to_excel(writer, sheet_name="tf")
    df_idf.to_excel(writer, sheet_name="idf")
    df_tfidf.to_excel(writer, sheet_name="tfidf")
    df_normalisasi_tfidf.to_excel(writer, sheet_name="normalisasi_tfidf")

x = np.sum(tf, axis = 1)
y = np.sum(normalisasitfidf, axis = 1)

#mendapatkan data klastering baru pada fitur X,Y
#X jumlah tf
#Y jumlah tfidf yg telah dinormalisasi
dataset_klaster = []
for i in range(len(datasastrawi)):
    dataset_klaster.insert(i, [round(x[i],2),round(y[i],2)])
dataset_klaster = pd.DataFrame(dataset_klaster, columns=['X','Y'])
print("Data Untuk Klastering\n")
print(dataset_klaster)
dataset_klaster.to_csv("dataset/klastering.csv",index=False)