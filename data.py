#pengolahan dataset
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

#konversi csv ke variabel array
def csv_ke_array():
	sumber_data = pd.read_csv("dataset/dataset.csv")
	data_mentah = []
	for i, kalimat in enumerate(sumber_data['Text']):
		data_mentah.append(kalimat)
	return data_mentah

#mengubah kata kerja dalam kalimat menjadi kata dasar
def kata_dasar():
	dataset = []
	rawdata = csv_ke_array()
	for i, dokumen in enumerate(rawdata):
		kata_dasar = stemmer.stem(dokumen)
		dataset.append(kata_dasar)
	return dataset

def data_sastrawi():
	sumber_data = pd.read_csv("dataset/sastrawi.csv")
	data_sastrawi = []
	for i, kalimat in enumerate(sumber_data['Text']):
		data_sastrawi.append(kalimat)
	return data_sastrawi