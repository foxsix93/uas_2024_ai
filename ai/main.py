#Import library numpy, pandas dan scikit-learn
import numpy as np
import pandas as pd
from sklearn import tree
     

#Membaca Dataset dari File ke Pandas dataFrame
irisDataset = pd.read_csv('balance-scale.csv', delimiter=',', header=0)
     

#Mengubah kelas (kolom "Species") dari String ke Unique-Integer
irisDataset["Class"] = pd.factorize(irisDataset.Class)[0]
     

#Menghapus kolom "Id"
# irisDataset = irisDataset.drop(labels="Id", axis=1)
     

#Mengubah dataFrame ke array Numpy
irisDataset = irisDataset.to_numpy()
     

#Membagi Dataset => 80 baris data untuk training dan 20 baris data untuk testing
# dataTraining = np.concatenate((irisDataset[0:40, :], irisDataset[50:90, :]), 
#                               axis=0)
# dataTesting = np.concatenate((irisDataset[40:50, :], irisDataset[90:100, :]), 
#                              axis=0)
dataTraining = irisDataset
dataTesting = irisDataset
     

#Memecah Dataset ke Input dan Label
inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]
     

#Mendefinisikan Decision Tree Classifier
model = tree.DecisionTreeClassifier()
     

#Mentraining Model
model = model.fit(inputTraining, labelTraining)
     

#Memprediksi Input Data Testing
hasilPrediksi = model.predict(inputTesting)
print("Label Sebenarnya : ", labelTesting)
print("Hasil Prediksi : ", hasilPrediksi)
     

#Menghitung Akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("Prediksi Benar :", prediksiBenar, "data")
print("Prediksi Salah :", prediksiSalah, "data")
print("Akurasi :", prediksiBenar/(prediksiBenar+prediksiSalah) * 100, "%")
     