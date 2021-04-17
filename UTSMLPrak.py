import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy import stats
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

#Import CSV Data dan Pemilihan Kolom yang Sesuai
data = pd.read_csv(r'C:\Users\Acer\cosmetics_modified.csv')
select = ["Label", "Price", "Combination", "Dry", "Normal", "Oily", "Sensitive"]
data = data[select]
df = pd.DataFrame(data)

#Preprocessing Data, menghilangkan Na dan atau outlier
df=df.dropna()
z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]
print("=============== PREPROCESSING DONE ===============")
print("\n", df, "\n")

#Data Split Training dan Test
select = ["Price", "Combination", "Dry", "Normal", "Oily", "Sensitive"]
x = df[select]
y = df['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
print("=============== SPLITTED DATA ===============")
print()
print(x_train,"==============\n", x_test,"==============\n",y_train,"==============\n", y_test)
print()

#Transformasi Data, melakukan Simple Feature Scaling terhadap data yang dipakai
x_train["Price"] = x_train["Price"] / x_train["Price"].max()
x_test["Price"] = x_test["Price"] / x_test["Price"].max()
print("=============== DATA TRAINING TRANSFORMATION DONE ===============")
print("\n", x_train, "\n")
print("=============== DATA TESTING TRANSFORMATION DONE ===============")
print("\n", x_test, "\n")

#DecisionTree
select1= ['Price', 'Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
Class = ['1','2','3','4','5','6','class']
Class1 = ['1','2','3','4','5','6']
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("=============== HASIL TESTING ===============")
print("\n", y_pred, "\n")
print("Akurasi: ", metrics.accuracy_score(y_test, y_pred), "\n")
fig1 = plt.figure(1)
tree.plot_tree(clf, feature_names=select1 ,class_names=Class)
fig1.show()

#Confusion Matrix
cm = np.array(confusion_matrix(y_test, y_pred))
print("===CONFUSION MATRIX===")
print(cm, "\n")
fig2=plt.figure(2)
sna = sns.heatmap(cm, annot=True, xticklabels= Class1, yticklabels=Class1)
fig2.show()
