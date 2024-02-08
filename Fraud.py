from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

import streamlit



# def convert_strings_to_floats(input_array):
#     output_array = []
#     o = []
#     for element in input_array:
#         for i in element:
#             converted_float = float(i)
#             output_array.append(converted_float)
#         o.append(output_array)
#         output_array = []
#     return output_array

irisData_URL = "Fraud_Normal_Dataset.csv"
# print(irisData_URL)

# names = ['Trans Closed - Time',	'Voided Line',	'Voided Qty + Final Qty',	'Voided Qty',	'Final Qty',	'Voided Value',	'Voided Item',	'Voided Item Cat',	'First Tendered Item',	'First Tendered Item Cat',	'Final Tendered Qty',	'Final Tendered Amt','Category']
dataset = read_csv(irisData_URL)
# dataset.to_csv("check.csv")

print(dataset.head(10))
print(dataset.describe())
dataset.dropna(inplace=True)
# print(dataset.groupby('Category').size())
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()
dataset.astype('float')
### Split-out validation dataset
array = dataset.values
# array = convert_strings_to_floats(array)
print(array)
X = array[:,0:-1] # Pattern
# print(X)
y = array[:,-1] # Class
# print(y)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, shuffle=True)
# print(X_train)
# print(X_validation)
# #
# print(Y_train)
# print(Y_validation)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)

# X_new = np.array([[2, 8.9]])

# prediction = knn.predict(X_new)
# print(prediction)

print(knn.score(X_validation, Y_validation))

ADPredict = knn.predict(X_validation)
print(classification_report(Y_validation, ADPredict))
print("The percentage of Accuracy =",accuracy_score(Y_validation, ADPredict))
print(confusion_matrix(Y_validation, ADPredict))



### Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
### evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = StratifiedKFold(n_splits=10)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


#https://www.youtube.com/watch?v=IpGxLWOIZy4













	
