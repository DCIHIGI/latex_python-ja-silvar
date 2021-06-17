# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:54:55 2021
Codigo base obetnido de: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
Implementacion para nearestcentroid: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
@author: Juan Andres Silva
asd
"""
# Librerias utilizadas
import numpy as np
import pandas as pd
from typing import Set, Any 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


#Funcion para eliminar columnas
def remove_others(df: pd, columns: Set[Any]):
    cols_total: Set[Any] = set(df.columns)
    diff: Set[Any] = cols_total - columns
    df.drop(diff, axis=1, inplace=True)

#Lectura del archivo y edicion de los datos
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
remove_others(df, {"Weight", "Height", "NObeyesdad"})

df=df.replace(to_replace="Insufficient_Weight",value="0")
df=df.replace(to_replace="Normal_Weight",value="1")
df=df.replace(to_replace="Overweight_Level_I",value="2")
df=df.replace(to_replace="Overweight_Level_II",value="2")
df=df.replace(to_replace="Obesity_Type_I",value="3")
df=df.replace(to_replace="Obesity_Type_II",value="4")
df=df.replace(to_replace="Obesity_Type_III",value="5")

#Reescritura de los datos modificados
df.to_csv("asd.csv",index=False)
    

# step size in the mesh
h = .02  

#Declaracion de los clasificadores y sus parámetros
names = ["Nearest Neighbors", "SVC (RBF)"]

classifiers = [
    KNeighborsClassifier(15),
    SVC(kernel="rbf", C=10, gamma =0.1),]



#Lectura del dataset modificado y almacenado en un df
data = []
with open('asd.csv', newline='') as csvfile:
    next(csvfile)
    df = pd.read_csv(csvfile, sep=',', names=["x1", "x2", "y"])
    data.append([df.values[:, 0:2], df.values[:, 2]])
    
data = tuple(data[0])


#Figura donde se incluyen las graficas
figure = plt.figure(figsize=(12,5))

#Punto de prueba para calcular el imc 
my_imc = [[1.7, 50],[1.77,92],[1.6,120],
          [1.62, 60],[1.56,130],[1.6,95],
          [1.75, 110],[1.59,84],[1.69,60],
          [1.72, 76],[1.69,82],[1.75,70],
          [1.67, 50],[1.55,92],[1.77,70]]

print(my_imc)



"""En esta seccion se crean las graficas, primero el input data que son en
este caso los puntos de entrenamiento que son utilizados para calcular 
las fronteras de decision de cada clasificador en el ciclo for"""
X, y = data
X = np.append(X, my_imc, axis = 0)
X = StandardScaler().fit_transform(X)
X12 = X[-len(my_imc): ]

X = (np.delete(X, slice(-(len(my_imc)+1), -1), 0))
X_train, y_train = X , y
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
ax = plt.subplot(1, len(classifiers) + 1, 1)
ax.set_title("Input data")
scatter=ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes")
i = 2   
# iterate over classifiers
for name, clf in zip(names, classifiers):
        ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.8)
      
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        for dot in X12:
            plt.plot(dot[0],dot[1],'wo')

        i += 1

plt.tight_layout()
plt.show()

#Leyenda del significado de cada clase
print("Clase 0 = Insufficient_Weight")
print("Clase 1 = Normal_Weight")
print("Clase 2 = Overweight")
print("Clase 3 = Obesity_Type_I")
print("Clase 4 = Obesity_Type_II")
print("Clase 5 = Obesity_Type_III")
