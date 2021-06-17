# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:23:05 2021

@author: juans
"""

import pandas as pd
import glob
from pandas import DataFrame
import numpy as np
from typing import Set, Any 
from sklearn.linear_model import LinearRegression
import math

def remove_others(df: DataFrame, columns: Set[Any]):
    cols_total: Set[Any] = set(df.columns)
    diff: Set[Any] = cols_total - columns
    df.drop(diff, axis=1, inplace=True)

path = r'C:\Users\joana\Desktop\INE' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
i=0




for filename in all_files:

    df = pd.read_csv(filename, thousands=r',', index_col=None, header=0)
    
    # Get names of indexes for which column Stock has value No
    indexNames = df[ df['ENTIDAD'] != 11 ].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)
    df = df.rename(columns = {'LISTA': 'TOTAL_LISTA'}, inplace = False)
    if (i==0):
        remove_others(df, {"SECCION", "TOTAL_LISTA"})
        df = df.sort_values(by=['SECCION'])
        df = df.iloc[:3142,:]
    else:
        remove_others(df, {"SECCION", "TOTAL_LISTA"})
        df = df.sort_values(by=['SECCION'])
        remove_others(df, {"TOTAL_LISTA"})
        df = df.iloc[:3142,:]
   

    li.append(df)
    i=1

frame = np.append(li[0], li[1], axis=1)
j =2
for i in range(14):
    frame = np.append(frame, li[j], axis=1)
    j = j+1
    

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
regresion_lineal = LinearRegression()
predicciones = []
for i in range (len(frame)):

    y = frame[i, 1:].reshape(-1,1)  # values converts it into a numpy array
    regresion_lineal.fit(x.reshape(-1,1), y) 
    nuevo_x = np.array([18]) 
    predicciones.append(math.ceil(float(regresion_lineal.predict(nuevo_x.reshape(-1,1)))))
    
del predicciones[0]
casillas = []

for prediccion in predicciones:
    casillas.append(math.ceil(prediccion/750))

print(sum(casillas))
