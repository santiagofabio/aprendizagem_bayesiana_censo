import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score,confusion_matrix ,ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
import sys
import os
import seaborn as sns
os.system ('cls')



base_census = pd.read_csv('census.csv')

#Imprime os cinco primeiros elementos do cabeçalho
print(f'{base_census.head(5)}')

#Descrição dos dados
print(base_census.describe())


#Mostra  as conlunas com qual vamos trabalhar
print(f'{base_census.columns }')

# Previsores e classe.
x_census =base_census.iloc[:,0:14].values
y_census =base_census.iloc[:,14].values


# LabelEnconder


label_encoder_workclass =LabelEncoder()
label_encoder_education =LabelEncoder()
label_encoder_marital =LabelEncoder()
label_encoder_occupation =LabelEncoder()
label_encoder_relationship =LabelEncoder()
label_encoder_race =LabelEncoder()
label_encoder_sex =LabelEncoder()
label_encoder_country =LabelEncoder()

x_census[:,1] =label_encoder_workclass.fit_transform(x_census[:,1])
x_census[:,3] =label_encoder_education.fit_transform(x_census[:,3])
x_census[:,5] =label_encoder_marital.fit_transform(x_census[:,5])
x_census[:,6] =label_encoder_occupation.fit_transform(x_census[:,6])
x_census[:,7] =label_encoder_relationship.fit_transform(x_census[:,7])
x_census[:,8] =label_encoder_race.fit_transform(x_census[:,8])
x_census[:,9] =label_encoder_sex.fit_transform(x_census[:,9])
x_census[:,13] =label_encoder_country.fit_transform(x_census[:,13q])













# Realização do tratamento dos dados categoricos

"""
with open('census.pkl', 'rb') as f:
      x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste = pickle.load(f)


#Base treinamento
print(f'Base treinamento')
print(f'{x_censo_treinamento.shape}')
print(f'{y_censo_treinamento.shape} ')

#base previsora
print(f'Base previsora')
print(f'{x_censo_teste.shape}')
print(f'{y_censo_teste.shape} ')

naive_census =GaussianNB()
naive_census.fit(x_censo_treinamento, y_censo_treinamento)
previsoes = naive_census.predict(x_censo_teste)

precisao = accuracy_score(y_censo_teste,previsoes)
print(f'Taxa de precisão:  {precisao}')
cm =ConfusionMatrix(naive_census)
cm.fit( x_censo_treinamento, y_censo_treinamento)
score_cm = cm.score(x_censo_teste, y_censo_teste)
print(f'Score Confusion Matriz {score_cm}')
plt.savefig("matriz_de_confusao.tiff", dpi =300, format='tiff') 
cm.show()
"""