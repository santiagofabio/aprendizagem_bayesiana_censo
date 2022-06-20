import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score,confusion_matrix ,ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
from aplica_label_enconder import aplica_label_encoder
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
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

x_census = aplica_label_encoder(x_census)


onehotencoder_census = ColumnTransformer( transformers=[('OneHot', OneHotEncoder(), [ 1,3,5,6,7,8,9,13])], remainder= 'passthrough')
x_census = onehotencoder_census.fit_transform(x_census).toarray()



scale_census = StandardScaler()
x_census = scale_census.fit_transform(x_census)


x_censo_treinamento, x_censo_teste,y_censo_treinamento, y_censo_teste = train_test_split(x_census, y_census, test_size =0.15, random_state= 0)

print(f'{x_censo_treinamento.shape}')
print(f'{y_censo_treinamento.shape} ')

with open('census2.pkl', mode= 'wb'  ) as f:
      pickle.dump([x_censo_treinamento, x_censo_teste,y_censo_treinamento, y_censo_teste],f)


# Realização do tratamento dos dados categoricos


#Base treinamento
print(f'Base treinamento')

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
plt.savefig("matriz_de_confusao.png", dpi =300, format='png') 
cm.show()
