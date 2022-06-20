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