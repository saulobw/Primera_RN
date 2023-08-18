import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('diabetes_train.csv')
dataset

dataset.iloc[:, :8].values

X = dataset.iloc[:, :8].values

y = dataset.iloc[:, -1].values
y

Estructura de la RED NEURONAL

model = Sequential()

model.add(Dense(12, activation='relu', input_dim= 8))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

historial= model.fit(X, y, batch_size=10, epochs=2500)

import matplotlib.pyplot as plt

plt.xlabel('#Epocas')
plt.ylabel('Magnitud de Perdida')

plt.plot(historial.history['loss'])

dataset_test= pd.read_csv('diabetes_test.csv')

X_test= dataset_test.iloc[: , :8].values

y_test= dataset_test.iloc[:, -1].values

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred

resultados = pd.DataFrame(np.round(y_pred), columns=['Estimacion'])
resultados

resultados['Real']= pd.Series(y_test)
resultados

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

matriz= confusion_matrix(resultados['Real'], resultados['Estimacion'])

matriz

display= ConfusionMatrixDisplay(matriz)

display.plot()
