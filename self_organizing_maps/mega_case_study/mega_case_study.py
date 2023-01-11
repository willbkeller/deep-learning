
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

# Train SOM
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x, num_iteration=100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i,j in enumerate(x):
    w = som.winner(j)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor =colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]),axis=0)
frauds = sc.inverse_transform(frauds)

customers = dataset.iloc[:,1:].values
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
# input layer and first hidden layer
ann.add(Dense(units=6, activation='relu'))
#output layer
ann.add(Dense(units=1, activation='sigmoid'))

#Training ANN
#Compiling ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #categorical_crossentropy loss for non binary networks
ann.fit(customers, is_fraud, batch_size = 1, epochs = 2)

#Prediction probabilities
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
print(y_pred)