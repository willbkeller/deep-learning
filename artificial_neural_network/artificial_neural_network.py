
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Label Encoding Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# One Hot Encoding Country
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Have to apply feature scaling for ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential()
# input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#second hiddenn layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Training ANN
#Compiling ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #categorical_crossentropy loss for non binary networks
ann.fit(x_train, y_train, batch_size = 32, epochs = 25)

#Prediction
print(ann.predict(sc.transform([[1.0, 0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

