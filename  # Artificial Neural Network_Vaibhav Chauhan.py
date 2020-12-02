#final Artificial Neural Network 


#Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf




#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values



#Encoding categorical data
#encoding gender column 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])



#One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')				#1 here is the index on which we have to do one hot encoding
X = np.array(ct.fit_transform(X))




#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




#Feature Scaling																					#absolutely compulsory on deep learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





### Building the ANN
#Initializing the ANN
ann = tf.keras.models.Sequential()																	#sequential is a class && models is a module && keras(lib) is now integrated in tf




#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))													#dense class has 6 neurons(units),is used to add layer && in hidden layer activation fn is rectifier(relu)




#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))




#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))												#in output we use sigmoid as we have 2 categories if there were multiple cat. We would have used softmax	





#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])						#Adam does stochastastic gradient descent(single &fast& finds global min) and update the weights after comparing 
																							#the predictions with the real answers and since we have to predict 2 categories loss= binary if more than 2 categories 
																								#loss fn=categorical_crossentropy loss (and above layer softmax) 




#Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)													#comparing 32 predictions with real answers at a time && epochs=100




#Predicting the result of a single observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)							#just transform not fit transform && take care of categorical data && double brackets && >0.5 to output 
																								#answer inn yes or not …. As predict will return probability 



#Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))							#to display predicted and actual result side by side .



#Making the Confusion Matrix & accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)















