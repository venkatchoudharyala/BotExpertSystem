import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

#Loading the Data set into the Environment
path="/content/drive/My Drive/MachineLearning/FFBotClassification/FF_ML.csv"
Data=pd.read_csv(path)

#Preprocessing (Transformation)
le = preprocessing.LabelEncoder()

for column in Data.columns:
  print(column)
  temp_new = le.fit_transform(Data[column].astype('category'))
  Data.drop(labels=[column], axis="columns", inplace=True)
  Data[column] = temp_new
  encoded_labels = le.classes_
  encoded_values = range(len(encoded_labels))
  labelMap = dict(zip(encoded_labels, encoded_values))
  print(labelMap)

#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(Data.iloc[:,0:5], Data.EXPERIENCE_AND_ROLE, random_state=0)

# Create MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                      solver='sgd', verbose=10, random_state=1,
                      learning_rate_init=0.01)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(X_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

#Saving the model
path="/content/drive/My Drive/MachineLearning/FFBotClassification/BackProp.pkl"
with open(path,'wb') as file:
  pickle.dump(model,file)
