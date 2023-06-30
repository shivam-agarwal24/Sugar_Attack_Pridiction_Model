#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from statsmodels.formula.api import logit
import pickle



df=pd.read_csv("diabetes.csv")
# df = df[["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]]
df_imputer = df.copy(deep = True)




X = df_imputer.drop(['Outcome', 'Pregnancies', 'SkinThickness', 'DiabetesPedigreeFunction'], axis = 1)
# print(X.head())
y = df_imputer['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size  = 0.7)



randomForest = RandomForestClassifier(random_state  = 0, n_jobs=-1)
randomForest.fit(X_train, y_train)
predictions_rf = randomForest.predict(X_test)


# print(f1_score(predictions_rf, y_test))


pickle.dump(randomForest, open("model.pkl", "wb"))