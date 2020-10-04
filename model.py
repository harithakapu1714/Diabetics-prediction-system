import gui
from tkinter import *
# import the libraries needed to read the dataset
import os
import pandas as pd
import numpy as np
dataset= pd.read_csv('dia.csv')
dataset.shape
# Calculate the median value for BMI
median_bmi = dataset['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
dataset['BMI'] = dataset['BMI'].replace(to_replace=0, value=median_bmi)

# Calculate the median value for BloodP
median_bloodp = dataset['BloodPressure'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
dataset['BloodPressure'] = dataset['BloodPressure'].replace(
    to_replace=0, value=median_bloodp)

# Calculate the median value for PlGlcConc
median_plglcconc = dataset['Glucose'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
dataset['Glucose'] = dataset['Glucose'].replace(
    to_replace=0, value=median_plglcconc)

# Calculate the median value for SkinThick
median_skinthick = dataset['SkinThickness'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
dataset['SkinThickness'] = dataset['SkinThickness'].replace(
    to_replace=0, value=median_skinthick)

# Calculate the median value for TwoHourSerIns
median_twohourserins = dataset['Insulin'].median()
# Substitute it in the TwoHourSerIns column of the
# dataset where values are 0
dataset['Insulin'] = dataset['Insulin'].replace(
    to_replace=0, value=median_twohourserins)

X=dataset.drop("Outcome",axis=1)
y=dataset.Outcome

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Import all the algorithms we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import model_selection

models = []
models.append(('LogisticRegre ', LogisticRegression()))
models.append(('Navie bayes ', GaussianNB()))
models.append(('Support vector', SVC(kernel='linear')))
models.append(('Decision Tree ', DecisionTreeRegressor()))
models.append(('RandomForest ', RandomForestClassifier()))


names = []
scores = []
mat=[]

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
    mat.append(confusion_matrix(y_test,y_pred))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores,'confusin matrix': mat})
##print(tr_split)


# Apply a scaler
from sklearn.preprocessing import MinMaxScaler as Scaler
scaler = Scaler()
scaler.fit(x_train)
train_set_scaled = scaler.transform(x_train)
test_set_scaled = scaler.transform(x_test)


# Prepare an array with all the algorithms
models = []
models.append(('LogisticRegre ', LogisticRegression()))
models.append(('Navie bayes ', GaussianNB()))
models.append(('Support vector', SVC(kernel='linear')))
models.append(('Decision Tree ', DecisionTreeRegressor()))
models.append(('RandomForest ', RandomForestClassifier()))


# Prepare the configuration to run the test
seed = 7
results = []
names = []
X = train_set_scaled
Y = y_train

# Every algorithm is tested and results are
# collected and printed
for name, model in models:
    
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #print(kfold)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    #print("cv",cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s :  %f (%f)" % (name, cv_results.mean(), cv_results.std())
    ##print(msg)
    
#Find the best parameters for SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [1.0, 10.0, 50.0],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'shrinking': [True, False],
    'gamma': ['auto', 1, 0.1],
    'coef0': [0.0, 0.1, 0.5]
}

model_svc = SVC()

grid_search = GridSearchCV(model_svc, param_grid, cv=10, scoring='accuracy')
#print(grid_search)
grid_search.fit(train_set_scaled, y_train)


# Create an instance of the algorithm using parameters

# from best_estimator_ property

svc = grid_search.best_estimator_



# Use the whole dataset to train the model

X = np.append(train_set_scaled, test_set_scaled, axis=0)

Y = np.append(y_train,y_test, axis=0)



# Train the model

svc.fit(X, Y)

#defining a funtion for prediction of diabetics
def predicting():
    p=float(gui.preg.get())
    g=float(gui.glucose.get())
    bp=float(gui.bp.get())
    s=float(gui.skin.get())
    i=float(gui.ins.get())
    b=float(gui.bmi.get())
    p=float(gui.pedi.get())
    a=float(gui.age.get())
    report=pd.DataFrame([[p,g,bp,s,i,b,p,a]])
    new_df_scaled = scaler.transform(report)
    prediction = svc.predict(new_df_scaled)
    print(prediction)
    if prediction==1:
        result=Label(gui.window,text="Tested_Positive",font=("arial",20,"bold"),bg="red")
        result.place(x=100,y=400)
    else:
        result=Label(gui.window,text="Tested_Negtive",font=("arial",20,"bold"),bg="green yellow")
        result.place(x=100,y=400)



