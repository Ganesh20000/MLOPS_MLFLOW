import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine



mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine=load_wine()

X=wine.data

y=wine.target


X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


max_depth=7

n_estimators=7

#! our own exp manual created 


mlflow.set_experiment('Mlflow_exp2')


with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X_train,y_train)


    y_pred=rf.predict(x_test)

    accuracy =accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('max_depth',max_depth)
    print(accuracy)


    #!print the confusion matrix
    cm=confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,cmap='PuBu_r',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.title("confusion matrix")
    plt.xlabel('prected')
    plt.ylabel('actual')


    #* plt save
    plt.savefig("confusion_matrix.png")

    #^ log artifacts 
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)


    #! tag
    mlflow.set_tag({"Author":'Ganesh', "Project":"placement prediction"})


    #^ log the model 
    mlflow.sklearn.load_model(rf,"Random forest model")