import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer



# mlflow.set_tracking_uri("http://127.0.0.1:5000")

import dagshub 
dagshub.init(repo_owner='ganeshkarli19', repo_name='MLOPS_MLFLOW', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/ganeshkarli19/MLOPS_MLFLOW.mlflow')


wine=load_breast_cancer()

X=wine.data

y=wine.target


X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


max_depth=10

n_estimators=100

max_samples=0.5
max_features=0.2
criterion="gini"

#! our own exp manual created 


# mlflow.set_experiment('Mlflow_exp2')

mlflow.autolog()
mlflow.set_experiment("autolog experiment")

with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,max_samples=max_samples,criterion=criterion)
    rf.fit(X_train,y_train)


    y_pred=rf.predict(x_test)

    accuracy =accuracy_score(y_test,y_pred)
    # mlflow.log_metric('accuracy',accuracy)
    # mlflow.log_param('n_estimators',n_estimators)
    # mlflow.log_param('max_depth',max_depth)
    print(accuracy)


    #!print the confusion matrix
    cm=confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,cmap='PuBu_r',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.title("confusion matrix")
    plt.xlabel('prected')
    plt.ylabel('actual')


    #* plt save
    # plt.savefig("confusion_matrix.png")

    #^ log artifacts 
    # mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)


    #! tag
    mlflow.set_tags({"Author":'Ganesh', "Project":"placement prediction"})


    #^ log the model 
    # mlflow.sklearn.log_model(rf,"Random forest model")