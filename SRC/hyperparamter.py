import numpy as np
import pandas as pd
import os
# import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_breast_cancer
import mlflow
import mlflow.sklearn


import dagshub 
dagshub.init(repo_owner='ganeshkarli19', repo_name='MLOPS_MLFLOW', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/ganeshkarli19/MLOPS_MLFLOW.mlflow')


#!config
exp_name='mlflow breast cancer prediction'
model_path='random_forest'
local_file='hyperparameter/model'

data=load_breast_cancer(


)

X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')


# print(X)
# print(y)

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


random=RandomForestClassifier()

xbg=XGBClassifier()


# random.fit(X_train,y_train)

# y_pred=random.predict(x_test)

# print(f"accuracy of random forest is {accuracy_score(y_test,y_pred) }")


model ={


    'random':RandomForestClassifier(n_estimators=500,max_depth=10),
    'xgb':XGBClassifier(learning_rate=1,n_estimators=500,max_depth=9)
}

result= []
for name , mod in model.items():
    print(f"{name}")
    mod.fit(X_train,y_train)

    y_pred=mod.predict(x_test)

    acc=accuracy_score(y_test,y_pred)


    result.append({
        'model':name,
        'accuracy':acc
    })


    print(result)

param={
    'n_estimators':[100,200,300,500],
    'max_depth':[None,10,20]
}

grid=GridSearchCV(estimator=mod,cv=5,param_grid=param,scoring='accuracy',verbose=True)


#^ doing this part on mlflow
# grid.fit(X_train,y_train)


# y_pred=grid.predict(x_test)

# print(f'grid seach cv {accuracy_score(y_test,y_pred)}')


# best_para=grid.best_params_
# # best_est=grid.best_estimator_
# best_score=grid.best_score_

# print(best_para)
# print(best_score)
# # print(best_est)

# mlflow.autolog()

mlflow.set_experiment(exp_name) 
with mlflow.start_run()as parent:
    grid.fit(X_train,y_train)
    best_params=grid.best_params_
    best_score=grid.best_score_


    # #! log all children paramter

    # for i in range(len(grid.cv_results_["params"])):

    #     with mlflow.start_run(nested=True) as child:
    #         mlflow.log_param(grid.cv_results_["params"][i])
    #         mlflow.log_metric(grid.cv_results_["mean_test_score"][i])

    cv = grid.cv_results_
for i in range(len(cv['params'])):

    params_i = cv['params'][i]  # dict

    with mlflow.start_run(nested=True):

        # log all params of this candidate
        mlflow.log_params(params_i)

        # log metrics
        mlflow.log_metric("mean_test_score", float(cv['mean_test_score'][i]))
        mlflow.log_metric("std_test_score", float(cv['std_test_score'][i]))
        mlflow.log_metric("rank_test_score", int(cv['rank_test_score'][i]))

        mlflow.set_tag("combo_id", i)


    mlflow.log_param("parameter",best_params)
    mlflow.log_metric("accuracy",best_score)



    y_pred=grid.predict(x_test)
    acc=accuracy_score(y_test,y_pred)

#     #^ log training data
    train_df=X_train.copy()
    train_df['target']=y_train 


    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df," training data")

    #* same with test data

    test_df=x_test.copy()
    test_df['target']=y_test

    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing data")


    mlflow.log_artifact(__file__)

os.makedirs(local_file,exist_ok=True)

try:

    #!log the best model of grid search cv
    mlflow.sklearn.save_model(grid.best_estimator_,local_file)
    mlflow.log_artifacts(local_file,artifact_path=model)
    print("successfully excute the file ")
    print(best_params)
    print(best_score)
except Exception as e:
    print("failed to save model file ",e)

