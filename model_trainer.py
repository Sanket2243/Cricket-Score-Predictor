import numpy as np
import pandas as pd
import pickle
from yaml import safe_load
from tqdm import tqdm
from logger import logging
import os

final_df=pickle.load(open('data_transformation.pkl','rb'))

X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])]
,remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',StandardScaler()),
    ('step3',XGBRegressor(n_estimators=1000,learning_rate=0.2,max_depth=12,random_state=1))
])

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
logging.info("model has been trained")
pickle.dump(pipe,open('model.pkl','wb'))
os.system('python app.py')