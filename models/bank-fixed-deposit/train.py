# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import pickle
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from azureml.core.run import Run

os.makedirs('./outputs', exist_ok=True)

script_dir = os.getcwd()
file = 'bankdata.csv'
df = pd.read_csv(os.path.normcase(os.path.join(script_dir,'dataset', file)))

run = Run.get_context()
run.log('csv path is:  ',df)

TARGET_COL = 'term_deposit_subscribed'
features = [c for c in df.columns if c not in [TARGET_COL]]
X = df[features]
y = df[TARGET_COL]

train, test = train_test_split(df, test_size=0.2, random_state = 1, stratify = df[TARGET_COL])
X_train, X_test = train[features], test[features]
y_train, y_test = train[TARGET_COL], test[TARGET_COL]

data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

# list of numbers from 0.0 to 1.0 with a 0.05 interval
model= lgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=10,
              min_child_weight=6, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
model.fit(data["train"]["X"], data["train"]["y"])
y_pred=model.predict(data["test"]["X"])
run.log('f1_score:',f1_score(y_pred, data["test"]["y"]))
run.log('Predicted Value:  ',y_pred)
# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
model_file_name = 'bank_fixed_deposit.pkl'
with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    pickle.dump(model, file)
print('predicted value is {0}'.format(y_pred))
