import pandas as pd
import sklearn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
import os

data = pd.read_csv("C:\\Users\\Dell\\Course\\projects\\CustomerChurn\\Data\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['tenure']*data['MonthlyCharges'])
data['SeniorCitizen'] = data['SeniorCitizen'].astype(object)
data['MultipleLines'] = data['MultipleLines'].replace({'No phone service':'No'})

for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    data[col] = data[col].replace({'No internet service':'No'})
data['Churn'] = data['Churn'].replace({'No':0,'Yes':1})

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
strat_split
train_index, test_index = next(strat_split.split(data, data["Churn"]))

strat_train_set = data.loc[train_index]
strat_test_set = data.loc[test_index]
X_train = strat_train_set.drop(columns = 'Churn')
y_train = strat_train_set['Churn']
X_test = strat_test_set.drop(columns = 'Churn')
y_test = strat_test_set['Churn']


X_train.to_pickle('Data/X_train.pkl')
y_train.to_pickle('Data/y_train.pkl')
X_test.to_pickle('Data/X_test.pkl')
y_test.to_pickle('Data/y_test.pkl')



categorical_cols = strat_train_set.select_dtypes("object").columns.to_list()
cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
cat_model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test))
y_pred = cat_model.predict(X_test)
pd.DataFrame([y_pred,y_test]).T
accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

score = pd.DataFrame([accuracy, recall, roc_auc, precision]).T
score.columns = ['accuracy', 'recall', 'roc_auc', 'precision']

score.to_excel("Data/Score.xlsx")


model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "catboost_model.cbm")
cat_model.save_model(model_path)
