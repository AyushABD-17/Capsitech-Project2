from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

def get_model(model_name: str, task: str, model_params: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(**model_params)
    if model_name == "random_forest":
        return (RandomForestClassifier(**model_params) if task=="classification"
                else RandomForestRegressor(**model_params))
    if model_name == "xgboost":
        return (XGBClassifier(use_label_encoder=False, eval_metric='logloss', **model_params)
                if task=="classification"
                else XGBRegressor(**model_params))
    raise ValueError(model_name)
