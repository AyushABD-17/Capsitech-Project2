from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

def compute_metrics(y_true, y_pred, task="classification"):
    if task == "classification":
        return {"accuracy": accuracy_score(y_true, y_pred)}
    else:
        return {"r2": r2_score(y_true, y_pred), "mse": mean_squared_error(y_true, y_pred)}
