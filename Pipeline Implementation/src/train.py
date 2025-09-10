import os
import hydra
from omegaconf import DictConfig
import mlflow
import joblib
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from src.data_loader import load_dataset
from src.preprocess import build_preprocessor
from src.models import get_model
from src.metrics import compute_metrics

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # load data
    X, y = load_dataset(cfg.data.name)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    model = get_model(cfg.model.name, cfg.task, dict(cfg.model.params))

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('estimator', model)
    ])

    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name):
        # log config params
        mlflow.log_params(dict(cfg.model.params))
        mlflow.log_param("dataset", cfg.data.name)
        mlflow.log_param("task", cfg.task)

        # Grid search or direct fit
        if cfg.grid_search.enabled:
            param_grid = {f"estimator__{k}": v for k, v in cfg.grid_search.param_grid.items()}
            gs = GridSearchCV(pipeline, param_grid, cv=cfg.grid_search.cv, scoring=cfg.grid_search.scoring, n_jobs=cfg.grid_search.n_jobs, verbose=cfg.grid_search.verbose)
            gs.fit(X, y)
            best = gs.best_estimator_
            mlflow.log_param("best_params", gs.best_params_)
            # cross-validated scores
            scores = cross_val_score(best, X, y, cv=cfg.grid_search.cv, scoring=cfg.grid_search.scoring)
            mlflow.log_metric("cv_mean_score", float(scores.mean()))
            model_to_save = best
        else:
            pipeline.fit(X, y)
            preds = pipeline.predict(X)
            metrics = compute_metrics(y, preds, cfg.task)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            model_to_save = pipeline

        # save model
        model_path = "model.joblib"
        joblib.dump(model_to_save, model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
