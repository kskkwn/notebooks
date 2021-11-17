import torch
import numpy as np
import mlflow
from mlflow.tracking.client import MlflowClient
from omegaconf import DictConfig, ListConfig


class MLflowWriter(object):
    def __init__(self, exp_name, save_dir, log_every, **mlflow_cfg):
        mlflow.set_tracking_uri(save_dir)
        self.client = MlflowClient(**mlflow_cfg)
        mlflow.set_experiment(exp_name)
        self.experiment_id = self.client.get_experiment_by_name(
            exp_name).experiment_id
        self.run_id = self.client.create_run(self.experiment_id).info.run_id

        self.log_every = log_every
        self.clear()

    def log_params_from_omegaconf(self, params):
        self._explore_recursive("", params)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            iterator = element.items()
        elif isinstance(element, ListConfig):
            iterator = enumerate(element)

        for k, v in iterator:
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                self._explore_recursive(f"{parent_name}{k}.", v)
            else:
                self.client.log_param(
                    self.run_id, f"{parent_name}{k}", v)

    def log_torch_model(self, model, epoch):
        with mlflow.start_run(self.run_id):
            mlflow.pytorch.log_model(model, "model_%04d" % epoch)

    def log_metric(self, key, value, is_training):
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().numpy())

        metric_name = "train/" if is_training else "valid/"
        metric_name += str(key)

        if key in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def next_iteration(self):
        self.iterations += 1
        if self.iterations % self.log_every == 0:
            self.toMlflow(nb_data=self.log_every)

    def toMlflow(self, nb_data=0, step=0):
        for key, value in self.metrics.items():
            self.client.log_metric(
                self.run_id, key,
                np.mean(value[-nb_data:]), step=step)

    def get_mean(self, key, is_training):
        metric_name = "train/" if is_training else "valid/"
        metric_name += str(key)

        return np.mean(self.metrics[metric_name])

    def clear(self):
        self.metrics = {}
        self.iterations = 0

    def log_artifact(self, path):
        self.client.log_artifact(self.run_id, local_path=path)

    def terminate(self):
        self.client.set_terminated(self.run_id)
