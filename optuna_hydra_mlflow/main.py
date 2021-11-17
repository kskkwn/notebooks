import hydra

from omegaconf import OmegaConf
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from ohm import models
from ohm.data import get_dataloader
from ohm.writer import MLflowWriter
from ohm.trainer import SupervisedTrainer


@hydra.main(config_name="base", config_path="parameters")
def main(cfg):
    save_dir = hydra.utils.get_original_cwd() + "/mlruns"

    writer = MLflowWriter(save_dir=save_dir, **cfg.writer)
    writer.log_params_from_omegaconf(cfg)
    print(OmegaConf.to_yaml(cfg))

    model_class = getattr(models, cfg.model.name)
    model = model_class(**cfg.model).to(cfg.device)

    train_loader, valid_loader = get_dataloader(**cfg.data)
    optimizer = Adam(model.parameters(), **cfg.optimizer)

    criterion = CrossEntropyLoss()
    trainer = SupervisedTrainer(writer, model, optimizer, train_loader, valid_loader,
                                criterion,
                                device=cfg.device, **cfg.trainer)
    trainer.train()

    writer.log_artifact("./.hydra/config.yaml")
    writer.log_artifact("./.hydra/hydra.yaml")
    writer.log_artifact("./.hydra/overrides.yaml")
    writer.log_artifact("./main.log")

    writer.terminate()

    return writer.get_mean("loss", is_training=False)


if __name__ == '__main__':
    main()
