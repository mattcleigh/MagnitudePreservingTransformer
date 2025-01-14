"""Basic training script."""

import logging

import hydra
import lightning.pytorch as pl
import rootutils
import torch as T
from omegaconf import DictConfig

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from src.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main training script."""
    log.info("Setting up full job config")

    if cfg.full_resume:
        log.info("Attempting to resume previous job")
        old_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)
        if old_cfg is not None:
            cfg = old_cfg
    print_config(cfg)

    log.info(f"Setting seed to: {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    if cfg.weight_ckpt_path:
        log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
        model_class = hydra.utils.get_class(cfg.model._target_)
        model = model_class.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
    else:
        model = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Logging all hyperparameters")
    log_hyperparameters(cfg, model, trainer)
    log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
