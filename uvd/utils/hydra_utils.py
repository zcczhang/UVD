import hydra.utils
from omegaconf import DictConfig

__all__ = ["hydra_instantiate"]


def hydra_instantiate(config: DictConfig, *args, **kwargs):
    if "__target__" in config:
        config = config.copy()
        config.update(_target_=config.pop("__target__"))
    return hydra.utils.instantiate(config=config, *args, **kwargs)
