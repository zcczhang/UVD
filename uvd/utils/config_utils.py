import contextlib
import functools
import json
import os
import threading
from typing import Literal
from typing import Optional, List, Any, Union, Tuple

import torch
from allenact.utils.system import get_logger
from omegaconf import OmegaConf, DictConfig

__all__ = ["threadsafe", "Config", "Component", "get_file"]


def threadsafe(function):
    """Decorator making sure that the decorated function (e.g. modify cls
    variables) is thread safe."""
    lock = threading.Lock()

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        with lock:
            return function(self, *args, **kwargs)

    return wrapper


class Component:
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "__LOCKED", False)

    @property
    def is_locked(self):
        """Returns True if the config is locked (no key or value updates
        allowed)."""
        return object.__getattribute__(self, "__LOCKED")

    def lock(self):
        object.__setattr__(self, "__LOCKED", True)
        for v in self.__dict__.values():
            if isinstance(v, (type(self), Component)):
                v.lock()

    def unlock(self):
        object.__setattr__(self, "__LOCKED", False)
        for v in self.__dict__.values():
            if isinstance(v, (type(self), Component)):
                v.unlock()

    @contextlib.contextmanager
    def unlocked(self):
        """A context scope for modifying a Config object.

        Within the scope, both keys and values can be updated. Upon
        leaving the scope, the initial level of locking is restored.
        """
        lock_state = object.__getattribute__(self, "__LOCKED")
        self.unlock()
        yield
        if lock_state:
            self.lock()

    def to_dict(self):
        return {
            k: (v.to_dict() if hasattr(v, "to_dict") else v)
            for k, v in self.__dict__.items()
            if not k.startswith("__")
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setattr__(self, key, value):
        if self.is_locked:
            raise ValueError(f"{self.__class__.__name__} is locked")
        super().__setattr__(key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(self.to_dict(), indent=4)})"


class Config(Component):
    """Used for getting experiment cfg from yaml config file.

    An example of usage of yaml config:
        ```
        tag: experiment tag

        general:  # below would be Attributes
          num_processes: num of processes in parallel
          worker_per_device: num of worker per device
          train_gpus : train gpu ids, -1 for cpu, null for all
          validation_gpus: gpu list for validation (null for auto-detect)
          validation_tasks: num of tasks for validation
          visualize: whether visualize in validation
          visualize_test: whether visualize in test
          viz_mode: visualization mode (rgb, rgb_array, debug, or obs for MuJuCo visual)
          viz_class: name(s) for AllenAct visualizer
          viz_fps: visualization fps
          testing_gpus: gpu list for testing (null for auto-detect)
          train/eval/test_dataset_dir: dataset dir for thor tasks

        model_kwargs:
          ...  # kwargs for model, see usage in `ExperimentConfigBase`

        training_pipeline:  # below would be Attributes
          lr: learning rate
          loss_name: name of loss, or overwrite in `training_pipeline()`
          loss_steps: total training steps
          end_lr: end linear rate, 1.0 for constant
          num_steps: num of steps for rollout
          training_setting_kwargs:
            ...  # see `TrainingSettings`
          loss_kwargs:
            ...  # kwargs for loss
        ```

        Default exp cfg values are shown below.
    """

    tag: str = ""
    num_processes: int = 0
    val_processes: int = 1
    test_processes: int = 1
    worker_per_device: int = 1
    train_gpus: Optional[Union[List[int], int]] = None
    num_gpu_use: Optional[int] = None
    headless: bool = True
    train_dataset_dir: Optional[str] = None
    validation_dataset_dir: Optional[str] = None
    test_dataset_dir: Optional[str] = None
    preprocessor_kwargs: Optional[dict] = None
    eval_env: Optional[str] = None
    test_env: Optional[str] = None
    validation_gpus: Optional[List[int]] = None
    validation_tasks: Optional[int] = None
    testing_tasks: Optional[int] = None
    testing_gpus: Optional[List[int]] = None
    visualize: bool = False
    visualize_test: bool = False
    viz_mode: Literal["rgb", "rgb_array", "debug", "obs"] = "rgb"
    viz_resolution: Optional[Tuple[int, int]] = None
    viz_class: Optional[Union[List[str], str]] = "video_viz"
    viz_fps: int = 40
    lr: float = 3e-4
    end_lr: float = 0.0
    lr_scheduler: Literal["linear", "cos"] = "linear"
    lr_scheduler_steps: Optional[int] = None
    loss_name: Optional[str] = "ppo_loss"
    loss_steps: Union[int, list] = int(1e8)
    num_steps: Union[int, list] = 300
    sampler_kwargs: DictConfig = {}
    valid_sampler_kwargs: Optional[DictConfig] = None
    test_sampler_kwargs: Optional[DictConfig] = None
    model_kwargs: DictConfig
    training_setting_kwargs: DictConfig
    loss_kwargs: DictConfig
    # for wandb callback logging
    callback_kwargs: dict

    @threadsafe
    def __init__(self, cfg_path: str):
        super().__init__()
        self._initialized = False
        cfg = OmegaConf.load(cfg_path)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        assert {
            "tag",
            "general",
            "sampler_kwargs",
            "model_kwargs",
            "training_pipeline",
        }.issubset(
            cfg.keys()
        ), "Config yaml should consist of 'tag', 'general', 'model_kwargs', 'training_pipeline'."
        cuda_available = torch.cuda.is_available()
        for key, value in cfg.items():
            if key in ["general", "training_pipeline"]:
                for k, v in value.items():
                    if k == "train_gpus":
                        if v is None:
                            self.num_gpu_use = (
                                torch.cuda.device_count() if cuda_available else 0
                            )
                            self.train_gpus = list(range(self.num_gpu_use)) * self.worker_per_device  # type: ignore
                        else:
                            if isinstance(v, int) and cuda_available:
                                self.train_gpus = [v] if v != -1 else []
                            else:
                                self.train_gpus = list(v) if cuda_available else []
                            self.num_gpu_use = len(self.train_gpus)
                    elif k in ["validation_gpus", "testing_gpus"] and v is None:
                        setattr(
                            Config,
                            k,
                            []
                            if not cuda_available
                            else [torch.cuda.device_count() - 1],
                        )
                    elif k == "loss_steps":
                        if isinstance(v, (int, float)):
                            self.loss_steps = int(v)
                        else:
                            self.loss_steps = list(v)
                    else:
                        setattr(Config, k, v)
            else:
                setattr(Config, key, value)
        self._initialized = True
        self.lock()
        # get_logger().debug(f"initialized {self}")

    def __setattr__(self, key: str, value: Any):
        if self.is_locked:
            raise ValueError(f"{self.__class__.__name__} is locked")
        if hasattr(self, "_initialized") and self._initialized:
            prev_value = getattr(self, key, None)
            if prev_value != value:
                debug_str = f"trying to update the config: cfg.{key}={value} (type: {type(value)})"
                if hasattr(self, key):
                    debug_str += (
                        f" previous value: {getattr(self, key)} (type: {type(value)})"
                    )
                get_logger().debug(debug_str)
        super().__setattr__(key, value)

    def to_dict(self) -> dict:
        dict_cfg = {}
        for key in dir(self):
            value = getattr(self, key)
            if not callable(getattr(self, key)) and not key.startswith("__"):
                if isinstance(value, (type(self), Config)):
                    dict_cfg[key] = value.to_dict()
                elif isinstance(value, (list, tuple)):
                    dict_cfg[key] = type(value)(
                        item.to_dict()
                        if isinstance(item, (type(self), Config))
                        else item
                        for item in value
                    )
                else:
                    dict_cfg[key] = value
        return dict_cfg

    def debug(self):
        """Debug mode for few steps."""
        assert self._initialized
        with self.unlocked():
            self.num_processes = 1
            self.val_processes = 1
            self.validation_tasks = 2
            self.train_gpus = [0] if torch.cuda.is_available() else []
            self.validation_gpus = [0] if torch.cuda.is_available() else []
            self.training_setting_kwargs["num_mini_batch"] = 1
            self.callback_kwargs["wandb_project"] = "test"
            self.callback_kwargs["output_dir"] = "tmp"
            self.loss_steps = 1000


def json_str(data: dict, indent: int = 4) -> str:
    def _serialize(item, level=0):
        if isinstance(item, dict):
            return "\n" + "\n".join(
                [
                    f'{" " * (level + 1) * indent}{k}: {_serialize(v, level + 1)}'
                    for k, v in item.items()
                ]
            )
        elif hasattr(item, "to_dict"):
            return item.to_dict()
        elif hasattr(item, "__class__"):
            if hasattr(item, "__repr__"):
                return item.__repr__()
            else:
                return item.__class__.__name__
        else:
            return item

    return "\n" + "\n".join(
        [f'{" " * indent}{k}: {_serialize(v)}' for k, v in data.items()]
    )


def get_file(f_cur: str, tar_dir: str, tar_name: str):
    """Return abs path of the target file in terms of current file.

    Example: get f_name file abs path in current file.
        - parent_dir
             - tar_dir
                - tar_name
            - f_cur
        or
        - parent_dir
            - tar_dir
                - tar_name
            - cur_dir
                f_cur
    """
    parent_dir = os.path.dirname(os.path.realpath(f_cur))
    tar_dir_path = os.path.join(parent_dir, tar_dir)
    if not os.path.exists(tar_dir_path):
        tar_dir_path = os.path.join(os.path.dirname(parent_dir), tar_dir)
    return os.path.join(tar_dir_path, tar_name)
