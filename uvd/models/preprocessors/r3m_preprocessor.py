from __future__ import annotations

_R3M_IMPORT_ERROR = None
try:
    import r3m
except ImportError as e:
    _R3M_IMPORT_ERROR = e

import torch
from torchvision import transforms as T

import uvd.utils as U
from uvd.models.preprocessors.base import Preprocessor


class R3MPreprocessor(Preprocessor):
    def __init__(
        self,
        model_type: str = "resnet50",
        device: torch.device | str | None = None,
        random_crop: bool = False,
        remove_bn: bool = False,
        bn_to_gn: bool = False,
        remove_pool: bool = False,
        **kwargs,
    ):
        if _R3M_IMPORT_ERROR is not None:
            raise ImportError(_R3M_IMPORT_ERROR)
        self.random_crop = random_crop
        kwargs.pop("preprocess_with_fc", None)
        kwargs.pop("save_fc", None)
        super().__init__(
            model_type=model_type,
            device=device,
            remove_bn=remove_bn,
            bn_to_gn=bn_to_gn,
            remove_pool=remove_pool,
            preprocess_with_fc=False,
            save_fc=False,
            **kwargs,
        )

    def _get_model_and_transform(self, model_type: str) -> tuple[r3m.R3M, T.Compose]:
        r3m.device = self.device
        r3m_: r3m.R3M = r3m.load_r3m(modelid=model_type).module.to(self.device)
        model = r3m_.convnet
        if self.remove_pool:
            self._pool = U.freeze_module(model.avgpool)
            model = torch.nn.Sequential(*(list(model.children())[:-2]))
        transform = (
            T.Compose([T.Resize(224), r3m_.normlayer])
            if not self.random_crop
            else T.Compose([T.Resize(232), T.RandomCrop(224), r3m_.normlayer])
        )
        return model, transform
