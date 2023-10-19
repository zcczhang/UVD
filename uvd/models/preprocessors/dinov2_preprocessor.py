from __future__ import annotations

import torch
from torchvision import transforms as T

from uvd.models.preprocessors.base import Preprocessor
import uvd.utils as U

__all__ = ["DINOv2Preprocessor"]


class DINOv2Preprocessor(Preprocessor):
    def __init__(
        self,
        model_type: str = "vitl14",
        device: torch.device | str | None = None,
        random_crop: bool = False,
        remove_bn: bool = False,
        bn_to_gn: bool = False,
        remove_pool: bool = False,
        **kwargs,
    ):
        self.random_crop = random_crop
        super().__init__(
            model_type=model_type,
            device=device,
            remove_bn=remove_bn,
            bn_to_gn=bn_to_gn,
            remove_pool=remove_pool,
            preprocess_with_fc=False,
            save_fc=False,
        )

    def _get_model_and_transform(
        self, model_type: str
    ) -> tuple[torch.nn.Module, T.Compose]:
        model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_type}")
        model = model.to(device=self.device)
        normlayer = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        transform = (
            T.Compose([T.Resize(224), normlayer])
            if not self.random_crop
            else T.Compose([T.Resize(256), T.RandomCrop(224), normlayer])
        )
        return model, transform
