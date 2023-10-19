from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torchvision.models
from torchvision.models._api import WeightsEnum
from torchvision.transforms._presets import ImageClassification

import uvd.utils as U
from uvd.models.preprocessors.base import Preprocessor


class ResNetPreprocessor(Preprocessor):
    def __init__(
        self,
        model_type: str = "resnet50",
        from_pretrained: bool = True,
        device: torch.device | str | None = None,
        random_crop: bool = False,
        remove_bn: bool = False,
        bn_to_gn: bool = False,
        remove_pool: bool = False,
    ):
        self.random_crop = random_crop
        self.from_pretrained = from_pretrained
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
    ) -> tuple[torch.nn.Module, ImageClassification]:
        model_fn, weights_enum = get_resnet_builder_and_weight(model_type=model_type)
        weights = weights_enum.DEFAULT if self.from_pretrained else None  # type: ignore
        model = model_fn(weights=weights).to(self.device)
        if self.remove_pool:
            self._pool = U.freeze_module(model.avgpool)
            model = torch.nn.Sequential(*(list(model.children())[:-2]))
        transforms = (
            weights.transforms()
            if self.from_pretrained
            else ImageClassification(crop_size=224)
        )
        return model, transforms


def get_resnet_builder_and_weight(
    model_type: str,
) -> tuple[Callable[..., torch.nn.Module], WeightsEnum]:
    models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ]
    assert model_type in models, f"{model_type} not in {models}"
    weights = [
        "ResNet18_Weights",
        "ResNet34_Weights",
        "ResNet50_Weights",
        "ResNet101_Weights",
        "ResNet152_Weights",
        "ResNeXt50_32X4D_Weights",
        "ResNeXt101_32X8D_Weights",
        "ResNeXt101_64X4D_Weights",
        "Wide_ResNet50_2_Weights",
        "Wide_ResNet101_2_Weights",
    ]
    fn = getattr(torchvision.models, model_type)
    weight = getattr(torchvision.models, weights[models.index(model_type)])
    return fn, weight
