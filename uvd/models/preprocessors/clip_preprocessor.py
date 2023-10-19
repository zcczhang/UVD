from __future__ import annotations

_CLIP_IMPORT_ERROR = None
try:
    import clip
    from clip.model import CLIP
except ImportError as e:
    _CLIP_IMPORT_ERROR = e
import torch

from torchvision import transforms as T

from uvd.models.preprocessors.base import Preprocessor
import uvd.utils as U


class ClipPreprocessor(Preprocessor):
    def __init__(
        self,
        model_type: str = "RN50",
        device: torch.device | str | None = None,
        random_crop: bool = False,
        remove_bn: bool = False,
        bn_to_gn: bool = False,
        remove_pool: bool = False,
        **kwargs,
    ):
        if _CLIP_IMPORT_ERROR is not None:
            raise ImportError(_CLIP_IMPORT_ERROR)
        self.random_crop = random_crop
        model_type = model_type.replace("resnet", "RN")
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

    def _get_model_and_transform(self, model_type: str) -> tuple[CLIP, T.Compose]:
        model, transform = clip.load(model_type, device=self.device)
        model = model.visual
        if self.remove_pool:
            self._pool = U.freeze_module(model.attnpool)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        normlayer = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        transform = (
            T.Compose([T.Resize(224), normlayer])
            if not self.random_crop
            else T.Compose([T.Resize(232), T.RandomCrop(224), normlayer])
        )
        return model, transform
