from __future__ import annotations

import torch

_VC1_IMPORT_ERROR = None
try:
    import vc_models
    from vc_models.models.vit import model_utils
except ImportError as e:
    _VC1_IMPORT_ERROR = e

from torchvision import transforms as T

import uvd.utils as U
from uvd.models.preprocessors.base import Preprocessor


AVAILABLE_VC1_MODEL_TYPES = ["vc1_vitb", "vc1_vitl"]


class VC1Preprocessor(Preprocessor):
    def __init__(
        self,
        model_type: str | None = None,
        device: torch.device | str | None = None,
        remove_bn: bool = False,
        bn_to_gn: bool = False,
        remove_pool: bool = False,
        preprocess_with_fc: bool = False,
        save_fc: bool = False,
        random_crop: bool = False,
        ckpt: str | None = None,
        use_language_goal: bool = False,
    ):
        if _VC1_IMPORT_ERROR is not None:
            raise ImportError(_VC1_IMPORT_ERROR)
        model_type = model_type or "vc1_vitb"
        assert model_type in AVAILABLE_VC1_MODEL_TYPES, (
            model_type,
            AVAILABLE_VC1_MODEL_TYPES,
        )
        self.random_crop = random_crop
        self.ckpt = ckpt
        if save_fc or preprocess_with_fc:
            U.rank_zero_print(f"WARNING: LIV no fc to save", color="red")
            save_fc = False
            preprocess_with_fc = False
        bn_to_gn = False
        super().__init__(
            model_type=model_type,
            device=device,
            remove_bn=remove_bn,
            bn_to_gn=bn_to_gn,
            remove_pool=remove_pool,
            preprocess_with_fc=preprocess_with_fc,
            save_fc=save_fc,
            use_language_goal=use_language_goal,
        )

        self._cached_language_embedding = {}

    def _get_model_and_transform(self, model_type: str) -> tuple:
        model_utils.download_model_if_needed(model_type + ".pth")
        model, embd_size, model_transforms, model_info = model_utils.load_model(
            model_type
        )
        model = model.to(self.device)
        normlayer = model_transforms.transforms[-1]
        assert isinstance(normlayer, T.Normalize)
        transform = (
            T.Compose([T.Resize(224), normlayer])
            if not self.random_crop
            else T.Compose([T.Resize(232), T.RandomCrop(224), normlayer])
        )
        return model, transform

    def _encode_image(self, img_tensors: torch.Tensor) -> torch.FloatTensor:
        with torch.no_grad():
            return self.model(img_tensors)
