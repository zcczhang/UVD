from __future__ import annotations

import numpy as np
import torch

_VOLTRON_IMPORT_ERROR = None
try:
    import voltron
    from voltron import instantiate_extractor, load
except ImportError as e:
    _VOLTRON_IMPORT_ERROR = e

from torchvision import transforms as T

import uvd.utils as U
from uvd.models.preprocessors.base import Preprocessor


AVAILABLE_VOLTRON_MODEL_TYPES = [
    # === Voltron ViT-Small (Sth-Sth) Models ===
    "v-cond",
    "v-dual",
    "v-gen",
    # === Voltron ViT-Base Model ===
    "v-cond-base",
    # === Data-Locked Reproductions ===
    # "r-mvp",
    # "r-r3m-vit",
    # "r-r3m-rn50",
]


class VoltronPreprocessor(Preprocessor):
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
        if _VOLTRON_IMPORT_ERROR is not None:
            raise ImportError(_VOLTRON_IMPORT_ERROR)
        model_type = model_type or "v-cond"
        assert model_type in AVAILABLE_VOLTRON_MODEL_TYPES, (
            model_type,
            AVAILABLE_VOLTRON_MODEL_TYPES,
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
        vcond, preprocess = load(model_type, freeze=True)
        vector_extractor = instantiate_extractor(vcond)()
        self.vector_extractor = vector_extractor.to(self.device)
        preprocess: T.Compose
        normlayer = preprocess.transforms[-1]
        assert isinstance(normlayer, T.Normalize)
        transform = (
            T.Compose([T.Resize(224), normlayer])
            if not self.random_crop
            else T.Compose([T.Resize(232), T.RandomCrop(224), normlayer])
        )
        return vcond.to(self.device), transform

    def _encode_image(self, img_tensors: torch.Tensor) -> torch.FloatTensor:
        with torch.no_grad():
            return self.vector_extractor(self.model(img_tensors, mode="visual"))

    def _encode_text(
        self, text: str | np.ndarray | list | torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, text: str | np.ndarray | list | torch.Tensor) -> torch.Tensor:
        return self.cached_language_embed(text)

    def cached_language_embed(self, text: str):
        if text in self._cached_language_embedding:
            return self._cached_language_embedding[text]
        text_embed = self._encode_text(text)
        self._cached_language_embedding[text] = text_embed
        return text_embed


def sim(tensor1, tensor2, metric: str = "l2", device=None):
    if type(tensor1) == np.ndarray:
        tensor1 = torch.from_numpy(tensor1).to(device)
        tensor2 = torch.from_numpy(tensor2).to(device)
    if metric == "l2":
        d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
    elif metric == "cos":
        tensor1 = tensor1 / tensor1.norm(dim=-1, keepdim=True)
        tensor2 = tensor2 / tensor2.norm(dim=-1, keepdim=True)
        d = torch.nn.CosineSimilarity(-1)(tensor1, tensor2)
    else:
        raise NotImplementedError
    return d


PROMPT_DICT = dict(
    microwave="open the microwave",
    kettle="move the kettle to the top left stove",
    light_switch="turn on the light",
    hinge_cabinet="open the left hinge cabinet",
    slide_cabinet="open the right slide cabinet",
    top_burner="turn on the top left burner",
    bottom_burner="turn on the bottom left burner",
)
PROMPT_DICT.update({k.replace("_", " "): v for k, v in PROMPT_DICT.items()})
