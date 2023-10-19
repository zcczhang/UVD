from __future__ import annotations

from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch

_LIV_IMPORT_ERROR = None
try:
    import liv
except ImportError as e:
    _LIV_IMPORT_ERROR = e

_CLIP_IMPORT_ERROR = None
try:
    import clip
except ImportError as e:
    _CLIP_IMPORT_ERROR = e

from torch import nn
from torchvision import transforms as T

import uvd.utils as U
from uvd.models.preprocessors.base import Preprocessor


class LIVPreprocessor(Preprocessor):
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
        if _LIV_IMPORT_ERROR is not None:
            raise ImportError(_LIV_IMPORT_ERROR)
        model_type = model_type or "resnet50"
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

    def _get_model_and_transform(
        self, model_type: str | None = None
    ) -> tuple[liv.LIV, Optional[T]]:
        if model_type is not None:
            assert model_type == "resnet50", f"{model_type} not support"
        liv.device = self.device
        liv_ = load_liv(modelid="resnet50", ckpt_path=self.ckpt).module
        clip = liv_.model.to(self.device)
        if self.remove_pool:
            self._pool = U.freeze_module(clip.visual.attnpool)
            self._fc = None
            clip.visual.attnpool = nn.Identity()
        model = clip
        normlayer = liv_.transforms_tensor[-1]
        transform = (
            T.Compose([T.Resize(224), normlayer])
            if not self.random_crop
            else T.Compose([T.Resize(232), T.RandomCrop(224), normlayer])
        )
        return model, transform

    def _encode_image(self, img_tensors: torch.Tensor) -> torch.FloatTensor:
        with torch.no_grad():
            return self.model.encode_image(img_tensors)

    def _encode_text(
        self, text: str | np.ndarray | list | torch.Tensor
    ) -> torch.Tensor:
        if _CLIP_IMPORT_ERROR is not None:
            raise ImportError(_CLIP_IMPORT_ERROR)
        if not torch.is_tensor(text):
            if isinstance(text, str):
                text = [text]
            else:
                assert isinstance(text, (np.ndarray, list)), type(text)
            assert isinstance(text[0], str)
            text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(text)

    def encode_text(self, text: str | np.ndarray | list | torch.Tensor) -> torch.Tensor:
        return self.cached_language_embed(text)

    def cached_language_embed(self, text: str):
        if text in self._cached_language_embedding:
            return self._cached_language_embedding[text]
        text_embed = self._encode_text(text)
        self._cached_language_embedding[text] = text_embed
        return text_embed


def load_liv(modelid: str = "resnet50", ckpt_path: str | None = None):
    if ckpt_path is None:
        return liv.load_liv(modelid)
    home = U.f_join("~/.liv")
    folderpath = U.f_mkdir(home, modelid)
    configpath = U.f_join(home, modelid, "config.yaml")
    if not U.f_exists(configpath):
        try:
            liv.hf_hub_download(
                repo_id="jasonyma/LIV", filename="config.yaml", local_dir=folderpath
            )
        except:
            configurl = (
                "https://drive.google.com/uc?id=1GWA5oSJDuHGB2WEdyZZmkro83FNmtaWl"
            )
            liv.gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = liv.cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    vip_state_dict = torch.load(ckpt_path, map_location="cpu")["vip"]
    rep.load_state_dict(vip_state_dict)
    return rep


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
