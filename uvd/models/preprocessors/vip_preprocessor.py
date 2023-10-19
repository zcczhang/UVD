from __future__ import annotations

from typing import Optional

import hydra
import omegaconf
import torch

_VIP_IMPORT_ERROR = None
try:
    import vip
except ImportError as e:
    _VIP_IMPORT_ERROR = e

from torch import nn
from torchvision import transforms as T

import uvd.utils as U
from uvd.models.preprocessors.base import Preprocessor


class VipPreprocessor(Preprocessor):
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
        **kwargs,
    ):
        if _VIP_IMPORT_ERROR is not None:
            raise ImportError(_VIP_IMPORT_ERROR)
        model_type = model_type or "resnet50"
        self.random_crop = random_crop
        self.ckpt = ckpt
        super().__init__(
            model_type=model_type,
            device=device,
            remove_bn=remove_bn,
            bn_to_gn=bn_to_gn,
            remove_pool=remove_pool,
            preprocess_with_fc=preprocess_with_fc,
            save_fc=save_fc,
            **kwargs,
        )

    def _get_model_and_transform(
        self, model_type: str | None = None
    ) -> tuple[vip.VIP, Optional[T]]:
        if model_type is not None:
            assert model_type == "resnet50", f"{model_type} not support"
        vip.device = self.device
        vip_ = load_vip(modelid="resnet50", ckpt_path=self.ckpt).module
        resnet = vip_.convnet.to(self.device)
        if self.remove_pool:
            # if self.save_fc:
            self._pool = U.freeze_module(resnet.avgpool)
            self._fc = U.freeze_module(resnet.fc)
            model = nn.Sequential(*(list(resnet.children())[:-2]))
        else:
            model = resnet
        # crop_transform = T.RandomCrop(224) if self.random_crop else T.CenterCrop(224)
        transform = (
            # nn.Sequential(T.Resize(224), vip_.normlayer)
            T.Compose([T.Resize(224), vip_.normlayer])
            if not self.random_crop
            # else nn.Sequential(T.Resize(232), T.RandomCrop(224), vip_.normlayer)
            else T.Compose([T.Resize(232), T.RandomCrop(224), vip_.normlayer])
        )
        return model, transform


def load_vip(modelid: str = "resnet50", ckpt_path: str | None = None):
    if ckpt_path is None:
        return vip.load_vip(modelid)
    home = U.f_join("~/.vip")
    folderpath = U.f_mkdir(home, modelid)
    configpath = U.f_join(home, modelid, "config.yaml")
    if not U.f_exists(configpath):
        try:
            configurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/config.yaml"
            vip.load_state_dict_from_url(configurl, folderpath)
        except:
            configurl = (
                "https://drive.google.com/uc?id=1XSQE0gYm-djgueo8vwcNgAiYjwS43EG-"
            )
            vip.gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = vip.cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    vip_state_dict = torch.load(ckpt_path, map_location="cpu")["vip"]
    rep.load_state_dict(vip_state_dict)
    return rep
