from .base import Preprocessor
from .clip_preprocessor import ClipPreprocessor
from .dinov2_preprocessor import DINOv2Preprocessor
from .liv_preprocessor import LIVPreprocessor
from .r3m_preprocessor import R3MPreprocessor
from .resnet_preprocessor import ResNetPreprocessor
from .vc1_preprocessor import VC1Preprocessor
from .vip_preprocessor import VipPreprocessor
from .voltron_preprocessor import VoltronPreprocessor


def get_preprocessor(name: str, **kwargs) -> Preprocessor:
    if name == "clip":
        return ClipPreprocessor(**kwargs)
    elif name == "r3m":
        return R3MPreprocessor(**kwargs)
    elif name == "resnet":
        return ResNetPreprocessor(**kwargs)
    elif name == "vip":
        return VipPreprocessor(**kwargs)
    elif name == "liv":
        return LIVPreprocessor(**kwargs)
    elif name == "voltron":
        return VoltronPreprocessor(**kwargs)
    elif name == "vc1":
        return VC1Preprocessor(**kwargs)
    elif name == "dinov2":
        return DINOv2Preprocessor(**kwargs)
    else:
        raise NotImplementedError(name)
