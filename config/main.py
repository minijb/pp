from .dataset import dataset_cfg
from .model_config import swin_config, convNext_config, twowaydecoder_config
from .train_config import trian

cfg = {
    "dataset" : dataset_cfg,
    "swin" : swin_config,
    "convNext" : convNext_config,
    "twowaydecoder" : twowaydecoder_config,
    "train" : trian
}