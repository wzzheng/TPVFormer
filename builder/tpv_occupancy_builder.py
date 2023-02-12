from tpvformer04 import *
from mmseg.models import build_segmentor

def build(model_config):
    model = build_segmentor(model_config)
    model.init_weights()
    return model

