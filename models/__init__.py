# from .relationformer import build_relationformer
from .relationformer_2D import build_relationformer, build_relation_embed
from .relationDINO import build_relationformer_dino


def build_model(config, **kwargs):
    return build_relationformer(config, **kwargs)


__all__ = ['build_relation_embed', 'build_relationformer_dino']