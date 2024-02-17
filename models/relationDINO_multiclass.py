from torch import nn

from .deformable_detr_backbone import build_backbone
from .relationDINO import RelationFormerDINO, build_deformable_transformer_dino


class RelationFormerDINOmulti(RelationFormerDINO):
    def __init__(self, encoder, decoder, config):
        super(RelationFormerDINOmulti, self).__init__(encoder, decoder, config)
        self.multi_class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 3)

    def forward(self, samples, seg=True, targets=None):
        hs, out, srcs = super(RelationFormerDINOmulti, self).forward(samples, seg=seg, targets=targets)

        pad_loc = out['dn_meta']['pad_size'] if out['dn_meta'] else 0
        sel_desc = hs[-1][:, pad_loc:, :] # [B, N, D]
        multi_class_embed = self.multi_class_embed(sel_desc)
        out['pts_pred_class'] = multi_class_embed
        return hs, out, srcs


def build_relationformer_dino_multi(config, **kwargs):
    encoder = build_backbone(config)
    decoder = build_deformable_transformer_dino(config)

    model = RelationFormerDINOmulti(
        encoder,
        decoder,
        config,
        **kwargs
    )

    return model
